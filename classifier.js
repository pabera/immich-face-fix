// Pair classifier for telling two look-alike people apart.
//
// Nearest-neighbour search asks "which face is this closest to?", and for
// look-alikes the answer is dominated by pose and lighting. A model trained on
// just these two people instead learns the direction that separates them.
//
// The labels are the Immich assignments we are trying to fix, so they are noisy.
// Three things stop the model from simply confirming those mistakes:
//   - Every score is out-of-fold: it comes from a model that never saw that
//     face, or any other face from the same day (bursts are near-duplicates).
//   - Faces the first pass strongly disputes are dropped before the final fit.
//   - Classes are balanced, so a person who "captured" the other's faces can't
//     outvote them.
//
// Loaded three ways: <script> tag, Web Worker, and require() in the tests.

const FaceModel = (() => {
  const DEFAULTS = {
    folds: 5,
    iters: 150,
    lambda: 0.005,
    lr: 0.03,
    trustedWeight: 10,    // a human decision outweighs an inherited label
    disputeProb: 0.1,     // drop labels the first pass gives under this probability
    maxPerClass: 3000,    // keeps training a few seconds on huge libraries
    minPerClass: 30,
    minDays: 5,
  };

  function parsePayload(buffer) {
    const headerLen = new DataView(buffer).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLen)));
    return {
      dim: header.dim,
      faces: header.faces.map(([id, assetId, label, ts]) => ({ id, assetId, label, ts })),
      embeddings: new Float32Array(buffer, 4 + headerLen),
      people: header.people,    // gallery payloads only
      cooccur: header.cooccur,  // gallery payloads only
    };
  }

  // Unit-normalise, centre on the pair's mean, and scale coordinates to roughly
  // unit variance. The two people share nearly all of their embedding, so the
  // shared part has to go before the difference between them is visible.
  function prepare(embeddings, dim) {
    const n = embeddings.length / dim;
    const X = new Float32Array(embeddings.length);
    const mean = new Float64Array(dim);
    for (let i = 0; i < n; i++) {
      const o = i * dim;
      let norm = 0;
      for (let j = 0; j < dim; j++) norm += embeddings[o + j] ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < dim; j++) {
        X[o + j] = embeddings[o + j] / norm;
        mean[j] += X[o + j] / n;
      }
    }
    const scale = Math.sqrt(dim);
    for (let i = 0; i < n; i++) {
      const o = i * dim;
      for (let j = 0; j < dim; j++) X[o + j] = (X[o + j] - mean[j]) * scale;
    }
    return X;
  }

  // Weighted L2-regularised logistic regression, full-batch Adam.
  function trainLogistic(X, dim, rows, y, weights, opts) {
    const { iters, lambda, lr } = opts;
    const w = new Float64Array(dim + 1); // last slot is the bias
    const m1 = new Float64Array(dim + 1), m2 = new Float64Array(dim + 1);
    const grad = new Float64Array(dim + 1);
    let total = 0;
    for (const r of rows) total += weights[r];
    for (let t = 1; t <= iters; t++) {
      grad.fill(0);
      for (const r of rows) {
        const o = r * dim;
        let z = w[dim];
        for (let j = 0; j < dim; j++) z += w[j] * X[o + j];
        const err = (1 / (1 + Math.exp(-z)) - y[r]) * weights[r] / total;
        for (let j = 0; j < dim; j++) grad[j] += err * X[o + j];
        grad[dim] += err;
      }
      for (let j = 0; j <= dim; j++) {
        const g = j < dim ? grad[j] + lambda * w[j] : grad[j];
        m1[j] = 0.9 * m1[j] + 0.1 * g;
        m2[j] = 0.999 * m2[j] + 0.001 * g * g;
        const a = m1[j] / (1 - 0.9 ** t), b = m2[j] / (1 - 0.999 ** t);
        w[j] -= lr * a / (Math.sqrt(b) + 1e-8);
      }
    }
    return w;
  }

  // Whole days go to one fold, spread evenly across the timeline.
  function assignFolds(faces, folds) {
    const days = [...new Set(faces.map((f) => Math.floor(f.ts / 86400)))].sort((a, b) => a - b);
    const foldOfDay = new Map(days.map((d, i) => [d, i % folds]));
    return { dayCount: days.length, foldOf: faces.map((f) => foldOfDay.get(Math.floor(f.ts / 86400))) };
  }

  // Two faces in one photo both labelled the same person: one label is certainly
  // wrong and we can't yet say which, so neither is usable for training.
  function conflictedFaces(faces) {
    const seen = new Map();
    faces.forEach((f, i) => {
      if (f.label < 0) return;
      const key = `${f.assetId}/${f.label}`;
      seen.set(key, [...(seen.get(key) || []), i]);
    });
    const out = new Set();
    for (const list of seen.values()) if (list.length > 1) list.forEach((i) => out.add(i));
    return out;
  }

  function subsample(list, max, trusted) {
    if (list.length <= max) return list;
    const keep = list.filter((i) => trusted[i]);
    const rest = list.filter((i) => !trusted[i]);
    const room = Math.max(0, max - keep.length);
    for (let k = 0; k < room; k++) keep.push(rest[Math.floor((k * rest.length) / room)]);
    return keep;
  }

  function crossFit(X, dim, y, baseWeight, isTrusted, foldOf, opts, onFold) {
    const n = y.length;
    const logits = new Float32Array(n);
    for (let fold = 0; fold < opts.folds; fold++) {
      const pos = [], neg = [];
      for (let i = 0; i < n; i++) {
        if (foldOf[i] !== fold && baseWeight[i] > 0) (y[i] ? pos : neg).push(i);
      }
      const posRows = subsample(pos, opts.maxPerClass, isTrusted);
      const negRows = subsample(neg, opts.maxPerClass, isTrusted);
      if (!posRows.length || !negRows.length) throw new Error("a fold has only one class");

      const sum = (rows) => rows.reduce((s, r) => s + baseWeight[r], 0);
      const posScale = 1 / sum(posRows), negScale = 1 / sum(negRows);
      const weights = new Float32Array(n);
      for (const r of posRows) weights[r] = baseWeight[r] * posScale;
      for (const r of negRows) weights[r] = baseWeight[r] * negScale;

      const w = trainLogistic(X, dim, [...posRows, ...negRows], y, weights, opts);
      for (let i = 0; i < n; i++) {
        if (foldOf[i] !== fold) continue;
        const o = i * dim;
        let z = w[dim];
        for (let j = 0; j < dim; j++) z += w[j] * X[o + j];
        logits[i] = z;
      }
      if (onFold) onFold();
    }
    return logits;
  }

  function auc(logits, y, include) {
    const rows = [];
    for (let i = 0; i < y.length; i++) if (include[i]) rows.push(i);
    rows.sort((a, b) => logits[a] - logits[b]);
    let rank = 0, posRanks = 0, pos = 0;
    for (const r of rows) {
      rank++;
      if (y[r]) { posRanks += rank; pos++; }
    }
    const neg = rows.length - pos;
    return pos && neg ? (posRanks - (pos * (pos + 1)) / 2) / (pos * neg) : null;
  }

  // labels: 0 = wrong person, 1 = correct person, -1 = unlabelled (scored, never
  // trained on). trusted: faceId -> 0, 1, or -1 (a human said this face is neither
  // person). Logits above 0 favour "correct".
  function fit({ dim, faces, embeddings }, trusted = {}, options = {}, onProgress) {
    const opts = { ...DEFAULTS, ...options };
    const n = faces.length;
    const { dayCount, foldOf } = assignFolds(faces, opts.folds);
    const conflicted = conflictedFaces(faces);

    const y = new Uint8Array(n);
    const baseWeight = new Float32Array(n);
    const isTrusted = new Uint8Array(n);
    let conflictedCount = 0;
    faces.forEach((f, i) => {
      const t = trusted[f.id];
      if (t === -1) return; // neither person: never a training example
      if (t === 0 || t === 1) {
        y[i] = t; baseWeight[i] = opts.trustedWeight; isTrusted[i] = 1;
      } else if (f.label < 0) {
        return;
      } else if (conflicted.has(i)) {
        y[i] = f.label; conflictedCount++;
      } else {
        y[i] = f.label; baseWeight[i] = 1;
      }
    });

    const count = (c) => { let k = 0; for (let i = 0; i < n; i++) if (baseWeight[i] > 0 && y[i] === c) k++; return k; };
    if (count(0) < opts.minPerClass || count(1) < opts.minPerClass) {
      return { ok: false, reason: `needs at least ${opts.minPerClass} faces per person` };
    }
    if (dayCount < opts.minDays) return { ok: false, reason: `needs photos from at least ${opts.minDays} different days` };

    const X = prepare(embeddings, dim);
    const totalFolds = opts.folds * 2;
    let done = 0;
    const tick = () => { if (onProgress) onProgress(++done / totalFolds); };

    const firstPass = crossFit(X, dim, y, baseWeight, isTrusted, foldOf, opts, tick);

    const finalWeight = baseWeight.slice();
    const disputeLogit = Math.log(opts.disputeProb / (1 - opts.disputeProb));
    let disputed = 0;
    for (let i = 0; i < n; i++) {
      if (baseWeight[i] === 0 || isTrusted[i]) continue;
      const support = y[i] ? firstPass[i] : -firstPass[i];
      if (support < disputeLogit) { finalWeight[i] = 0; disputed++; }
    }

    const logits = crossFit(X, dim, y, finalWeight, isTrusted, foldOf, opts, tick);
    return {
      ok: true,
      logits,
      auc: auc(logits, y, baseWeight),
      disputed,
      conflicted: conflictedCount,
      faceCount: n,
    };
  }

  // Decide one photo. `faces` are the faces labelled as the wrong person, each
  // with a logit (or null when there is no signal). Returns { fix, keep } or null
  // for "a human should look at this".
  //
  // One person can't appear twice in a photo, so two wrong-labelled faces mean
  // exactly one of them has to move. Comparing them head to head is also a
  // stronger test than two absolute scores, since they share lighting and camera.
  function decidePhoto(faces, correctPresent, threshold) {
    if (faces.some((f) => f.logit === null || f.logit === undefined)) return null;
    if (faces.length === 1) {
      const [f] = faces;
      if (f.logit <= -threshold) return { fix: [], keep: [f] };
      // Moving it would put the correct person in this photo twice.
      if (f.logit >= threshold && !correctPresent) return { fix: [f], keep: [] };
      return null;
    }
    if (faces.length === 2 && !correctPresent) {
      const [lo, hi] = [...faces].sort((a, b) => a.logit - b.logit);
      if (hi.logit > 0 && lo.logit < 0 && hi.logit - lo.logit >= 1.5 * threshold) return { fix: [hi], keep: [lo] };
    }
    return null;
  }

  // Sort a look-alike pair in both directions. Each photo is decided on its own:
  // one person can't appear twice, so two faces are settled head to head, and
  // anything else that isn't clear-cut is left alone. Human decisions are final.
  // Returns moves ({ face, to }), how many faces were already right, and the unsure.
  function planPairFixes(faces, logits, threshold, trusted = {}) {
    const photos = new Map();
    faces.forEach((face, i) => {
      if (trusted[face.id] === -1) return;
      const pinned = trusted[face.id] === 0 || trusted[face.id] === 1;
      const logit = pinned ? (trusted[face.id] ? Infinity : -Infinity) : logits[i];
      const list = photos.get(face.assetId) || [];
      list.push({ face, logit, pinned });
      photos.set(face.assetId, list);
    });

    const moves = [], unsure = [];
    let settled = 0;
    const assign = (item, to) => {
      if (item.pinned) return;
      if (item.face.label === to) settled++;
      else moves.push({ face: item.face, to, logit: item.logit });
    };
    for (const list of photos.values()) {
      if (list.length === 1) {
        const [item] = list;
        if (Math.abs(item.logit) >= threshold) assign(item, item.logit > 0 ? 1 : 0);
        else unsure.push(item.face);
        continue;
      }
      const [lo, hi] = [...list].sort((a, b) => a.logit - b.logit);
      const unlabelledClear = (x) => x.face.label >= 0 || Math.abs(x.logit) >= threshold;
      if (list.length === 2 && hi.logit > 0 && lo.logit < 0 && hi.logit - lo.logit >= 1.5 * threshold
          && unlabelledClear(lo) && unlabelledClear(hi)) {
        assign(hi, 1);
        assign(lo, 0);
      } else {
        for (const item of list) if (!item.pinned) unsure.push(item.face);
      }
    }
    return { moves, settled, unsure };
  }

  // --- Unnamed people ---
  //
  // An unnamed person is a cluster Immich built on its own. Matching it to a named
  // person has two ways to go wrong: the cluster is a stranger who merely resembles
  // someone, or it is the right family but the wrong member. No distance cut-off is
  // right for every library, so it is measured here on the library's own named
  // people, whose answers are known.
  const MATCH = {
    topK: 5,              // a person's score is the mean of their K closest faces
    minFaces: 3,          // smaller clusters are too noisy, and are often a human's unassign
    minMargin: 0.04,      // lead over the second-closest person
    minAgreement: 0.9,    // share of the cluster's own faces that prefer the same person
    reviewSim: 0.45,      // below this a cluster isn't worth showing (Immich's own limit is 0.5)
    maxError: 0.01,       // calibration: tolerated share of wrong merges
    sample: 30,           // faces per cluster used for the agreement check
    trialDays: 3,         // calibration: days held out per person
    trialFaces: 12,       // calibration: largest held-out cluster
    minPersonFaces: 20,   // calibration: people with less can't be held out
    minTrials: 40,
  };

  function buildGallery({ dim, faces, embeddings, people }) {
    const X = new Float32Array(embeddings.length);
    for (let i = 0; i < faces.length; i++) {
      const o = i * dim;
      let norm = 0;
      for (let j = 0; j < dim; j++) norm += embeddings[o + j] ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < dim; j++) X[o + j] = embeddings[o + j] / norm;
    }
    const rowsOf = people.map(() => []);
    faces.forEach((f, i) => rowsOf[f.label].push(i));
    const named = [], unnamed = [];
    people.forEach((p, i) => { if (rowsOf[i].length) (p.named ? named : unnamed).push(i); });
    return { dim, X, faces, rowsOf, named, unnamed };
  }

  function centroid(g, rows) {
    const q = new Float32Array(g.dim);
    for (const r of rows) {
      const o = r * g.dim;
      for (let j = 0; j < g.dim; j++) q[j] += g.X[o + j];
    }
    let norm = 0;
    for (let j = 0; j < g.dim; j++) norm += q[j] ** 2;
    norm = Math.sqrt(norm) || 1;
    for (let j = 0; j < g.dim; j++) q[j] /= norm;
    return q;
  }

  // Averaging a person's closest faces rather than all of them copes with children
  // growing up; averaging several rather than one shrugs off a single mislabel.
  function personSim(g, q, person, hidden) {
    const top = [];
    for (const r of g.rowsOf[person]) {
      if (hidden && hidden[r]) continue;
      const o = r * g.dim;
      let s = 0;
      for (let j = 0; j < g.dim; j++) s += q[j] * g.X[o + j];
      if (top.length < MATCH.topK) top.push(s);
      else {
        const m = top.indexOf(Math.min(...top));
        if (s > top[m]) top[m] = s;
      }
    }
    return top.length ? top.reduce((a, b) => a + b, 0) / top.length : -Infinity;
  }

  function scoreCluster(g, rows, hidden) {
    const q = centroid(g, rows);
    let best = null, second = null, sim = -Infinity, next = -Infinity;
    for (const p of g.named) {
      const s = personSim(g, q, p, hidden);
      if (s > sim) { next = sim; second = best; sim = s; best = p; }
      else if (s > next) { next = s; second = p; }
    }
    if (best === null || sim === -Infinity) return null;
    if (next === -Infinity) second = null;

    // Does the cluster hold together, or is it two people the centroid blends?
    const step = Math.max(1, rows.length / MATCH.sample);
    let votes = 0, total = 0;
    for (let k = 0; k < rows.length; k += step) {
      const face = g.X.subarray(rows[Math.floor(k)] * g.dim, (rows[Math.floor(k)] + 1) * g.dim);
      total++;
      if (second === null || personSim(g, face, best, hidden) >= personSim(g, face, second, hidden)) votes++;
    }
    return { best, second, sim, margin: second === null ? sim : sim - next, agreement: votes / total };
  }

  const passesShape = (m) => m.margin >= MATCH.minMargin && m.agreement >= MATCH.minAgreement;

  // Replays the two failures on named people: hide one day of a person and see if
  // that day finds them again, then hide the person entirely so the "cluster" is a
  // stranger and any match is wrong. The similarity bar goes just above the mistakes.
  function calibrate(g, onProgress) {
    const hidden = new Uint8Array(g.faces.length);
    const held = g.named.filter((p) => g.rowsOf[p].length >= MATCH.minPersonFaces);
    const trials = [];
    held.forEach((p, index) => {
      const byDay = new Map();
      for (const r of g.rowsOf[p]) {
        const day = Math.floor(g.faces[r].ts / 86400);
        byDay.set(day, [...(byDay.get(day) || []), r]);
      }
      const days = [...byDay.values()].filter((rows) => rows.length >= MATCH.minFaces);
      const step = Math.max(1, days.length / MATCH.trialDays);
      for (let k = 0; k < days.length; k += step) {
        const rows = days[Math.floor(k)].slice(0, MATCH.trialFaces);
        days[Math.floor(k)].forEach((r) => { hidden[r] = 1; });
        const known = scoreCluster(g, rows, hidden);
        if (known) trials.push({ ...known, genuine: true, right: known.best === p });
        g.rowsOf[p].forEach((r) => { hidden[r] = 1; });
        const stranger = scoreCluster(g, rows, hidden);
        if (stranger) trials.push({ ...stranger, genuine: false, right: false });
        g.rowsOf[p].forEach((r) => { hidden[r] = 0; });
      }
      if (onProgress) onProgress((index + 1) / held.length);
    });

    if (trials.length < MATCH.minTrials) {
      return { ok: false, reason: `only ${trials.length} test clusters could be built from your named people`, minSim: null };
    }
    const wrong = trials.filter((t) => !t.right && passesShape(t)).map((t) => t.sim).sort((a, b) => b - a);
    const allowed = Math.floor(trials.length * MATCH.maxError);
    const minSim = Math.max(MATCH.reviewSim, wrong.length > allowed ? wrong[allowed] + 1e-4 : MATCH.reviewSim);
    return { ok: true, minSim, ...evaluateRule(trials, minSim), trials };
  }

  function evaluateRule(trials, minSim) {
    const taken = trials.filter((t) => t.sim >= minSim && passesShape(t));
    const genuine = trials.filter((t) => t.genuine).length;
    return {
      tested: trials.length,
      wrongTaken: taken.filter((t) => !t.right).length,
      coverage: genuine ? taken.filter((t) => t.right).length / genuine : 0,
    };
  }

  // Scores every unnamed person. Nothing is decided yet: clusterStatus() does that,
  // so moving the similarity bar needs no rescan.
  function matchClusters(data, onProgress) {
    const g = buildGallery(data);
    const rule = calibrate(g, onProgress && ((p) => onProgress(p * 0.5)));
    const together = new Set((data.cooccur || []).map(([u, p]) => `${u}/${p}`));
    const clusters = g.unnamed.map((person, index) => {
      const rows = g.rowsOf[person];
      const match = scoreCluster(g, rows);
      if (onProgress && index % 25 === 0) onProgress(0.5 + (index / g.unnamed.length) * 0.5);
      return {
        person,
        rows,
        size: rows.length,
        ...match,
        // Same photo as the candidate: they can't be the same person throughout.
        sharesPhoto: !!match && together.has(`${person}/${match.best}`),
        // Same photo twice in one cluster: merging would double someone up.
        selfConflict: new Set(rows.map((r) => g.faces[r].assetId)).size < rows.length,
      };
    });
    return { rule, clusters };
  }

  // "merge", "split" (a look-alike: only per-face sorting is safe), "review", or "none".
  function clusterStatus(c, minSim, lookalikes = new Set()) {
    if (c.best === undefined || c.sim < MATCH.reviewSim) return { status: "none", why: [] };
    const passes = minSim !== null && c.sim >= minSim;
    if (lookalikes.has(c.best) && passes) return { status: "split", why: [] };
    const why = [];
    if (lookalikes.has(c.best)) why.push("look-alike");
    if (!passes) why.push("low similarity");
    if (c.size < MATCH.minFaces) why.push("too few faces");
    if (c.sharesPhoto) why.push("shares a photo with this person");
    if (c.selfConflict) why.push("cluster appears twice in one photo");
    if (c.margin < MATCH.minMargin) why.push("close to another person");
    if (c.agreement < MATCH.minAgreement) why.push("cluster looks mixed");
    return { status: why.length ? "review" : "merge", why };
  }

  // Unattended changes stay locked until a human has watched enough confident
  // recommendations and agreed with nearly all of them.
  const GATE = { minSeen: 20, minAgreement: 0.95 };
  function gateOpen(gate) {
    return gate.seen >= GATE.minSeen && gate.agreed / gate.seen >= GATE.minAgreement;
  }

  const probToLogit = (p) => Math.log(p / (1 - p));
  const logitToProb = (z) => 1 / (1 + Math.exp(-z));

  return { parsePayload, fit, decidePhoto, planPairFixes, matchClusters, clusterStatus, evaluateRule, gateOpen, probToLogit, logitToProb, GATE, MATCH };
})();

if (typeof module !== "undefined") {
  module.exports = FaceModel;
} else if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
  self.onmessage = ({ data }) => {
    try {
      const onProgress = (progress) => self.postMessage({ type: "progress", id: data.id, progress });
      const result = data.task === "match"
        ? FaceModel.matchClusters(data.gallery, onProgress)
        : FaceModel.fit(data.model, data.trusted, undefined, onProgress);
      self.postMessage({ type: "done", id: data.id, result });
    } catch (err) {
      self.postMessage({ type: "error", id: data.id, message: String(err.message || err) });
    }
  };
}
