// Run with: node --test test/
const test = require("node:test");
const assert = require("node:assert/strict");
const FaceModel = require("../classifier.js");
const { makeLibrary, makeGallery, featurePhotoBaseline } = require("./synthetic.js");

const T = FaceModel.probToLogit(0.97);
const FAST = { iters: 80 };

function accuracy(lib, predict) {
  return lib.faces.filter((_, i) => predict(i) === lib.truth[i]).length / lib.faces.length;
}

test("beats the single feature photo despite lopsided label errors", () => {
  const lib = makeLibrary({ seed: 11, faces: 1200, days: 100 });
  const result = FaceModel.fit(lib, {}, FAST);
  assert.equal(result.ok, true);
  const baseline = featurePhotoBaseline(lib);
  const modelAcc = accuracy(lib, (i) => (result.logits[i] > 0 ? 1 : 0));
  const baseAcc = accuracy(lib, (i) => baseline[i]);
  assert.ok(modelAcc > baseAcc + 0.2, `model ${modelAcc} vs baseline ${baseAcc}`);
});

test("scores are out-of-fold: a mislabelled face isn't confirmed by its own label", () => {
  const lib = makeLibrary({ seed: 12, faces: 1200, days: 100, noiseBtoA: 0.3 });
  const result = FaceModel.fit(lib, {}, FAST);
  const mislabelled = lib.faces.flatMap((f, i) => (f.label !== lib.truth[i] ? [i] : []));
  const caught = mislabelled.filter((i) => (result.logits[i] > 0 ? 1 : 0) === lib.truth[i]).length;
  assert.ok(caught / mislabelled.length > 0.7, `caught ${caught}/${mislabelled.length}`);
  assert.ok(result.disputed > 0);
});

test("stays quiet on faces with no identity signal", () => {
  const lib = makeLibrary({ seed: 13, faces: 1200, days: 100, identityGap: 0, overlap: 0 });
  const result = FaceModel.fit(lib, {}, FAST);
  const confident = result.logits.filter((z) => Math.abs(z) >= T).length;
  assert.ok(confident / lib.faces.length < 0.02, `${confident} confident guesses`);
});

test("human decisions improve the model and override inherited labels", () => {
  const lib = makeLibrary({ seed: 14, faces: 800, days: 80, noiseBtoA: 0.45 });
  const trusted = {};
  lib.faces.slice(0, 150).forEach((f, i) => { trusted[f.id] = lib.truth[i]; });
  const rest = (i) => i >= 150;
  const score = (r) => {
    const rows = lib.faces.flatMap((_, i) => (rest(i) ? [i] : []));
    return rows.filter((i) => (r.logits[i] > 0 ? 1 : 0) === lib.truth[i]).length / rows.length;
  };
  const before = score(FaceModel.fit(lib, {}, FAST));
  const after = score(FaceModel.fit(lib, trusted, FAST));
  assert.ok(after > before, `before ${before}, after ${after}`);
});

test("a face marked as neither person is left out of training", () => {
  const lib = makeLibrary({ seed: 15, faces: 200, days: 30 });
  const trusted = Object.fromEntries(lib.faces.filter((f) => f.label === 1).map((f) => [f.id, -1]));
  const result = FaceModel.fit(lib, trusted, FAST);
  assert.equal(result.ok, false);
});

test("refuses to fit on too little data", () => {
  assert.equal(FaceModel.fit(makeLibrary({ seed: 16, faces: 40, days: 30 }), {}, FAST).ok, false);
  assert.equal(FaceModel.fit(makeLibrary({ seed: 17, faces: 400, days: 3 }), {}, FAST).ok, false);
});

test("same-person duplicates in a photo are excluded from training", () => {
  const lib = makeLibrary({ seed: 18, faces: 400, days: 40 });
  lib.faces[1].assetId = lib.faces[0].assetId;
  lib.faces[1].label = lib.faces[0].label;
  assert.equal(FaceModel.fit(lib, {}, FAST).conflicted, 2);
});

test("parses the server's binary payload", () => {
  const header = new TextEncoder().encode(JSON.stringify({ dim: 2, faces: [["f", "a", 1, 100]] }) + "  ");
  const headerLen = header.length + (4 - (header.length % 4)) % 4;
  const buf = new ArrayBuffer(4 + headerLen + 8);
  new DataView(buf).setUint32(0, headerLen, true);
  new Uint8Array(buf, 4).fill(32, 0, headerLen);
  new Uint8Array(buf, 4).set(header);
  new Float32Array(buf, 4 + headerLen).set([0.5, -0.25]);
  const parsed = FaceModel.parsePayload(buf);
  assert.deepEqual(parsed.faces, [{ id: "f", assetId: "a", label: 1, ts: 100 }]);
  assert.deepEqual([...parsed.embeddings], [0.5, -0.25]);
});

const f = (id, logit) => ({ id, logit });

test("solo face: acts only when confident", () => {
  assert.deepEqual(FaceModel.decidePhoto([f("a", 5)], false, T), { fix: [f("a", 5)], keep: [] });
  assert.deepEqual(FaceModel.decidePhoto([f("a", -5)], false, T), { fix: [], keep: [f("a", -5)] });
  assert.equal(FaceModel.decidePhoto([f("a", 1)], false, T), null);
  assert.equal(FaceModel.decidePhoto([f("a", null)], false, T), null);
});

test("never creates a duplicate of the correct person", () => {
  assert.equal(FaceModel.decidePhoto([f("a", 9)], true, T), null);
  assert.deepEqual(FaceModel.decidePhoto([f("a", -9)], true, T).fix, []);
});

test("two faces: exactly one moves, and only on a clear head-to-head win", () => {
  const clear = FaceModel.decidePhoto([f("a", -3), f("b", 3)], false, T);
  assert.deepEqual(clear.fix.map((x) => x.id), ["b"]);
  assert.deepEqual(clear.keep.map((x) => x.id), ["a"]);
  assert.equal(FaceModel.decidePhoto([f("a", 5), f("b", 6)], false, T), null, "both can't be the correct person");
  assert.equal(FaceModel.decidePhoto([f("a", -5), f("b", -6)], false, T), null, "both can't be the wrong person");
  assert.equal(FaceModel.decidePhoto([f("a", -1), f("b", 1)], false, T), null, "too close to call");
  assert.equal(FaceModel.decidePhoto([f("a", -5), f("b", 5), f("c", 0)], false, T), null, "three faces go to a human");
});

test("the gate needs enough agreement, and reopens only by earning it", () => {
  assert.equal(FaceModel.gateOpen({ seen: 19, agreed: 19 }), false);
  assert.equal(FaceModel.gateOpen({ seen: 20, agreed: 19 }), true);
  assert.equal(FaceModel.gateOpen({ seen: 20, agreed: 18 }), false);
});

test("unlabelled faces are scored but not counted as training data", () => {
  const lib = makeLibrary({ seed: 19, faces: 800, days: 80, noiseBtoA: 0.1 });
  const hidden = lib.faces.map((_, i) => i).filter((i) => i % 8 === 0);
  hidden.forEach((i) => { lib.faces[i].label = -1; });
  lib.faces[0].assetId = lib.faces[8].assetId; // unlabelled faces can't be label conflicts
  const result = FaceModel.fit(lib, {}, FAST);
  assert.equal(result.ok, true);
  assert.equal(result.conflicted, 0);
  const right = hidden.filter((i) => (result.logits[i] > 0 ? 1 : 0) === lib.truth[i]).length;
  assert.ok(right / hidden.length > 0.75, `${right}/${hidden.length} unlabelled faces right`);
});

const pf = (id, assetId, label) => ({ id, assetId, label });
const plan = (faces, logits, trusted) => FaceModel.planPairFixes(faces, logits, T, trusted);

test("pair sort moves faces in both directions", () => {
  const r = plan([pf("a", "1", 0), pf("b", "2", 1), pf("c", "3", 0)], [9, -9, -9]);
  assert.deepEqual(r.moves.map((m) => [m.face.id, m.to]), [["a", 1], ["b", 0]]);
  assert.equal(r.settled, 1);
});

test("pair sort assigns unnamed faces only when confident", () => {
  const r = plan([pf("u", "1", -1), pf("v", "2", -1)], [9, 0.5]);
  assert.deepEqual(r.moves.map((m) => [m.face.id, m.to]), [["u", 1]]);
  assert.deepEqual(r.unsure.map((f) => f.id), ["v"]);
});

test("pair sort never puts one person in a photo twice", () => {
  assert.equal(plan([pf("a", "1", 1), pf("b", "1", 1)], [9, 8]).moves.length, 0, "both look like the same person");
  const swap = plan([pf("a", "1", 1), pf("b", "1", 0)], [-9, 9]);
  assert.deepEqual(swap.moves.map((m) => [m.face.id, m.to]).sort(), [["a", 0], ["b", 1]]);
  const same = plan([pf("a", "1", 1), pf("b", "1", 1)], [-9, 9]);
  assert.deepEqual(same.moves.map((m) => [m.face.id, m.to]), [["a", 0]]);
  assert.equal(plan([pf("a", "1", 1), pf("b", "1", 0), pf("c", "1", -1)], [-9, 9, 9]).moves.length, 0, "three faces go to a human");
});

test("pair sort obeys human decisions", () => {
  assert.equal(plan([pf("a", "1", 0)], [9], { a: 0 }).moves.length, 0, "a kept face stays");
  assert.equal(plan([pf("a", "1", 0)], [9], { a: -1 }).moves.length, 0, "a neither-person face stays");
  const r = plan([pf("a", "1", 1), pf("b", "1", 1)], [2, -9], { a: 1 });
  assert.deepEqual(r.moves.map((m) => [m.face.id, m.to]), [["b", 0]], "a pinned face settles a clear other one");
  const third = plan([pf("a", "1", 1), pf("b", "1", 1)], [2, 1], { a: 1 });
  assert.equal(third.moves.length, 0, "the other face may be a third person, so it needs its own evidence");
  assert.deepEqual(third.unsure.map((f) => f.id), ["b"]);
});

function statuses(lib, match, lookalikes) {
  return lib.clusters.map((c) => {
    const found = match.clusters.find((x) => x.person === c.person);
    return { ...c, found, ...FaceModel.clusterStatus(found, match.rule.minSim, lookalikes) };
  });
}

test("calibration learns a bar that merges known clusters and refuses strangers", () => {
  const lib = makeGallery({ seed: 21 });
  const match = FaceModel.matchClusters(lib);
  assert.equal(match.rule.ok, true);
  const all = statuses(lib, match);
  const known = all.filter((c) => typeof c.truth === "number");
  const merged = known.filter((c) => c.status === "merge");
  assert.ok(merged.length >= known.length * 0.8, `merged ${merged.length}/${known.length}`);
  for (const c of merged) assert.equal(c.found.best, c.truth, "merged into the wrong person");
  assert.equal(all.filter((c) => c.truth === null && c.status === "merge").length, 0, "merged a stranger");
});

test("a cluster that blends two look-alikes is never merged whole", () => {
  const lib = makeGallery({ seed: 22 });
  const match = FaceModel.matchClusters(lib);
  for (const c of statuses(lib, match).filter((x) => x.truth === "mixed")) assert.notEqual(c.status, "merge");
});

test("clusters near the chosen look-alikes are split, not merged", () => {
  const lib = makeGallery({ seed: 23, knownClusters: 0, strangers: 0, mixed: 0 });
  const person = lib.people.length;
  lib.people.push({ id: "u", named: false });
  const dim = lib.dim, source = lib.faces.flatMap((f, i) => (f.label === 0 ? [i] : [])).slice(0, 6);
  const extra = new Float32Array(lib.embeddings.length + source.length * dim);
  extra.set(lib.embeddings);
  source.forEach((r, k) => {
    extra.set(lib.embeddings.subarray(r * dim, (r + 1) * dim), lib.embeddings.length + k * dim);
    lib.faces.push({ id: `x${k}`, assetId: `x${k}`, label: person, ts: 0 });
  });
  const match = FaceModel.matchClusters({ ...lib, embeddings: extra });
  const [c] = match.clusters;
  assert.equal(FaceModel.clusterStatus(c, 0.5, new Set()).status, "merge");
  assert.equal(FaceModel.clusterStatus(c, 0.5, new Set([0, 1])).status, "split");
  assert.equal(FaceModel.clusterStatus(c, 0.99, new Set([0, 1])).status, "review", "below the bar it isn't sorted automatically");
});

test("a shared photo, a doubled face, or a tiny cluster blocks a merge", () => {
  const good = { best: 1, second: 2, sim: 0.8, margin: 0.2, agreement: 1, size: 8, sharesPhoto: false, selfConflict: false };
  const status = (extra) => FaceModel.clusterStatus({ ...good, ...extra }, 0.6).status;
  assert.equal(status({}), "merge");
  assert.equal(status({ sharesPhoto: true }), "review");
  assert.equal(status({ selfConflict: true }), "review");
  assert.equal(status({ size: 2 }), "review");
  assert.equal(status({ margin: 0.01 }), "review");
  assert.equal(status({ agreement: 0.5 }), "review");
  assert.equal(status({ sim: 0.3 }), "none");
  assert.equal(FaceModel.clusterStatus(good, null).status, "review", "without calibration nothing is automatic");
});

test("too few named people means no automatic merges, not a guess", () => {
  const lib = makeGallery({ seed: 24, named: 3, siblings: 0, facesEach: 10, knownClusters: 2, strangers: 1, mixed: 0 });
  const match = FaceModel.matchClusters(lib);
  assert.equal(match.rule.ok, false);
  assert.equal(match.rule.minSim, null);
});
