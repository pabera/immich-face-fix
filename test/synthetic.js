// Synthetic look-alike library. Embeddings are shaped like the real problem:
// two identities a small distance apart, swamped by pose/lighting variation that
// both people share, plus a per-day session effect. Each identity's assigned
// label is wrong at a given rate, and `hoard` makes the errors lopsided, the way
// one person "captures" the other's faces in Immich. `overlap` is identity noise
// along the separating direction, `blurry` is the share of faces with no identity
// signal at all, and `drift` rotates the separating direction over the timeline
// as children grow, so a model must not be confident about everything.
function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeLibrary({ seed = 1, dim = 128, faces = 2000, days = 150, identityGap = 0.35, nuisance = 12, noiseAtoB = 0.02, noiseBtoA = 0.3, overlap = 0.15, blurry = 0.15, drift = 0.25 } = {}) {
  const rand = makeRng(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  const vec = () => Float64Array.from({ length: dim }, gauss);
  const unit = (v) => { const n = Math.hypot(...v); return v.map((x) => x / n); };

  const base = unit(vec());
  const gapStart = unit(vec());
  const gapEnd = unit(vec());
  const axes = Array.from({ length: nuisance }, () => unit(vec()));
  const dayShift = Array.from({ length: days }, () => unit(vec()));

  const out = [];
  const embeddings = new Float32Array(faces * dim);
  const truth = new Uint8Array(faces);
  for (let i = 0; i < faces; i++) {
    const person = rand() < 0.5 ? 0 : 1;
    const day = Math.floor(rand() * days);
    const e = base.slice();
    const sign = person ? 1 : -1;
    const age = day / days;
    const gapDir = unit(gapStart.map((x, j) => x * (1 - age * drift) + gapEnd[j] * age * drift));
    const strength = rand() < blurry ? 0 : identityGap * 0.5 * (1 + (gauss() * overlap) / identityGap);
    for (let j = 0; j < dim; j++) e[j] += sign * gapDir[j] * strength + dayShift[day][j] * 0.2;
    for (const axis of axes) {
      const c = gauss() * 0.22;
      for (let j = 0; j < dim; j++) e[j] += axis[j] * c;
    }
    for (let j = 0; j < dim; j++) e[j] += (gauss() * 0.25) / Math.sqrt(dim);
    embeddings.set(e, i * dim);
    truth[i] = person;
    const flip = rand() < (person ? noiseBtoA : noiseAtoB);
    out.push({ id: `f${i}`, assetId: `a${i}`, label: flip ? 1 - person : person, ts: day * 86400 });
  }
  return { dim, faces: out, embeddings, truth };
}

// What the app does without a model: compare to one randomly picked feature
// photo per person, drawn from the (noisy) labels.
function featurePhotoBaseline({ dim, faces, embeddings }, seed = 7) {
  const rand = makeRng(seed);
  const pick = (label) => {
    const list = faces.flatMap((f, i) => (f.label === label ? [i] : []));
    return list[Math.floor(rand() * list.length)];
  };
  const cos = (a, b) => {
    let s = 0, na = 0, nb = 0;
    for (let j = 0; j < dim; j++) {
      const x = embeddings[a * dim + j], y = embeddings[b * dim + j];
      s += x * y; na += x * x; nb += y * y;
    }
    return s / Math.sqrt(na * nb);
  };
  const ref0 = pick(0), ref1 = pick(1);
  return faces.map((_, i) => (cos(i, ref1) > cos(i, ref0) ? 1 : 0));
}

// A whole library for matching unnamed people. Each named person has their own
// embedding centre; `siblings` of them sit close to person 0. Unnamed clusters are
// either more faces of a named person (cluster.truth = person index), a stranger
// nobody named (truth = null), or a blend of two look-alikes (truth = "mixed").
function makeGallery({ seed = 1, dim = 128, named = 12, facesEach = 120, days = 40, siblings = 1, knownClusters = 10, strangers = 10, mixed = 2, clusterSize = 8 } = {}) {
  const rand = makeRng(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  const unit = (v) => { const n = Math.hypot(...v); return v.map((x) => x / n); };
  const vec = () => unit(Float64Array.from({ length: dim }, gauss));
  const centres = Array.from({ length: named }, vec);
  for (let s = 1; s <= siblings; s++) {
    const near = vec();
    centres[s] = unit(centres[0].map((x, j) => x * 0.9 + near[j] * 0.28));
  }

  const people = [], faces = [], vectors = [];
  let photo = 0;
  const addFace = (person, centre, day) => {
    const noise = vec();
    vectors.push(unit(centre.map((x, j) => x + noise[j] * 0.55)));
    faces.push({ id: `f${faces.length}`, assetId: `a${photo++}`, label: person, ts: day * 86400 });
  };
  for (let p = 0; p < named; p++) {
    people.push({ id: `p${p}`, named: true });
    for (let i = 0; i < facesEach; i++) addFace(p, centres[p], i % days);
  }
  const clusters = [];
  const addCluster = (truth, pickCentre) => {
    const person = people.length;
    people.push({ id: `u${person}`, named: false });
    for (let i = 0; i < clusterSize; i++) addFace(person, pickCentre(i), 1000);
    clusters.push({ person, truth });
  };
  for (let k = 0; k < knownClusters; k++) { const t = 2 + siblings + (k % (named - 2 - siblings)); addCluster(t, () => centres[t]); }
  for (let k = 0; k < strangers; k++) { const c = vec(); addCluster(null, () => c); }
  for (let k = 0; k < mixed; k++) addCluster("mixed", (i) => centres[i % 2]);

  const embeddings = new Float32Array(faces.length * dim);
  vectors.forEach((v, i) => embeddings.set(v, i * dim));
  return { dim, faces, embeddings, people, cooccur: [], clusters };
}

module.exports = { makeLibrary, makeGallery, featurePhotoBaseline };
