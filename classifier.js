// Browser port of the scikit-learn pipeline:
// normalise -> TF-IDF(word 1-2gram) + TF-IDF(char_wb 3-5gram), sublinear tf, L2 per block -> logistic regression
function normalise(text) {
  let t = text.toLowerCase();
  t = t.replace(/(https?:\/\/\S+|www\.\S+|\S+\.(com|net|org|co\.uk)\S*)/g, " urltoken ");
  t = t.replace(/\S+@\S+/g, " emailtoken ");
  t = t.replace(/[£$€]\s?\d+(\.\d+)?|\d+(\.\d+)?\s?(p|pence|pounds?)\b/g, " moneytoken ");
  t = t.replace(/\b\d{5,}\b/g, " phonetoken ");
  t = t.replace(/\d+/g, " numtoken ");
  t = t.replace(/[^a-z!? ]/g, " ");
  return t.replace(/\s+/g, " ").trim();
}

function createClassifier(model) {
  const index = {};
  for (const b of ["word", "char"]) {
    const m = new Map();
    model[b].terms.forEach((term, i) => m.set(term, i));
    index[b] = m;
  }

  // Feature extraction that also records which chip (whitespace word) each n-gram came from
  function extract(clean) {
    const chips = clean ? clean.split(" ") : [];
    const word = [], char = [];
    const toks = [];
    chips.forEach((w, ci) => {
      for (const m of w.matchAll(/[a-z]{2,}/g)) toks.push({ t: m[0], ci });
      const p = " " + w + " ";
      for (let n = 3; n <= 5; n++) {
        let off = 0;
        char.push({ g: p.slice(off, off + n), cis: [ci] });
        while (off + n < p.length) { off++; char.push({ g: p.slice(off, off + n), cis: [ci] }); }
        if (off === 0) break;
      }
    });
    toks.forEach(x => word.push({ g: x.t, cis: [x.ci] }));
    for (let i = 0; i + 1 < toks.length; i++)
      word.push({ g: toks[i].t + " " + toks[i + 1].t, cis: [toks[i].ci, toks[i + 1].ci] });
    return { chips, word, char };
  }

  function block(name, grams, chipScores) {
    const idx = index[name], { idf, coef } = model[name];
    const counts = new Map();
    for (const x of grams) {
      const i = idx.get(x.g);
      if (i === undefined) continue;
      if (!counts.has(i)) counts.set(i, []);
      counts.get(i).push(x.cis);
    }
    let norm = 0;
    const vals = [];
    for (const [i, occ] of counts) {
      const v = (1 + Math.log(occ.length)) * idf[i];
      vals.push([i, v, occ]); norm += v * v;
    }
    norm = Math.sqrt(norm) || 1;
    let s = 0;
    for (const [i, v, occ] of vals) {
      const c = coef[i] * v / norm;
      s += c;
      // split the feature's contribution evenly over its occurrences and the chips they span
      for (const cis of occ) for (const ci of cis) chipScores[ci] += c / occ.length / cis.length;
    }
    return s;
  }

  function predict(text) {
    const clean = normalise(text);
    const { chips, word, char } = extract(clean);
    const chipScores = new Array(chips.length).fill(0);
    const z = model.intercept + block("word", word, chipScores) + block("char", char, chipScores);
    const p = 1 / (1 + Math.exp(-z));
    return { p, z, clean, chips: chips.map((c, i) => ({ text: c, score: chipScores[i] })) };
  }
  return { predict, normalise };
}

if (typeof module !== "undefined") module.exports = { createClassifier, normalise };
