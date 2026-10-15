import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import path from 'node:path';
import isCancer, { fetchHfDatasetRows } from '../src/index.js';

if (!process.env.TYPESAFE_API_KEY && fsSync.existsSync('.env')) {
  const m = fsSync.readFileSync('.env', 'utf8').match(/TYPESAFE_API_KEY=([^\s]+)/);
  if (m) process.env.TYPESAFE_API_KEY = m[1];
}
const API_KEY = process.env.TYPESAFE_API_KEY;
const DATASET = 'Hemg/Brain-Tumor-MRI-Dataset';
const CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary'];

// N=100 stratified samples (25 per class across glioma, meningioma, notumor, pituitary)
// Excluding the 4 exemplar offsets: 10, 3010, 5300, 6510
function buildStratifiedOffsets() {
  const glioma = [];
  for (let i = 0; glioma.length < 25; i++) {
    if (i !== 10) glioma.push(i);
  }
  const meningioma = [];
  for (let i = 3000; meningioma.length < 25; i++) {
    if (i !== 3010) meningioma.push(i);
  }
  const notumor = [];
  for (let i = 5301; notumor.length < 25; i++) {
    if (i !== 5300) notumor.push(i);
  }
  const pituitary = [];
  for (let i = 6500; pituitary.length < 25; i++) {
    if (i !== 6510) pituitary.push(i);
  }
  return [...glioma, ...meningioma, ...notumor, ...pituitary];
}

function computeClassificationMetrics(records) {
  const N = records.length || 1;
  let tp = 0; // cancer predicted as cancer
  let tn = 0; // notumor predicted as notumor
  let fp = 0; // notumor predicted as cancer
  let fn = 0; // cancer predicted as notumor
  let exactChoice = 0;
  let brierSum = 0;
  let eceBins = Array.from({ length: 5 }, () => ({ count: 0, probSum: 0, accSum: 0 }));

  const perClass = {};
  for (const c of CLASSES) {
    perClass[c] = { total: 0, correct: 0, tp: 0, fp: 0, fn: 0 };
  }

  let highConfTotal = 0;
  let highConfChoiceCorrect = 0;
  let highConfBinaryCorrect = 0;
  let escalatedTotal = 0;
  let escalatedCaughtErrors = 0;
  let totalErrors = 0;

  const latencies = [];

  for (const r of records) {
    latencies.push(r.latencyMs || 0);
    const gt = r.groundTruth;
    const pred = r.choice;
    const gtCancer = r.groundTruthCancer;
    const predCancer = r.cancer;
    const prob = Number(r.probability ?? 0.5);
    const conf = Number(r.confidence ?? 0.5);

    // Binary confusion matrix
    if (gtCancer && predCancer) tp++;
    else if (!gtCancer && !predCancer) tn++;
    else if (!gtCancer && predCancer) fp++;
    else if (gtCancer && !predCancer) fn++;

    // Exact 4-class choice
    const isChoiceRight = pred === gt;
    if (isChoiceRight) exactChoice++;
    else totalErrors++;

    // Per-class recall/precision tracking
    if (perClass[gt]) {
      perClass[gt].total++;
      if (isChoiceRight) perClass[gt].correct++;
      if (isChoiceRight) perClass[gt].tp++;
      else perClass[gt].fn++;
    }
    if (perClass[pred] && pred !== gt) {
      perClass[pred].fp++;
    }

    // Brier score on binary cancer probability: (p - y)^2
    const y = gtCancer ? 1 : 0;
    brierSum += (prob - y) ** 2;

    // 5-bin ECE on top-choice confidence vs correctness
    const binIdx = Math.min(4, Math.floor(conf * 5));
    eceBins[binIdx].count++;
    eceBins[binIdx].probSum += conf;
    eceBins[binIdx].accSum += isChoiceRight ? 1 : 0;

    // Confidence gating (threshold = 0.65)
    if (!r.needsReview) {
      highConfTotal++;
      if (isChoiceRight) highConfChoiceCorrect++;
      if (predCancer === gtCancer) highConfBinaryCorrect++;
    } else {
      escalatedTotal++;
      if (!isChoiceRight) escalatedCaughtErrors++;
    }
  }

  const binaryAcc = (tp + tn) / N;
  const sensitivity = tp / ((tp + fn) || 1); // Recall on tumor
  const specificity = tn / ((tn + fp) || 1); // True negative rate on normal
  const precision = tp / ((tp + fp) || 1);
  const f1 = (2 * precision * sensitivity) / ((precision + sensitivity) || 1);
  const choiceAcc = exactChoice / N;

  // Macro F1 across 4 classes
  let macroF1Sum = 0;
  const perClassRecall = {};
  for (const c of CLASSES) {
    const pc = perClass[c];
    const rec = pc.tp / ((pc.tp + pc.fn) || 1);
    const prec = pc.tp / ((pc.tp + pc.fp) || 1);
    const cf1 = (2 * prec * rec) / ((prec + rec) || 1);
    macroF1Sum += cf1;
    perClassRecall[c] = Number((rec * 100).toFixed(1));
  }
  const macroF1 = macroF1Sum / CLASSES.length;

  // Expected Calibration Error (ECE)
  let ece = 0;
  for (const b of eceBins) {
    if (b.count > 0) {
      const avgConf = b.probSum / b.count;
      const avgAcc = b.accSum / b.count;
      ece += (b.count / N) * Math.abs(avgAcc - avgConf);
    }
  }

  latencies.sort((a, b) => a - b);
  const p50Latency = latencies[Math.floor(latencies.length * 0.5)] || 0;
  const p95Latency = latencies[Math.floor(latencies.length * 0.95)] || 0;
  const avgLatency = Math.round(latencies.reduce((a, b) => a + b, 0) / N);

  return {
    N,
    binaryAcc: Number((binaryAcc * 100).toFixed(1)),
    sensitivity: Number((sensitivity * 100).toFixed(1)),
    specificity: Number((specificity * 100).toFixed(1)),
    precision: Number((precision * 100).toFixed(1)),
    binaryF1: Number((f1 * 100).toFixed(1)),
    choiceAcc: Number((choiceAcc * 100).toFixed(1)),
    macroF1: Number((macroF1 * 100).toFixed(1)),
    perClassRecall,
    brierScore: Number((brierSum / N).toFixed(4)),
    ece: Number((ece * 100).toFixed(1)),
    highConfCoverage: Number(((highConfTotal / N) * 100).toFixed(1)),
    highConfTotal,
    highConfChoiceAcc: highConfTotal > 0 ? Number(((highConfChoiceCorrect / highConfTotal) * 100).toFixed(1)) : null,
    highConfBinaryAcc: highConfTotal > 0 ? Number(((highConfBinaryCorrect / highConfTotal) * 100).toFixed(1)) : null,
    errorCatchRate: totalErrors > 0 ? Number(((escalatedCaughtErrors / totalErrors) * 100).toFixed(1)) : 100.0,
    confusion: { tp, tn, fp, fn },
    latency: { avg: avgLatency, p50: p50Latency, p95: p95Latency }
  };
}

async function runBenchmark() {
  const offsets = buildStratifiedOffsets();
  console.log(`\nRunning N=${offsets.length} Stratified Clinical MRI Benchmark on ${DATASET}...`);

  // Pre-warm row cache in batches of 25 so HF Datasets Server is only queried 4 times total
  console.log('Pre-fetching 4 class splits from Hugging Face Datasets Server...');
  await fetchHfDatasetRows(DATASET, { offset: 0, length: 26 });
  await fetchHfDatasetRows(DATASET, { offset: 3000, length: 26 });
  await fetchHfDatasetRows(DATASET, { offset: 5300, length: 26 });
  await fetchHfDatasetRows(DATASET, { offset: 6500, length: 26 });

  const engines = [
    { id: 'jev', label: 'TypeSafe Jev 1.14 (+ Visual Exemplars)' },
    { id: 'jev-text', label: 'TypeSafe Jev 1.14 (Text-Only Criteria)' },
    { id: 'clip-exemplar', label: 'OpenJev Local ONNX · CLIP ViT-B/32 (+ Exemplars)' },
    { id: 'clip-zeroshot', label: 'OpenJev Local ONNX · CLIP ViT-B/32 (Zero-Shot Text)' },
    { id: 'openjev', label: 'OpenJev In-Tree Exemplar Fingerprint (0-dep)' }
  ];

  const rawRecords = Object.fromEntries(engines.map(e => [e.id, []]));

  // Evaluate in concurrent batches of 4 slices to finish cleanly and quickly
  const batchSize = 4;
  for (let i = 0; i < offsets.length; i += batchSize) {
    const batch = offsets.slice(i, i + batchSize);
    await Promise.all(
      batch.map(async (off) => {
        for (const eng of engines) {
          const res = await isCancer.fromHF(DATASET, {
            row: off,
            engine: eng.id,
            apiKey: API_KEY
          });
          rawRecords[eng.id].push(res);
        }
      })
    );
    console.log(`Completed ${Math.min(i + batchSize, offsets.length)}/${offsets.length} MRI slices across all 5 engines...`);
  }

  const summary = {};
  console.log('\n=========================================================================================================');
  console.log('ENGINE                                           | BIN ACC | SENS  | SPEC  | 4-CLS | MAC-F1 | BRIER  | ECE   | GATED ACC (COV) | P50 MS');
  console.log('---------------------------------------------------------------------------------------------------------');

  for (const eng of engines) {
    const m = computeClassificationMetrics(rawRecords[eng.id]);
    summary[eng.id] = { label: eng.label, ...m };
    const gatedStr = m.highConfChoiceAcc !== null
      ? `${String(m.highConfChoiceAcc).padStart(5)}% (${m.highConfTotal}/${m.N})`
      : '  N/A  (0/100)';
    console.log(
      `${eng.label.padEnd(48)} | ${String(m.binaryAcc).padStart(6)}% | ${String(m.sensitivity).padStart(4)}% | ${String(m.specificity).padStart(4)}% | ${String(m.choiceAcc).padStart(4)}% | ${String(m.macroF1).padStart(5)}% | ${m.brierScore.toFixed(4)} | ${String(m.ece).padStart(4)}% | ${gatedStr.padEnd(15)} | ${String(m.latency.p50).padStart(4)}ms`
    );
  }
  console.log('=========================================================================================================\n');

  const outFile = path.resolve('docs/data/benchmark-results.json');
  await fs.writeFile(outFile, JSON.stringify({
    generatedAt: new Date().toISOString(),
    dataset: DATASET,
    sampleCount: offsets.length,
    perClassCount: 25,
    engines: summary
  }, null, 2));
  console.log(`Saved full benchmark telemetry to ${outFile}`);
}

runBenchmark().catch(err => {
  console.error(err);
  process.exit(1);
});
