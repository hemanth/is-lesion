#!/usr/bin/env node
import isLesion, { DEFAULT_HF_DATASET } from '../src/index.js';

const args = process.argv.slice(2);

function printHelp() {
  console.log(`
  is-lesion — Detect lesions in MRI & medical scans via TypeSafe Jev, local OpenJev models & HF choices

  Usage:
    is-lesion <path-or-url> [--engine <id>] [--dataset <hf-repo>]
    is-lesion --hf [dataset-id] [--row <index>] [--engine <id>]
    is-lesion --choices [dataset-id]
    is-lesion --engines

  Engines (--engine):
    jev             TypeSafe Jev 1.14 (+ HF Visual Exemplars) [default]
    jev-text        TypeSafe Jev 1.14 (Text-Only Criteria)
    clip-exemplar   OpenJev Local ONNX · CLIP ViT-B/32 (+ Exemplars)
    clip-zeroshot   OpenJev Local ONNX · CLIP ViT-B/32 (Zero-Shot Text)
    ollama          OpenJev Local VLM via Ollama (--vision-model qwen2.5vl:7b)
    openjev         OpenJev In-Tree Exemplar Fingerprint (0-dep, 1ms)

  Examples:
    is-lesion ./scan.jpg
    is-lesion ./scan.jpg --engine clip-exemplar
    is-lesion ./scan.jpg --engine ollama --vision-model qwen2.5vl:7b
    is-lesion --hf Hemg/Brain-Tumor-MRI-Dataset --row 0 --engine jev
    is-lesion --choices RyanChen1209/Brain-Tumor-Classification-MRI
`);
}

async function main() {
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    printHelp();
    process.exit(0);
  }

  if (args.includes('--engines')) {
    console.log(JSON.stringify(isLesion.engines(), null, 2));
    return;
  }

  const getFlag = (name, fallback) => {
    const idx = args.indexOf(name);
    return idx !== -1 && args[idx + 1] && !args[idx + 1].startsWith('--') ? args[idx + 1] : fallback;
  };

  const dataset = getFlag('--dataset', getFlag('--hf', DEFAULT_HF_DATASET));
  const row = parseInt(getFlag('--row', '0'), 10);
  const engine = getFlag('--engine', undefined);
  const visionModel = getFlag('--vision-model', undefined);
  const jsonOut = args.includes('--json');

  if (args.includes('--choices')) {
    const targetDs = getFlag('--choices', DEFAULT_HF_DATASET);
    const choices = await isLesion.choices(targetDs);
    console.log(JSON.stringify({ dataset: targetDs, choices }, null, 2));
    return;
  }

  let result;
  if (args.includes('--hf')) {
    result = await isLesion.fromHF(dataset, { offset: row, engine, visionModel });
  } else {
    const targetInput = args[0];
    result = await isLesion(targetInput, { dataset, engine, visionModel });
  }

  if (jsonOut) {
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  console.log(`\n  Engine      : ${result.model}`);
  console.log(`  Dataset     : ${result.dataset}`);
  console.log(`  HF Choices  : [${result.choices.join(', ')}]`);
  if (result.groundTruth) {
    console.log(`  Ground Truth: ${result.groundTruth} (lesion=${result.groundTruthLesion})`);
  }
  console.log(`  Lesion      : ${result.lesion ? 'YES' : 'NO'} (probability=${(result.probability * 100).toFixed(1)}%, noul=${(result.noul * 100).toFixed(1)}%)`);
  console.log(`  Top Choice  : ${result.choice} (confidence=${(result.confidence * 100).toFixed(1)}%)`);
  console.log(`  Distribution: ${Object.entries(result.probabilities).map(([k, v]) => `${k}:${(v * 100).toFixed(0)}%`).join(' | ')}`);
  console.log(`  Severity    : ${result.severity.score.toFixed(2)} / ${result.severity.max}`);
  if (result.box) {
    console.log(`  Lesion Box  : top=${result.box.top.toFixed(2)}, left=${result.box.left.toFixed(2)}, bottom=${result.box.bottom.toFixed(2)}, right=${result.box.right.toFixed(2)}`);
  }
  console.log(`  Triage      : ${result.triage} (${result.latencyMs}ms)\n`);
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
