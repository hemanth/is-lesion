import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import path from 'node:path';
import isCancer, { DEFAULT_HF_DATASET, fetchHfDatasetRows, loadImageAsset } from '../src/index.js';

if (!process.env.TYPESAFE_API_KEY && fsSync.existsSync('.env')) {
  const m = fsSync.readFileSync('.env', 'utf8').match(/TYPESAFE_API_KEY=([^\s]+)/);
  if (m) process.env.TYPESAFE_API_KEY = m[1];
}
const apiKey = process.env.TYPESAFE_API_KEY;

const SAMPLE_DEFS = [
  { id: 'glioma-01', label: 'glioma', offset: 0, title: 'Axial T1-Weighted — Left Parietal Glioma' },
  { id: 'meningioma-01', label: 'meningioma', offset: 3000, title: 'Coronal T1+C — Right Convexity Meningioma' },
  { id: 'notumor-01', label: 'notumor', offset: 5305, title: 'Axial T2 — Normal Symmetrical Parenchyma' },
  { id: 'pituitary-01', label: 'pituitary', offset: 6500, title: 'Sagittal T1 — Sellar Pituitary Adenoma' }
];

async function seed() {
  console.log('Seeding golden MRI dataset & live TypeSafe Jev evaluations...');
  const samples = [];

  for (const def of SAMPLE_DEFS) {
    console.log(`Evaluating ${def.id} (row ${def.offset})...`);
    const rows = await fetchHfDatasetRows(DEFAULT_HF_DATASET, { offset: def.offset, length: 1 });
    const row = rows[0];
    const imgAsset = await loadImageAsset(row.imageUrl);
    const evalResult = await isCancer(row.imageUrl, {
      apiKey,
      dataset: DEFAULT_HF_DATASET,
      groundTruth: def.label
    });

    const { withDataset, ...serializableResult } = evalResult;
    samples.push({
      id: def.id,
      title: def.title,
      rowIndex: def.offset,
      dataset: DEFAULT_HF_DATASET,
      groundTruth: def.label,
      groundTruthCancer: def.label !== 'notumor',
      imageDataUri: `data:image/${imgAsset.format === 'png' ? 'png' : 'jpeg'};base64,${imgAsset.data_base64}`,
      width: imgAsset.width,
      height: imgAsset.height,
      evaluation: serializableResult
    });
  }

  const outDir = path.resolve('docs/data');
  await fs.mkdir(outDir, { recursive: true });
  const outFile = path.join(outDir, 'golden-samples.json');
  await fs.writeFile(outFile, JSON.stringify({ generatedAt: new Date().toISOString(), dataset: DEFAULT_HF_DATASET, samples }, null, 2));
  console.log(`Saved ${samples.length} evaluated MRI samples to ${outFile}`);
}

seed().catch(err => {
  console.error(err);
  process.exit(1);
});
