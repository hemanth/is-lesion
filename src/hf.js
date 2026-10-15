import { CLINICAL_DESCRIPTIONS, isNormalLabel, loadImageAsset } from './utils.js';

export const DEFAULT_HF_DATASET = 'Hemg/Brain-Tumor-MRI-Dataset';

export const CURATED_MRI_DATASETS = [
  {
    id: 'Hemg/Brain-Tumor-MRI-Dataset',
    name: 'Brain Tumor MRI (4-Class Standard)',
    choices: ['glioma', 'meningioma', 'notumor', 'pituitary'],
    sampleOffsets: { glioma: 0, meningioma: 3000, notumor: 5305, pituitary: 6500 },
    exemplarOffsets: { glioma: 10, meningioma: 3010, notumor: 5300, pituitary: 6510 }
  },
  {
    id: 'PranomVignesh/MRI-Images-of-Brain-Tumor',
    name: 'Clinical Brain MRI Benchmark',
    choices: ['glioma', 'meningioma', 'no-tumor', 'pituitary'],
    sampleOffsets: { glioma: 0, meningioma: 1500, 'no-tumor': 3000, pituitary: 4500 },
    exemplarOffsets: { glioma: 5, meningioma: 1505, 'no-tumor': 3005, pituitary: 4505 }
  },
  {
    id: 'RyanChen1209/Brain-Tumor-Classification-MRI',
    name: '7-Class Fine-Grained Oncology MRI',
    choices: ['Glioblastoma', 'Metastatic', 'Schwannoma', 'glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'],
    sampleOffsets: { Glioblastoma: 0, Meningioma: 200, Pituitary: 400 }
  },
  {
    id: 'BTX24/brain-cancer-mri-dataset',
    name: 'PMRAM Brain Cancer MRI',
    choices: ['brain_glioma', 'brain_menin', 'brain_tumor'],
    sampleOffsets: { brain_glioma: 0, brain_menin: 1 }
  }
];

const datasetMetaCache = new Map();
const exemplarAssetCache = new Map();

/**
 * Search Hugging Face Hub for MRI / tumor / oncology datasets.
 */
export async function searchHfDatasets(query = 'brain tumor mri', limit = 10) {
  const url = `https://huggingface.co/api/datasets?search=${encodeURIComponent(query)}&limit=${limit}&full=true`;
  const res = await fetch(url, { headers: { 'User-Agent': 'is-cancer/0.1.0' } });
  if (!res.ok) throw new Error(`Hugging Face search failed (${res.status})`);
  const items = await res.json();
  return (Array.isArray(items) ? items : []).map(item => ({
    id: item.id,
    downloads: item.downloads || 0,
    likes: item.likes || 0,
    description: (item.description || '').slice(0, 160)
  }));
}

/**
 * Resolve the `ClassLabel` choices and schema for any Hugging Face dataset.
 */
export async function resolveHfDatasetChoices(datasetId = DEFAULT_HF_DATASET, options = {}) {
  const cacheKey = `${datasetId}:${options.config || 'default'}:${options.split || 'train'}`;
  if (datasetMetaCache.has(cacheKey) && !options.force) {
    return datasetMetaCache.get(cacheKey);
  }

  const config = options.config || 'default';
  const split = options.split || 'train';
  const rowsUrl = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(datasetId)}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}&offset=0&length=1`;

  const res = await fetch(rowsUrl, { headers: { 'User-Agent': 'is-cancer/0.1.0' } });
  if (!res.ok) {
    // Fallback to curated choices if offline or rate-limited
    const curated = CURATED_MRI_DATASETS.find(d => d.id.toLowerCase() === datasetId.toLowerCase());
    if (curated) {
      return {
        dataset: datasetId,
        config,
        split,
        imageColumn: 'image',
        labelColumn: 'label',
        choices: curated.choices,
        criteria: buildChoiceCriteria(curated.choices)
      };
    }
    throw new Error(`Failed to inspect Hugging Face dataset "${datasetId}" (HTTP ${res.status})`);
  }

  const data = await res.json();
  const features = data.features || [];

  const imageFeature = features.find(f => f.type?._type === 'Image' || f.name === 'image') || { name: 'image' };
  const labelFeature = features.find(f => f.type?._type === 'ClassLabel' && Array.isArray(f.type?.names));

  const choices = labelFeature?.type?.names || ['tumor', 'notumor'];
  const meta = {
    dataset: datasetId,
    config,
    split,
    imageColumn: imageFeature.name,
    labelColumn: labelFeature ? labelFeature.name : 'label',
    choices,
    criteria: buildChoiceCriteria(choices)
  };

  datasetMetaCache.set(cacheKey, meta);
  return meta;
}

/**
 * Build TypeSafe `choice` criteria dictionary from dataset label names and optional image exemplars.
 */
export function buildChoiceCriteria(choices, exemplars = {}) {
  const criteria = {};
  for (const label of choices) {
    const key = String(label).toLowerCase();
    const desc = CLINICAL_DESCRIPTIONS[key] || (
      isNormalLabel(label)
        ? `Normal healthy MRI scan showing no tumor (${label}).`
        : `MRI scan showing ${String(label).replace(/[_-]+/g, ' ')} pathology.`
    );

    if (exemplars[label]) {
      criteria[label] = {
        description: desc,
        example: exemplars[label]
      };
    } else {
      criteria[label] = desc;
    }
  }
  return criteria;
}

const rowCache = new Map();

/**
 * Fetch specific rows from a Hugging Face dataset with resolved ground-truth label names.
 */
export async function fetchHfDatasetRows(datasetId = DEFAULT_HF_DATASET, { config = 'default', split = 'train', offset = 0, length = 4 } = {}) {
  const cacheKey = `${datasetId}:${config}:${split}:${offset}:${length}`;
  if (rowCache.has(cacheKey)) {
    return rowCache.get(cacheKey);
  }

  // Check if a larger pre-fetched batch in rowCache already covers [offset, offset + length)
  const prefix = `${datasetId}:${config}:${split}:`;
  for (const [k, cachedRows] of rowCache.entries()) {
    if (!k.startsWith(prefix)) continue;
    const parts = k.slice(prefix.length).split(':');
    const cOff = parseInt(parts[0], 10);
    const cLen = parseInt(parts[1], 10);
    if (offset >= cOff && (offset + length) <= (cOff + cLen) && cachedRows.length >= (offset - cOff + length)) {
      return cachedRows.slice(offset - cOff, offset - cOff + length);
    }
  }

  const meta = await resolveHfDatasetChoices(datasetId, { config, split });
  const rowsUrl = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(datasetId)}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}&offset=${offset}&length=${length}`;

  let lastStatus = 0;
  let data = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    const res = await fetch(rowsUrl, { headers: { 'User-Agent': 'is-cancer/0.1.0' } });
    lastStatus = res.status;
    if (res.ok) {
      data = await res.json();
      break;
    }
    await new Promise(r => setTimeout(r, 400 * (attempt + 1)));
  }

  if (!data) throw new Error(`Failed to fetch rows from ${datasetId} (HTTP ${lastStatus})`);

  const parsedRows = (data.rows || []).map(item => {
    const row = item.row || {};
    const imgObj = row[meta.imageColumn] || row.image || {};
    const labelIdx = row[meta.labelColumn] ?? row.label;
    const labelName = typeof labelIdx === 'number' && meta.choices[labelIdx] !== undefined
      ? meta.choices[labelIdx]
      : (typeof labelIdx === 'string' ? labelIdx : null);

    return {
      rowIndex: item.row_idx,
      dataset: datasetId,
      imageUrl: imgObj.src || null,
      width: imgObj.width || null,
      height: imgObj.height || null,
      labelIndex: labelIdx,
      groundTruth: labelName,
      groundTruthCancer: labelName ? !isNormalLabel(labelName) : null
    };
  });

  rowCache.set(cacheKey, parsedRows);
  return parsedRows;
}

/**
 * Fetch a balanced set of sample MRI scans across all classes in the dataset.
 */
export async function fetchBalancedHfSamples(datasetId = DEFAULT_HF_DATASET) {
  const curated = CURATED_MRI_DATASETS.find(d => d.id.toLowerCase() === datasetId.toLowerCase());
  if (curated && curated.sampleOffsets) {
    const entries = Object.entries(curated.sampleOffsets);
    const results = await Promise.all(
      entries.map(async ([, offset]) => {
        const rows = await fetchHfDatasetRows(datasetId, { offset, length: 1 });
        return rows[0];
      })
    );
    return results.filter(Boolean);
  }
  return fetchHfDatasetRows(datasetId, { offset: 0, length: 6 });
}

/**
 * Load and cache visual exemplar assets from Hugging Face dataset for visual criteria grounding.
 */
export async function loadHfExemplars(datasetId = DEFAULT_HF_DATASET) {
  if (exemplarAssetCache.has(datasetId)) {
    return exemplarAssetCache.get(datasetId);
  }

  const curated = CURATED_MRI_DATASETS.find(d => d.id.toLowerCase() === datasetId.toLowerCase());
  if (!curated || !curated.exemplarOffsets) {
    return null;
  }

  const exemplarRefs = {};
  const exemplarAssets = {};

  await Promise.all(
    Object.entries(curated.exemplarOffsets).map(async ([label, offset]) => {
      try {
        const rows = await fetchHfDatasetRows(datasetId, { offset, length: 1 });
        if (rows[0]?.imageUrl) {
          const loaded = await loadImageAsset(rows[0].imageUrl);
          const refKey = `exemplar-${label}-${loaded.ref.slice(-8)}`;
          exemplarRefs[label] = refKey;
          exemplarAssets[refKey] = loaded.asset;
        }
      } catch {
        // Ignore individual exemplar fetch failures cleanly
      }
    })
  );

  const bundle = { exemplarRefs, exemplarAssets };
  exemplarAssetCache.set(datasetId, bundle);
  return bundle;
}
