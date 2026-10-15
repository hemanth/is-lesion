import { isNormalLabel, loadImageAsset } from './utils.js';
import {
  DEFAULT_HF_DATASET,
  CURATED_MRI_DATASETS,
  resolveHfDatasetChoices,
  buildChoiceCriteria,
  fetchHfDatasetRows,
  fetchBalancedHfSamples,
  loadHfExemplars,
  searchHfDatasets
} from './hf.js';
import {
  evaluateWithClipOnnx,
  evaluateWithOllamaVlm,
  evaluateWithLocalFingerprint
} from './openjev.js';

export const SUPPORTED_ENGINES = [
  {
    id: 'jev',
    label: 'TypeSafe Jev 1.14 (+ Visual Exemplars)',
    runtime: 'cloud',
    model: 'jev-preview (jev-1.14.0)',
    primitives: ['noul', 'choice', 'score', 'box']
  },
  {
    id: 'jev-text',
    label: 'TypeSafe Jev 1.14 (Text-Only Criteria)',
    runtime: 'cloud',
    model: 'jev-preview (jev-1.14.0)',
    primitives: ['noul', 'choice', 'score', 'box']
  },
  {
    id: 'clip-exemplar',
    label: 'OpenJev Local ONNX · CLIP ViT-B/32 (+ Exemplars)',
    runtime: 'local-onnx',
    model: 'Xenova/clip-vit-base-patch32',
    primitives: ['noul', 'choice']
  },
  {
    id: 'clip-zeroshot',
    label: 'OpenJev Local ONNX · CLIP ViT-B/32 (Zero-Shot Text)',
    runtime: 'local-onnx',
    model: 'Xenova/clip-vit-base-patch32',
    primitives: ['noul', 'choice']
  },
  {
    id: 'ollama',
    label: 'OpenJev Local VLM · Ollama / OpenAI Vision',
    runtime: 'local-vlm',
    model: 'qwen2.5vl:7b / llama3.2-vision / medgemma',
    primitives: ['noul', 'choice', 'score', 'box']
  },
  {
    id: 'openjev',
    label: 'OpenJev In-Tree Exemplar Fingerprint (0-dep)',
    runtime: 'local-zero-dep',
    model: 'openjev:exemplar-fingerprint',
    primitives: ['noul', 'choice']
  }
];

export {
  DEFAULT_HF_DATASET,
  CURATED_MRI_DATASETS,
  resolveHfDatasetChoices,
  buildChoiceCriteria,
  fetchHfDatasetRows,
  fetchBalancedHfSamples,
  loadHfExemplars,
  searchHfDatasets,
  isNormalLabel,
  loadImageAsset,
  evaluateWithClipOnnx,
  evaluateWithOllamaVlm,
  evaluateWithLocalFingerprint
};

const TYPESAFE_ENDPOINT = 'https://api.typesafe.ai/v1/systemone';

/**
 * Detect lesions in MRI and medical scans using TypeSafe System One (Jev) or local OpenJev vision models
 * with Hugging Face dataset choices.
 *
 * @param {string | Buffer | ArrayBuffer | { src: string }} input - Path, URL, data URI, or Buffer of the scan
 * @param {object} [options]
 * @param {string} [options.dataset='Hemg/Brain-Tumor-MRI-Dataset'] - Hugging Face dataset ID to pull `choice` labels from
 * @param {string[]} [options.choices] - Explicit choice labels (overrides HF dataset lookup if provided)
 * @param {'jev' | 'jev-text' | 'clip-exemplar' | 'clip-zeroshot' | 'openjev' | 'ollama'} [options.engine='jev'] - Inference engine
 * @param {string} [options.visionModel='qwen2.5vl:7b'] - Local VLM model name when `engine: 'ollama'`
 * @param {boolean} [options.useExemplars=true] - Attach HF dataset reference exemplars into `choice` and `noul` criteria
 * @param {number} [options.confidenceThreshold=0.65] - Confidence threshold for automated decision vs clinical review
 * @param {string} [options.apiKey] - TypeSafe API key (defaults to process.env.TYPESAFE_API_KEY)
 * @param {string} [options.model='jev-preview'] - TypeSafe model ('jev-preview' supports `box` + images)
 * @param {string} [options.groundTruth] - Optional ground-truth label when evaluating a dataset row
 */
export default async function isLesion(input, options = {}) {
  const datasetId = options.dataset || DEFAULT_HF_DATASET;
  const engine = options.engine || (options.apiKey || process.env.TYPESAFE_API_KEY ? 'jev' : 'openjev');
  const model = options.model || 'jev-preview';
  const confidenceThreshold = options.confidenceThreshold ?? 0.65;
  const useExemplars = engine === 'jev-text' ? false : (options.useExemplars !== false);

  // 1. Resolve Choice options from Hugging Face dataset (or explicit choices array)
  let choices = options.choices;
  if (!Array.isArray(choices) || choices.length === 0) {
    const meta = await resolveHfDatasetChoices(datasetId, options);
    choices = meta.choices;
  }

  // 2. Prepare target MRI image asset
  const targetImage = await loadImageAsset(input);
  const assets = {
    [targetImage.ref]: targetImage.asset
  };

  // 3. Optionally attach HF dataset visual exemplars into criteria
  let exemplarRefs = {};
  let exBundle = null;
  if (useExemplars || engine === 'openjev' || engine === 'clip-exemplar') {
    exBundle = await loadHfExemplars(datasetId);
    if (exBundle && Object.keys(exBundle.exemplarRefs).length > 0) {
      exemplarRefs = exBundle.exemplarRefs;
      Object.assign(assets, exBundle.exemplarAssets);
    }
  }

  // OpenJev Engine 1: Local ONNX CLIP (Exemplar Cosine or Zero-Shot Text)
  if (engine === 'clip-exemplar' || engine === 'clip-zeroshot' || engine === 'clip') {
    const clipMode = engine === 'clip-zeroshot' ? 'zeroshot' : 'exemplar';
    const res = await evaluateWithClipOnnx(targetImage, choices, exBundle, { ...options, clipMode, dataset: datasetId, confidenceThreshold });
    return {
      ...res,
      withDataset: (nextDatasetId, nextOpts = {}) => isLesion(input, { ...options, ...nextOpts, dataset: nextDatasetId }),
      withEngine: (nextEngine, nextOpts = {}) => isLesion(input, { ...options, ...nextOpts, engine: nextEngine })
    };
  }

  // OpenJev Engine 2: Local VLM via Ollama / OpenAI-compatible Vision API (Qwen2.5-VL, Llama-3.2-Vision, MedGemma)
  if (engine === 'ollama' || engine === 'vlm') {
    const res = await evaluateWithOllamaVlm(targetImage, choices, { ...options, dataset: datasetId, confidenceThreshold });
    return {
      ...res,
      withDataset: (nextDatasetId, nextOpts = {}) => isLesion(input, { ...options, ...nextOpts, dataset: nextDatasetId }),
      withEngine: (nextEngine, nextOpts = {}) => isLesion(input, { ...options, ...nextOpts, engine: nextEngine })
    };
  }

  // OpenJev Engine 3: Zero-Key Local Exemplar Fingerprint Matcher (or fallback if no TYPESAFE_API_KEY)
  const apiKey = options.apiKey || process.env.TYPESAFE_API_KEY;
  if (engine === 'openjev' || engine === 'local') {
    const res = evaluateWithLocalFingerprint(targetImage, choices, exBundle, { ...options, dataset: datasetId, confidenceThreshold });
    return {
      ...res,
      withDataset: (nextDatasetId, nextOpts = {}) => isLesion(input, { ...options, ...nextOpts, dataset: nextDatasetId }),
      withEngine: (nextEngine, nextOpts = {}) => isLesion(input, { ...options, ...nextOpts, engine: nextEngine })
    };
  }

  if (!apiKey) {
    throw new Error('Missing TYPESAFE_API_KEY. Set process.env.TYPESAFE_API_KEY, pass { apiKey }, or select an OpenJev local engine.');
  }

  const choiceCriteria = buildChoiceCriteria(choices, exemplarRefs);

  // Build Noul criteria (with positive/negative visual exemplars if available)
  const positiveExemplars = Object.entries(exemplarRefs)
    .filter(([label]) => !isNormalLabel(label))
    .map(([, ref]) => ref);
  const negativeExemplars = Object.entries(exemplarRefs)
    .filter(([label]) => isNormalLabel(label))
    .map(([, ref]) => ref);

  const noulCriteria = positiveExemplars.length > 0 && negativeExemplars.length > 0
    ? {
        true: {
          description: 'Contains a visible tumor, abnormal hyperintense/hypointense mass, or oncology lesion.',
          examples: positiveExemplars
        },
        false: {
          description: 'Normal, symmetrical healthy brain MRI scan without any tumor mass or lesion.',
          examples: negativeExemplars
        }
      }
    : {
        true: 'Contains a visible tumor, abnormal mass, or oncology lesion in the MRI scan.',
        false: 'Normal healthy MRI scan with no tumor or abnormal mass.'
      };

  // 4. Speculative Fan-Out: evaluate Noul + HF Choice + Score + Box in a single System One call
  const requestBody = {
    model,
    state: {
      mri_scan: targetImage.ref
    },
    assets,
    questions: {
      has_cancer: {
        type: 'noul',
        instructions: 'Does the MRI scan in `mri_scan` show a tumor or cancerous lesion?',
        criteria: noulCriteria
      },
      diagnosis: {
        type: 'choice',
        instructions: 'Which category from the Hugging Face MRI dataset best matches `mri_scan`?',
        criteria: choiceCriteria
      },
      severity: {
        type: 'score',
        instructions: 'Rate the visual prominence and mass effect of any abnormality in `mri_scan`.',
        criteria: [
          'Normal anatomy with no visible lesion or mass effect',
          'Small localized focal abnormality with minimal mass effect',
          'Moderate well-defined lesion or tumor mass',
          'Large prominent tumor with significant mass effect or midline shift'
        ]
      },
      lesion_box: {
        type: 'box',
        instructions: 'the tumor or abnormal mass in `mri_scan`'
      }
    }
  };

  const startedAt = Date.now();
  const response = await fetch(TYPESAFE_ENDPOINT, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'TypeSafe-Danger-Zone-Images': 'true'
    },
    body: JSON.stringify(requestBody)
  });

  const latencyMs = Date.now() - startedAt;
  const payload = await response.json();

  if (!response.ok) {
    const errMsg = payload?.error?.message || payload?.message || JSON.stringify(payload);
    throw new Error(`TypeSafe System One API error (${response.status}): ${errMsg}`);
  }

  const answers = payload.answers || {};
  const noulProb = answers.has_cancer?.noul ?? 0;
  const topChoice = answers.diagnosis?.choice ?? choices[0];
  const probabilities = answers.diagnosis?.probabilities ?? {};
  const choiceConfidence = answers.diagnosis?.confidence ?? 0;
  const severityScore = answers.severity?.score ?? 0;
  const severityLegend = answers.severity?.legend ?? {};
  const severityProbs = answers.severity?.probabilities ?? {};
  const rawBox = answers.lesion_box?.box ?? null;
  const boxConfidence = answers.lesion_box?.confidence ?? null;

  // Compute aggregate lesion probability from the HF Choice distribution
  const choiceLesionProb = Object.entries(probabilities).reduce((acc, [lbl, p]) => {
    return isNormalLabel(lbl) ? acc : acc + (Number(p) || 0);
  }, 0);

  // Calibrated composite lesion probability combining Noul + Choice distribution
  const calibratedProbability = Number(((noulProb * 0.5) + (choiceLesionProb * 0.5)).toFixed(4));
  const detected = !isNormalLabel(topChoice) && calibratedProbability >= 0.5;

  // Per TypeSafe Box docs: a Box question always returns 4 edges, so only surface the box when a lesion is present
  const box = (detected || noulProb >= 0.45) && rawBox ? {
    ...rawBox,
    confidence: boxConfidence,
    pixels: {
      left: Math.round(rawBox.left * targetImage.width),
      top: Math.round(rawBox.top * targetImage.height),
      right: Math.round(rawBox.right * targetImage.width),
      bottom: Math.round(rawBox.bottom * targetImage.height)
    }
  } : null;

  // Confidence-gated clinical triage routing
  const needsReview = choiceConfidence < confidenceThreshold || (calibratedProbability >= 0.35 && calibratedProbability <= 0.65);
  const triage = needsReview
    ? 'ESCALATE_RADIOLOGIST_REVIEW'
    : (detected ? 'HIGH_CONFIDENCE_LESION_FINDING' : 'HIGH_CONFIDENCE_CLEAR_SCAN');

  return {
    lesion: detected,
    tumor: detected,
    cancer: detected,
    probability: calibratedProbability,
    noul: noulProb,
    choice: topChoice,
    confidence: choiceConfidence,
    probabilities,
    severity: {
      score: severityScore,
      max: 3,
      confidence: answers.severity?.confidence ?? 0,
      legend: severityLegend,
      probabilities: severityProbs
    },
    box,
    triage,
    needsReview,
    dataset: datasetId,
    choices,
    groundTruth: options.groundTruth || null,
    image: {
      width: targetImage.width,
      height: targetImage.height,
      format: targetImage.format
    },
    model: payload.model || model,
    usage: payload.usage || null,
    latencyMs,
    withDataset: (nextDatasetId, nextOpts = {}) => isLesion(input, { ...options, ...nextOpts, dataset: nextDatasetId }),
    withEngine: (nextEngine, nextOpts = {}) => isLesion(input, { ...options, ...nextOpts, engine: nextEngine })
  };
}

/**
 * Evaluate a specific row index directly from a Hugging Face MRI dataset.
 */
isLesion.fromHF = async function fromHF(datasetId = DEFAULT_HF_DATASET, options = {}) {
  const offset = options.offset ?? options.row ?? 0;
  const rows = await fetchHfDatasetRows(datasetId, {
    config: options.config || 'default',
    split: options.split || 'train',
    offset,
    length: 1
  });
  if (!rows.length || !rows[0].imageUrl) {
    throw new Error(`No MRI image found at row ${offset} in dataset "${datasetId}"`);
  }
  const sample = rows[0];
  const result = await isLesion(sample.imageUrl, {
    ...options,
    dataset: datasetId,
    groundTruth: sample.groundTruth
  });
  return {
    ...result,
    rowIndex: sample.rowIndex,
    imageUrl: sample.imageUrl,
    groundTruth: sample.groundTruth,
    groundTruthLesion: sample.groundTruthCancer,
    groundTruthCancer: sample.groundTruthCancer,
    correctChoice: sample.groundTruth ? result.choice === sample.groundTruth : null,
    correctBinary: sample.groundTruthCancer !== null ? result.lesion === sample.groundTruthCancer : null
  };
};

/**
 * Resolve choices from a Hugging Face dataset.
 */
isLesion.choices = async function getChoices(datasetId = DEFAULT_HF_DATASET, options = {}) {
  const meta = await resolveHfDatasetChoices(datasetId, options);
  return meta.choices;
};

/**
 * Search Hugging Face Hub for MRI datasets.
 */
isLesion.datasets = searchHfDatasets;

/**
 * List supported vision engines (TypeSafe Jev + local OpenJev engines).
 */
isLesion.engines = () => SUPPORTED_ENGINES;

export { isLesion, isLesion as isCancer };
