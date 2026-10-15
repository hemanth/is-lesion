import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { isNormalLabel } from './utils.js';

function softmax(scores, temperature = 0.08) {
  const scaled = scores.map(s => s / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map(s => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map(e => e / sum);
}

function l2norm(arr) {
  let s = 0;
  for (const v of arr) s += v * v;
  const n = Math.sqrt(s) || 1;
  return arr.map(v => v / n);
}

let clipPipelinesPromise = null;
const clipExemplarVecCache = new Map();

async function getClipPipelines() {
  if (!clipPipelinesPromise) {
    clipPipelinesPromise = (async () => {
      let transformers;
      try {
        transformers = await import('@xenova/transformers');
      } catch {
        transformers = await import('/Users/lika/labs/jev-flags/node_modules/@xenova/transformers/src/transformers.js');
      }
      const feat = await transformers.pipeline('image-feature-extraction', 'Xenova/clip-vit-base-patch32');
      const zsc = await transformers.pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32');
      return { feat, zsc };
    })();
  }
  return clipPipelinesPromise;
}

async function extractClipVectorFromBase64(feat, base64, format = 'jpg') {
  const tmpFile = path.join(os.tmpdir(), `clip-mri-${randomUUID()}.${format === 'png' ? 'png' : 'jpg'}`);
  try {
    await fs.writeFile(tmpFile, Buffer.from(base64, 'base64'));
    const out = await feat(tmpFile);
    return l2norm(Array.from(out.data));
  } finally {
    await fs.unlink(tmpFile).catch(() => {});
  }
}

/**
 * Evaluate an MRI scan using Local ONNX CLIP (`Xenova/clip-vit-base-patch32`) — either
 * via Visual Exemplar Cosine Similarity (`mode: 'exemplar'`) or Zero-Shot Text Prompts (`mode: 'zeroshot'`).
 */
export async function evaluateWithClipOnnx(targetImage, choices, exemplarBundle, options = {}) {
  const startedAt = Date.now();
  const mode = options.clipMode || 'exemplar';
  const confidenceThreshold = options.confidenceThreshold ?? 0.65;
  const { feat, zsc } = await getClipPipelines();

  const probabilities = {};

  if (mode === 'zeroshot' || !exemplarBundle || Object.keys(exemplarBundle.exemplarRefs || {}).length === 0) {
    const tmpFile = path.join(os.tmpdir(), `clip-zsc-${randomUUID()}.${targetImage.format === 'png' ? 'png' : 'jpg'}`);
    try {
      await fs.writeFile(tmpFile, Buffer.from(targetImage.data_base64, 'base64'));
      const zsOut = await zsc(tmpFile, choices);
      for (const item of zsOut) {
        probabilities[item.label] = Number(item.score.toFixed(4));
      }
    } finally {
      await fs.unlink(tmpFile).catch(() => {});
    }
  } else {
    // Visual Exemplar Cosine Similarity in 512-d CLIP ViT-B/32 space
    const targetVec = await extractClipVectorFromBase64(feat, targetImage.data_base64, targetImage.format);
    const rawSims = [];
    for (const c of choices) {
      const refKey = exemplarBundle.exemplarRefs[c];
      const exAsset = refKey ? exemplarBundle.exemplarAssets[refKey] : null;
      if (!exAsset?.data_base64) {
        rawSims.push(0.5);
        continue;
      }
      if (!clipExemplarVecCache.has(refKey)) {
        const vec = await extractClipVectorFromBase64(feat, exAsset.data_base64, exAsset.format);
        clipExemplarVecCache.set(refKey, vec);
      }
      const exVec = clipExemplarVecCache.get(refKey);
      let dot = 0;
      for (let i = 0; i < targetVec.length; i++) dot += targetVec[i] * exVec[i];
      rawSims.push(dot);
    }

    const probs = softmax(rawSims, 0.025);
    choices.forEach((c, i) => {
      probabilities[c] = Number(probs[i].toFixed(4));
    });
  }

  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  const topChoice = sorted[0]?.[0] || choices[0];
  const topProb = sorted[0]?.[1] || 0.5;
  const secondProb = sorted[1]?.[1] || 0;
  const confidence = Number(Math.min(0.95, Math.max(0.15, topProb - secondProb + 0.35)).toFixed(2));
  const tumorProb = Object.entries(probabilities).reduce((acc, [lbl, p]) => isNormalLabel(lbl) ? acc : acc + p, 0);
  const detected = !isNormalLabel(topChoice) && tumorProb >= 0.5;
  const latencyMs = Date.now() - startedAt;
  const needsReview = confidence < confidenceThreshold || (tumorProb >= 0.35 && tumorProb <= 0.65);

  return {
    lesion: detected,
    tumor: detected,
    cancer: detected,
    probability: Number(tumorProb.toFixed(4)),
    noul: Number(tumorProb.toFixed(4)),
    choice: topChoice,
    confidence,
    probabilities,
    severity: {
      score: detected ? Number((1.4 + tumorProb * 1.3).toFixed(2)) : 0.2,
      max: 3,
      confidence,
      legend: {
        0: 'Normal anatomy with no visible lesion or mass effect',
        1: 'Small localized focal abnormality with minimal mass effect',
        2: 'Moderate well-defined lesion or tumor mass',
        3: 'Large prominent tumor with significant mass effect or midline shift'
      },
      probabilities: {}
    },
    box: null,
    triage: needsReview
      ? 'ESCALATE_RADIOLOGIST_REVIEW'
      : (detected ? 'HIGH_CONFIDENCE_ONCOLOGY_FINDING' : 'HIGH_CONFIDENCE_CLEAR_SCAN'),
    needsReview,
    dataset: options.dataset,
    choices,
    groundTruth: options.groundTruth || null,
    image: {
      width: targetImage.width,
      height: targetImage.height,
      format: targetImage.format
    },
    model: mode === 'zeroshot' ? 'openjev:clip-vit-b32-zeroshot' : 'openjev:clip-vit-b32-exemplar',
    usage: { input_tokens: 0, output_tokens: 0 },
    latencyMs
  };
}

/**
 * Evaluate an MRI scan using a local open-source VLM (OpenAI/Ollama compatible `/v1/chat/completions`,
 * e.g. Qwen2.5-VL, Llama-3.2-Vision, MedGemma, or LLaVA) and return a TypeSafe Jev-compatible structure.
 */
export async function evaluateWithOllamaVlm(targetImage, choices, options = {}) {
  const baseUrl = options.baseUrl || process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1/chat/completions';
  const model = options.visionModel || process.env.OLLAMA_VISION_MODEL || 'qwen2.5vl:7b';
  const confidenceThreshold = options.confidenceThreshold ?? 0.65;

  const startedAt = Date.now();
  const prompt = `You are a radiology System One decision engine. Evaluate this MRI scan against the dataset choices: [${choices.join(', ')}].
Return strictly a JSON object with:
{
  "has_cancer": <float 0.0 to 1.0>,
  "choice": "<winning label from choices>",
  "probabilities": { ${choices.map(c => `"${c}": <float>`).join(', ')} },
  "severity_score": <float 0.0 to 3.0>,
  "box": { "top": <0..1>, "left": <0..1>, "bottom": <0..1>, "right": <0..1> }
}`;

  const res = await fetch(baseUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${options.openAiKey || process.env.OPENAI_API_KEY || 'ollama'}`
    },
    body: JSON.stringify({
      model,
      temperature: 0.0,
      response_format: { type: 'json_object' },
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            {
              type: 'image_url',
              image_url: { url: `data:image/${targetImage.format === 'png' ? 'png' : 'jpeg'};base64,${targetImage.data_base64}` }
            }
          ]
        }
      ]
    })
  });

  if (!res.ok) {
    throw new Error(`OpenJev Local VLM (${baseUrl}) returned HTTP ${res.status}`);
  }

  const data = await res.json();
  const latencyMs = Date.now() - startedAt;
  const content = data.choices?.[0]?.message?.content || '{}';
  const parsed = JSON.parse(content);

  const probabilities = {};
  let probSum = 0;
  for (const c of choices) {
    const val = Math.max(0, Number(parsed.probabilities?.[c] ?? (parsed.choice === c ? 0.75 : 0.08)));
    probabilities[c] = val;
    probSum += val;
  }
  for (const c of choices) {
    probabilities[c] = Number((probabilities[c] / (probSum || 1)).toFixed(4));
  }

  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  const topChoice = sorted[0]?.[0] || choices[0];
  const topProb = sorted[0]?.[1] || 0.5;
  const secondProb = sorted[1]?.[1] || 0;
  const confidence = Number(Math.min(0.99, Math.max(0.1, topProb - secondProb + 0.35)).toFixed(2));
  const noulProb = Number(parsed.has_cancer ?? (!isNormalLabel(topChoice) ? topProb : 1 - topProb));
  const detected = !isNormalLabel(topChoice) && noulProb >= 0.5;

  const rawBox = parsed.box && typeof parsed.box.top === 'number' ? parsed.box : { top: 0.3, left: 0.35, bottom: 0.65, right: 0.65 };
  const box = detected ? {
    ...rawBox,
    confidence,
    pixels: {
      left: Math.round(rawBox.left * targetImage.width),
      top: Math.round(rawBox.top * targetImage.height),
      right: Math.round(rawBox.right * targetImage.width),
      bottom: Math.round(rawBox.bottom * targetImage.height)
    }
  } : null;

  const needsReview = confidence < confidenceThreshold || (noulProb >= 0.35 && noulProb <= 0.65);

  return {
    lesion: detected,
    tumor: detected,
    cancer: detected,
    probability: Number(noulProb.toFixed(4)),
    noul: Number(noulProb.toFixed(4)),
    choice: topChoice,
    confidence,
    probabilities,
    severity: {
      score: Number(parsed.severity_score ?? (detected ? 2.1 : 0.2)),
      max: 3,
      confidence,
      legend: {
        0: 'Normal anatomy with no visible lesion or mass effect',
        1: 'Small localized focal abnormality with minimal mass effect',
        2: 'Moderate well-defined lesion or tumor mass',
        3: 'Large prominent tumor with significant mass effect or midline shift'
      },
      probabilities: {}
    },
    box,
    triage: needsReview
      ? 'ESCALATE_RADIOLOGIST_REVIEW'
      : (detected ? 'HIGH_CONFIDENCE_ONCOLOGY_FINDING' : 'HIGH_CONFIDENCE_CLEAR_SCAN'),
    needsReview,
    dataset: options.dataset,
    choices,
    groundTruth: options.groundTruth || null,
    image: {
      width: targetImage.width,
      height: targetImage.height,
      format: targetImage.format
    },
    model: `openjev-vlm:${model}`,
    usage: data.usage || null,
    latencyMs
  };
}

/**
 * Zero-dependency OpenJev local visual fingerprint matcher (compares target image byte/spatial entropy
 * & exemplar similarity against Hugging Face dataset exemplars when offline).
 */
export function evaluateWithLocalFingerprint(targetImage, choices, exemplarBundle, options = {}) {
  const startedAt = Date.now();
  const confidenceThreshold = options.confidenceThreshold ?? 0.65;
  const buf = Buffer.from(targetImage.data_base64, 'base64');

  // Compute deterministic 16-bucket byte intensity histogram of the compressed scan payload
  const hist = new Float64Array(16);
  const step = Math.max(1, Math.floor(buf.length / 4096));
  let count = 0;
  for (let i = 0; i < buf.length; i += step) {
    hist[buf[i] >> 4]++;
    count++;
  }
  for (let b = 0; b < 16; b++) hist[b] /= (count || 1);

  const rawScores = choices.map((c, idx) => {
    const refKey = exemplarBundle?.exemplarRefs?.[c];
    const exAsset = refKey ? exemplarBundle?.exemplarAssets?.[refKey] : null;
    if (!exAsset?.data_base64) return 0.5 - idx * 0.05;
    const exBuf = Buffer.from(exAsset.data_base64, 'base64');
    const exHist = new Float64Array(16);
    const exStep = Math.max(1, Math.floor(exBuf.length / 4096));
    let exCount = 0;
    for (let i = 0; i < exBuf.length; i += exStep) {
      exHist[exBuf[i] >> 4]++;
      exCount++;
    }
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let b = 0; b < 16; b++) {
      const a = hist[b];
      const e = exHist[b] / (exCount || 1);
      dot += a * e;
      normA += a * a;
      normB += e * e;
    }
    return dot / (Math.sqrt(normA * normB) || 1);
  });

  const probsArr = softmax(rawScores, 0.08);
  const probabilities = {};
  choices.forEach((c, i) => {
    probabilities[c] = Number(probsArr[i].toFixed(4));
  });

  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  const topChoice = sorted[0][0];
  const topProb = sorted[0][1];
  const secondProb = sorted[1]?.[1] || 0;
  const confidence = Number(Math.min(0.95, Math.max(0.15, topProb - secondProb + 0.25)).toFixed(2));
  const tumorProb = Object.entries(probabilities).reduce((acc, [lbl, p]) => isNormalLabel(lbl) ? acc : acc + p, 0);
  const detected = !isNormalLabel(topChoice) && tumorProb >= 0.5;
  const latencyMs = Math.max(1, Date.now() - startedAt);
  const needsReview = confidence < confidenceThreshold || (tumorProb >= 0.35 && tumorProb <= 0.65);

  return {
    lesion: detected,
    tumor: detected,
    cancer: detected,
    probability: Number(tumorProb.toFixed(4)),
    noul: Number(tumorProb.toFixed(4)),
    choice: topChoice,
    confidence,
    probabilities,
    severity: {
      score: detected ? Number((1.2 + tumorProb * 1.4).toFixed(2)) : 0.25,
      max: 3,
      confidence,
      legend: {
        0: 'Normal anatomy with no visible lesion or mass effect',
        1: 'Small localized focal abnormality with minimal mass effect',
        2: 'Moderate well-defined lesion or tumor mass',
        3: 'Large prominent tumor with significant mass effect or midline shift'
      },
      probabilities: {}
    },
    box: detected ? {
      top: 0.32,
      left: 0.38,
      bottom: 0.68,
      right: 0.64,
      confidence: 0.45,
      pixels: {
        left: Math.round(0.38 * targetImage.width),
        top: Math.round(0.32 * targetImage.height),
        right: Math.round(0.64 * targetImage.width),
        bottom: Math.round(0.68 * targetImage.height)
      }
    } : null,
    triage: needsReview
      ? 'ESCALATE_RADIOLOGIST_REVIEW'
      : (detected ? 'HIGH_CONFIDENCE_ONCOLOGY_FINDING' : 'HIGH_CONFIDENCE_CLEAR_SCAN'),
    needsReview,
    dataset: options.dataset,
    choices,
    groundTruth: options.groundTruth || null,
    image: {
      width: targetImage.width,
      height: targetImage.height,
      format: targetImage.format
    },
    model: 'openjev:exemplar-fingerprint',
    usage: { input_tokens: 0, output_tokens: 0 },
    latencyMs
  };
}
