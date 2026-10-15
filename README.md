# is-lesion

Detect lesions in MRI and medical scans using TypeSafe Jev System One, local OpenJev vision models, and Hugging Face dataset choices.

```bash
npm install is-lesion
```

## Quick start

```js
import isLesion from 'is-lesion';

const scan = await isLesion('./mri-slice.jpg');
console.log(scan.lesion, scan.choice, scan.probability, scan.box);
// => true 'glioma' 0.97 { top: 0.52, bottom: 0.8, left: 0.43, right: 0.64, pixels: {...} }

const row = await isLesion.fromHF('Hemg/Brain-Tumor-MRI-Dataset', { row: 3000 });
console.log(row.choice, row.groundTruth, row.triage);
// => 'meningioma' 'meningioma' 'HIGH_CONFIDENCE_LESION_FINDING'
```

`isLesion()` evaluates an image against Hugging Face `ClassLabel` choices (`noul`, `choice`, `score`, `box`). `fromHF()` evaluates a Hugging Face dataset row directly. `choices()` returns a dataset's labels. `engines()` lists supported cloud and local vision engines. That's the whole API.

## Select engine (TypeSafe Jev or local models)

```js
// 1. TypeSafe Jev 1.14 + HF visual exemplars (default when TYPESAFE_API_KEY is set)
const jevScan = await isLesion('./mri-slice.jpg', {
  engine: 'jev'
});

// 2. Local ONNX CLIP ViT-B/32 visual exemplar cosine similarity (offline CPU)
const clipScan = await isLesion('./mri-slice.jpg', {
  engine: 'clip-exemplar'
});

// 3. Local VLM via Ollama / OpenAI-compatible Vision API (Qwen2.5-VL, Llama-3.2-Vision, MedGemma)
const vlmScan = await isLesion('./mri-slice.jpg', {
  engine: 'ollama',
  visionModel: 'qwen2.5vl:7b'
});

// 4. Zero-dependency in-tree exemplar fingerprint matcher (1ms offline fallback)
const fastScan = await jevScan.withEngine('openjev');
```

- `jev` — TypeSafe System One (`jev-1.14.0`) with 1 Hugging Face visual reference exemplar per class (`noul` + `choice` + `score` + `box`)
- `jev-text` — TypeSafe System One (`jev-1.14.0`) with text-only clinical criteria
- `clip-exemplar` — Local ONNX `Xenova/clip-vit-base-patch32` 512-d visual exemplar cosine similarity
- `clip-zeroshot` — Local ONNX `Xenova/clip-vit-base-patch32` zero-shot text prompts
- `ollama` — Local multimodal VLM via `http://localhost:11434/v1/chat/completions` (`visionModel: 'qwen2.5vl:7b'`)
- `openjev` — Zero-dependency in-tree 16-bucket byte/spatial exemplar histogram matcher

## Use any Hugging Face dataset for choices

```js
const choices = await isLesion.choices('RyanChen1209/Brain-Tumor-Classification-MRI');
// => ['Glioblastoma', 'Metastatic', 'Schwannoma', 'glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

const scan = await isLesion('./mri-slice.jpg', {
  dataset: 'RyanChen1209/Brain-Tumor-Classification-MRI'
});
console.log(scan.probabilities);
// => { Glioblastoma: 0.72, glioma_tumor: 0.19, ... }
```

Pass any Hugging Face dataset repo ID in `{ dataset }` or switch datasets on an existing scan with `await scan.withDataset(datasetId)`.

## CLI

```bash
npx is-lesion ./scan.jpg
npx is-lesion ./scan.jpg --engine clip-exemplar
npx is-lesion ./scan.jpg --engine ollama --vision-model qwen2.5vl:7b
npx is-lesion --hf Hemg/Brain-Tumor-MRI-Dataset --row 0 --engine jev
npx is-lesion --choices RyanChen1209/Brain-Tumor-Classification-MRI
npx is-lesion --engines
```

Evaluate local files, URLs, or Hugging Face dataset rows from the terminal with `--engine`, `--dataset`, `--row`, and `--json`.

## Benchmarks

```bash
npm run bench
```

Evaluates `N = 100` stratified MRI slices (`25 glioma`, `25 meningioma`, `25 notumor`, `25 pituitary` — `500` total evaluations) on `Hemg/Brain-Tumor-MRI-Dataset`:

- `jev` (`jev-1.14.0` + HF exemplars): **84.0%** binary (`85.3%` sens, `80.0%` spec, `88.9%` F1), **69.0%** 4-class (`69.6%` Macro-F1), **0.0955 Brier** & **4.9% ECE** (best calibration), **93.0% binary / 86.0% 4-class** confidence-gated accuracy (`>= 0.65`, `57%` coverage, `74.2%` error catch), **165 ms P50**, native `score` + `box`
- `jev-text` (`jev-1.14.0` text-only): **81.0%** binary (`81.3%` sens, `80.0%` spec), **62.0%** 4-class (`62.2%` Macro-F1), **0.1113 Brier** & **7.1% ECE**, **89.3% binary / 80.4% 4-class** gated accuracy (`56%` coverage), **108 ms P50**
- `clip-exemplar` (ONNX `clip-vit-base-patch32` + exemplars): **89.0%** binary (`85.3%` sens, `100.0%` spec, `92.1%` F1), **69.0%** 4-class (`69.3%` Macro-F1), **0.0966 Brier** & **12.5% ECE** (`17%` gated coverage), **95 ms P50** offline CPU
- `clip-zeroshot` (ONNX `clip-vit-base-patch32` text-only): **75.0%** binary (`100%` sens, `0.0%` spec), **25.0%** 4-class (`10.0%` Macro-F1), **0.2125 Brier** & **69.0% ECE**, **106 ms P50**
- `openjev` (0-dep exemplar fingerprint): **88.0%** binary (`100%` sens, `52.0%` spec), **73.0%** 4-class (`73.3%` Macro-F1), **0.1655 Brier** & **46.4% ECE**, **1 ms P50**

## Playground

```bash
npm start
```

Opens the interactive MRI Radiology System One Workbench at `http://localhost:3480` with live model switching, Hugging Face dataset picker, lesion `box` overlay, and confidence triage gate.

## License

MIT © [Hemanth.HM](https://h3manth.com)
