const $ = (sel) => document.querySelector(sel);

const state = {
  dataset: 'Hemg/Brain-Tumor-MRI-Dataset',
  choices: ['glioma', 'meningioma', 'notumor', 'pituitary'],
  samples: [],
  activeSampleIdx: 0,
  customImageDataUri: null,
  customTitle: null,
  threshold: 0.65,
  showBox: true,
  currentEval: null
};

function isNormalLabel(label) {
  if (!label) return false;
  const norm = String(label).toLowerCase().replace(/[\s_-]+/g, '');
  return norm === 'notumor' || norm === 'normal' || norm === 'healthy' || norm === 'negative';
}

function setStatus(text) {
  const el = $('#status-banner');
  if (el) el.textContent = text;
}

// Render HF Dataset ClassLabel Choice Chips
function renderChoicesBar() {
  const container = $('#choices-container');
  container.innerHTML = '';
  const winner = state.currentEval?.choice;

  for (const ch of state.choices) {
    const span = document.createElement('span');
    span.className = 'choice-chip' + (winner === ch ? ' active-winner' : '');
    span.textContent = ch;
    container.appendChild(span);
  }
}

// Render left-hand MRI sample cards
function renderSampleList() {
  const list = $('#sample-list');
  list.innerHTML = '';
  $('#sample-count-badge').textContent = `${state.samples.length} SLICES`;

  state.samples.forEach((sample, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'sample-card' + (idx === state.activeSampleIdx && !state.customImageDataUri ? ' active' : '');

    const img = document.createElement('img');
    img.className = 'sample-thumb';
    img.src = sample.imageDataUri || sample.imageUrl;
    img.alt = sample.title || `Row ${sample.rowIndex}`;

    const meta = document.createElement('div');
    meta.className = 'sample-meta';

    const title = document.createElement('div');
    title.className = 'sample-title';
    title.textContent = sample.title || `HF Row #${sample.rowIndex}`;

    const gt = document.createElement('div');
    gt.className = 'sample-gt';
    gt.textContent = `GT: ${sample.groundTruth || 'unknown'} • row ${sample.rowIndex}`;

    meta.appendChild(title);
    meta.appendChild(gt);
    btn.appendChild(img);
    btn.appendChild(meta);

    btn.addEventListener('click', () => {
      state.customImageDataUri = null;
      state.customTitle = null;
      state.activeSampleIdx = idx;
      const chosen = state.samples[idx];
      const selectedEngine = $('#engine-select')?.value || 'jev';
      if (chosen.evaluation && selectedEngine === 'jev') {
        state.currentEval = chosen.evaluation;
        setStatus(`Loaded golden evaluation for ${chosen.id} (${chosen.groundTruth})`);
        renderAll();
      } else {
        renderAll();
        runLiveEvaluation();
      }
    });

    list.appendChild(btn);
  });
}

// Render center MRI slice + bounding box + right-hand telemetry
function renderViewportAndTelemetry() {
  const activeSample = state.samples[state.activeSampleIdx];
  const imgSrc = state.customImageDataUri || activeSample?.imageDataUri || activeSample?.imageUrl || '';
  const titleText = state.customTitle || activeSample?.title || `HF Slice #${activeSample?.rowIndex ?? 0}`;

  $('#viewport-slice-title').textContent = titleText;
  const imgEl = $('#mri-image');
  if (imgSrc) imgEl.src = imgSrc;

  const ev = state.currentEval;
  if (!ev) return;

  const hasLesion = Boolean(ev.lesion ?? ev.cancer ?? ev.tumor);

  // Dimensions & Model footer
  const w = ev.image?.width || activeSample?.width || 512;
  const h = ev.image?.height || activeSample?.height || 512;
  $('#viewport-dims').textContent = `${w} × ${h} px`;
  $('#viewport-model').textContent = `model: ${ev.model || 'jev-1.14.0'}`;

  // Bounding Box overlay
  const boxEl = $('#lesion-box');
  if (state.showBox && ev.box && hasLesion) {
    boxEl.hidden = false;
    const topPct = (ev.box.top * 100).toFixed(1);
    const leftPct = (ev.box.left * 100).toFixed(1);
    const widthPct = ((ev.box.right - ev.box.left) * 100).toFixed(1);
    const heightPct = ((ev.box.bottom - ev.box.top) * 100).toFixed(1);
    boxEl.style.top = `${topPct}%`;
    boxEl.style.left = `${leftPct}%`;
    boxEl.style.width = `${widthPct}%`;
    boxEl.style.height = `${heightPct}%`;
    boxEl.setAttribute('data-box-label', `${ev.choice.toUpperCase()} (${(ev.probability * 100).toFixed(0)}%)`);
    $('#viewport-box-coords').textContent = `box: [top:${ev.box.top.toFixed(2)}, left:${ev.box.left.toFixed(2)}, bottom:${ev.box.bottom.toFixed(2)}, right:${ev.box.right.toFixed(2)}]`;
  } else {
    boxEl.hidden = true;
    $('#viewport-box-coords').textContent = hasLesion
      ? (ev.box ? 'box overlay hidden' : 'box: none (engine does not emit 2D coordinates)')
      : 'box: suppressed (no lesion detected)';
  }

  // 1. Lesion Detection Card
  $('#verdict-cancer').textContent = hasLesion ? 'LESION DETECTED' : 'NO LESION DETECTED';
  $('#verdict-prob').textContent = `P = ${(ev.probability * 100).toFixed(1)}%`;
  $('#noul-bar').style.width = `${Math.round(ev.probability * 100)}%`;
  $('#raw-noul-readout').textContent = `has_lesion.noul: ${Number(ev.noul).toFixed(2)}`;
  const gtLabel = ev.groundTruth || activeSample?.groundTruth;
  $('#gt-comparison-readout').textContent = gtLabel ? `Ground truth: ${gtLabel}` : 'Custom MRI scan';

  // 2. Choice Distribution Card
  $('#choice-conf-tag').textContent = `CONF: ${(ev.confidence * 100).toFixed(0)}%`;
  const probList = $('#probabilities-list');
  probList.innerHTML = '';

  const probs = ev.probabilities || {};
  const sortedEntries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  for (const [label, probVal] of sortedEntries) {
    const pct = Math.round((Number(probVal) || 0) * 100);
    const item = document.createElement('div');
    item.className = 'prob-item';

    const topRow = document.createElement('div');
    topRow.className = 'prob-item-top' + (label === ev.choice ? ' winner' : '');
    topRow.innerHTML = `<span>${label}</span><span>${(Number(probVal) * 100).toFixed(1)}%</span>`;

    const track = document.createElement('div');
    track.className = 'meter-track';
    const fill = document.createElement('div');
    fill.className = 'meter-fill';
    fill.style.width = `${pct}%`;
    fill.style.opacity = label === ev.choice ? '1' : '0.45';
    track.appendChild(fill);

    item.appendChild(topRow);
    item.appendChild(track);
    probList.appendChild(item);
  }

  // 3. Severity Score Card
  const sevScore = ev.severity?.score ?? 0;
  const sevMax = ev.severity?.max ?? 3;
  $('#score-conf-tag').textContent = `SCORE: ${sevScore.toFixed(2)} / ${sevMax.toFixed(2)}`;
  $('#score-bar').style.width = `${Math.min(100, Math.round((sevScore / sevMax) * 100))}%`;
  const nearestLevel = String(Math.round(sevScore));
  const legendDesc = ev.severity?.legend?.[nearestLevel] || 'Calibrated mass-effect & midline shift rating';
  $('#score-legend-text').textContent = `Level ${nearestLevel}: ${legendDesc}`;

  // 4. Confidence-Gated Triage
  $('#latency-tag').textContent = `${ev.latencyMs || 165} ms`;
  const needsReview = ev.confidence < state.threshold || (ev.probability >= 0.35 && ev.probability <= 0.65);
  const triageLabel = needsReview
    ? 'ESCALATE_RADIOLOGIST_REVIEW'
    : (hasLesion ? 'HIGH_CONFIDENCE_LESION_FINDING' : 'HIGH_CONFIDENCE_CLEAR_SCAN');
  $('#triage-status-text').textContent = triageLabel;

  // 5. Payload Inspector
  if (ev.usage) {
    $('#token-usage-readout').textContent = `input_tokens: ${ev.usage.input_tokens} | output_tokens: ${ev.usage.output_tokens}`;
  }

  const sampleRequestPayload = {
    model: 'jev-preview',
    state: { mri_scan: 'mri-scan-asset-uuid' },
    assets: {
      'mri-scan-asset-uuid': { type: 'image', format: ev.image?.format || 'jpg', data_base64: '<base64_mri_slice>' }
    },
    questions: {
      has_cancer: {
        type: 'noul',
        instructions: 'Does the MRI scan in `mri_scan` show a tumor or cancerous lesion?'
      },
      diagnosis: {
        type: 'choice',
        instructions: 'Which category from the Hugging Face MRI dataset best matches `mri_scan`?',
        criteria: Object.fromEntries((ev.choices || state.choices).map(c => [c, `<clinical_rubric + hf_exemplar_${c}>`]))
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

  const sampleResponsePayload = {
    model: ev.model || 'jev-1.14.0',
    answers: {
      has_cancer: { type: 'noul', noul: ev.noul },
      diagnosis: {
        type: 'choice',
        choice: ev.choice,
        confidence: ev.confidence,
        probabilities: ev.probabilities
      },
      severity: {
        type: 'score',
        score: ev.severity?.score,
        confidence: ev.severity?.confidence,
        probabilities: ev.severity?.probabilities
      },
      lesion_box: ev.box ? {
        type: 'box',
        box: {
          top: ev.box.top,
          bottom: ev.box.bottom,
          left: ev.box.left,
          right: ev.box.right
        },
        confidence: ev.box.confidence
      } : { type: 'box', note: 'Suppressed because has_lesion is false or engine is embedding-only' }
    },
    usage: ev.usage
  };

  function highlightJson(obj) {
    const raw = JSON.stringify(obj, null, 2)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    return raw.replace(
      /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
      (match) => {
        if (/^"/.test(match)) {
          if (/:$/.test(match)) {
            return `<span class="tok-prop">${match.slice(0, -1)}</span><span class="tok-punc">:</span>`;
          }
          return `<span class="tok-str">${match}</span>`;
        }
        if (/true|false|null/.test(match)) {
          return `<span class="tok-kw">${match}</span>`;
        }
        return `<span class="tok-num">${match}</span>`;
      }
    );
  }

  $('#request-json-box').innerHTML = highlightJson(sampleRequestPayload);
  $('#response-json-box').innerHTML = highlightJson(sampleResponsePayload);
}

function renderAll() {
  renderChoicesBar();
  renderSampleList();
  renderViewportAndTelemetry();
}

// Fetch choices & rows from Hugging Face Datasets Server (either via local server or direct CORS)
async function loadDatasetFromHF(datasetId) {
  setStatus(`Fetching ClassLabel choices for ${datasetId} from Hugging Face...`);
  try {
    const res = await fetch(`https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(datasetId)}&config=default&split=train&offset=0&length=4`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const labelFeat = (data.features || []).find(f => f.type?._type === 'ClassLabel' && Array.isArray(f.type?.names));
    const choices = labelFeat?.type?.names || ['tumor', 'notumor'];

    state.dataset = datasetId;
    state.choices = choices;

    const newSamples = (data.rows || []).map((r, i) => {
      const row = r.row || {};
      const img = row.image || {};
      const lblName = typeof row.label === 'number' ? choices[row.label] : String(row.label ?? '');
      return {
        id: `hf-row-${r.row_idx ?? i}`,
        title: `${datasetId.split('/')[1] || datasetId} — Slice #${r.row_idx ?? i}`,
        rowIndex: r.row_idx ?? i,
        dataset: datasetId,
        groundTruth: lblName,
        groundTruthCancer: !isNormalLabel(lblName),
        imageUrl: img.src,
        width: img.width || 512,
        height: img.height || 512,
        evaluation: null
      };
    });

    if (newSamples.length > 0) {
      state.samples = newSamples;
      state.activeSampleIdx = 0;
      state.customImageDataUri = null;
    }

    renderAll();
    setStatus(`Loaded ${choices.length} ClassLabel choices [${choices.join(', ')}] from ${datasetId}. Click "Run Evaluation" to evaluate.`);
  } catch (err) {
    setStatus(`Could not load dataset "${datasetId}": ${err.message}`);
  }
}

// Execute live evaluation across selected engine/model
async function runLiveEvaluation() {
  const btn = $('#evaluate-btn');
  const btnText = $('#evaluate-btn-text');
  const selectedEngine = $('#engine-select')?.value || 'jev';
  const visionModel = $('#vision-model-input')?.value?.trim() || 'qwen2.5vl:7b';
  btn.disabled = true;
  btnText.textContent = 'Evaluating...';

  const activeSample = state.samples[state.activeSampleIdx];
  const userKey = $('#api-key-input').value.trim();

  try {
    setStatus(`Running ${selectedEngine === 'ollama' ? `ollama (${visionModel})` : selectedEngine} evaluation on MRI slice...`);

    // 1. Try local server /api/evaluate first
    const localRes = await fetch('/api/evaluate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(userKey ? { 'X-TypeSafe-Key': userKey } : {})
      },
      body: JSON.stringify({
        engine: selectedEngine,
        visionModel,
        dataset: state.dataset,
        choices: state.choices,
        image: state.customImageDataUri || activeSample?.imageUrl || activeSample?.imageDataUri,
        groundTruth: state.customImageDataUri ? null : activeSample?.groundTruth,
        confidenceThreshold: state.threshold
      })
    }).catch(() => null);

    if (localRes && localRes.ok) {
      const result = await localRes.json();
      state.currentEval = result;
      renderAll();
      setStatus(`Evaluated with ${result.model} in ${result.latencyMs}ms — verdict: ${result.choice}`);
      return;
    }

    // 2. If running on static GitHub Pages without backend, use pre-computed golden evaluation
    if (!state.customImageDataUri && activeSample?.evaluation && selectedEngine === 'jev') {
      state.currentEval = activeSample.evaluation;
      renderAll();
      setStatus(`Evaluated via TypeSafe Jev (${state.currentEval.model}) — verdict: ${state.currentEval.choice}`);
      return;
    }

    if (localRes && !localRes.ok) {
      const errData = await localRes.json().catch(() => ({}));
      throw new Error(errData.error || `Evaluation failed (${localRes.status})`);
    }

    throw new Error('Provide a TYPESAFE_API_KEY or run `npm start` locally to evaluate.');
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Run Evaluation';
  }
}

// Initialize Golden Samples & Event Listeners
async function init() {
  try {
    const res = await fetch('./data/golden-samples.json');
    const data = await res.json();
    state.dataset = data.dataset || state.dataset;
    state.samples = data.samples || [];
    if (state.samples.length > 0) {
      state.currentEval = state.samples[0].evaluation;
      state.choices = state.currentEval?.choices || state.choices;
    }
    renderAll();
  } catch (err) {
    console.error('Failed to load golden samples:', err);
  }

  // Copy install command
  $('#copy-install-btn')?.addEventListener('click', () => {
    navigator.clipboard.writeText('npx is-lesion --hf Hemg/Brain-Tumor-MRI-Dataset --row 0');
    const span = $('#copy-install-btn span');
    span.textContent = 'Copied';
    setTimeout(() => { span.textContent = 'Copy'; }, 1500);
  });

  // Model / Engine dropdown change -> toggle vision-model input & evaluate current slice
  $('#engine-select')?.addEventListener('change', (e) => {
    const vlmInput = $('#vision-model-input');
    if (vlmInput) {
      vlmInput.style.display = e.target.value === 'ollama' ? 'inline-block' : 'none';
    }
    runLiveEvaluation();
  });

  // Dataset dropdown change
  $('#dataset-select')?.addEventListener('change', async (e) => {
    const ds = e.target.value;
    if (ds === 'Hemg/Brain-Tumor-MRI-Dataset') {
      const res = await fetch('./data/golden-samples.json');
      const data = await res.json();
      state.dataset = data.dataset;
      state.samples = data.samples;
      state.activeSampleIdx = 0;
      state.customImageDataUri = null;
      state.currentEval = state.samples[0].evaluation;
      state.choices = state.currentEval.choices;
      renderAll();
      setStatus(`Loaded golden samples for ${ds}`);
    } else {
      await loadDatasetFromHF(ds);
    }
  });

  // Custom HF repo load button
  $('#load-dataset-btn')?.addEventListener('click', async () => {
    const customDs = $('#custom-dataset-input').value.trim();
    const targetDs = customDs || $('#dataset-select').value;
    await loadDatasetFromHF(targetDs);
  });

  // Evaluate button
  $('#evaluate-btn')?.addEventListener('click', runLiveEvaluation);

  // Toggle bounding box overlay
  $('#toggle-box-checkbox')?.addEventListener('change', (e) => {
    state.showBox = e.target.checked;
    renderViewportAndTelemetry();
  });

  // Threshold slider
  $('#threshold-slider')?.addEventListener('input', (e) => {
    state.threshold = parseFloat(e.target.value);
    $('#threshold-val-label').textContent = state.threshold.toFixed(2);
    renderViewportAndTelemetry();
  });

  // Custom MRI image upload
  $('#file-upload-input')?.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      state.customImageDataUri = reader.result;
      state.customTitle = `Uploaded Scan — ${file.name}`;
      setStatus(`Loaded ${file.name}. Evaluating...`);
      renderAll();
      runLiveEvaluation();
    };
    reader.readAsDataURL(file);
  });
}

init();
