import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import isLesion, {
  SUPPORTED_ENGINES,
  CURATED_MRI_DATASETS,
  DEFAULT_HF_DATASET,
  resolveHfDatasetChoices,
  fetchHfDatasetRows,
  searchHfDatasets
} from './src/index.js';

// Load .env if present
try {
  const envPath = path.resolve('.env');
  if (fs.existsSync(envPath)) {
    const lines = fs.readFileSync(envPath, 'utf-8').split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      const eq = trimmed.indexOf('=');
      if (eq !== -1) {
        const k = trimmed.slice(0, eq).trim();
        const v = trimmed.slice(eq + 1).trim();
        if (!process.env[k]) process.env[k] = v;
      }
    }
  }
} catch {
  // Ignore
}

const PORT = parseInt(process.env.PORT || '3480', 10);
const DOCS_DIR = path.resolve('docs');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.ico': 'image/x-icon'
};

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', chunk => {
      data += chunk;
      if (data.length > 20 * 1024 * 1024) {
        reject(new Error('Payload too large'));
      }
    });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

function sendJson(res, status, obj) {
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': '*'
  });
  res.end(JSON.stringify(obj));
}

const server = http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-TypeSafe-Key');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);

  // 1. Health & capabilities
  if (url.pathname === '/api/health' && req.method === 'GET') {
    return sendJson(res, 200, {
      status: 'ok',
      hasApiKey: Boolean(process.env.TYPESAFE_API_KEY),
      defaultDataset: DEFAULT_HF_DATASET,
      curatedDatasets: CURATED_MRI_DATASETS,
      supportedEngines: SUPPORTED_ENGINES
    });
  }

  // 2. Search Hugging Face MRI datasets
  if (url.pathname === '/api/hf/search' && req.method === 'GET') {
    try {
      const q = url.searchParams.get('q') || 'brain tumor mri';
      const datasets = await searchHfDatasets(q, 12);
      return sendJson(res, 200, { datasets });
    } catch (err) {
      return sendJson(res, 500, { error: err.message });
    }
  }

  // 3. Resolve dataset ClassLabel choices + sample rows
  if (url.pathname === '/api/hf/choices' && req.method === 'GET') {
    try {
      const dataset = url.searchParams.get('dataset') || DEFAULT_HF_DATASET;
      const offset = parseInt(url.searchParams.get('offset') || '0', 10);
      const length = parseInt(url.searchParams.get('length') || '4', 10);
      const meta = await resolveHfDatasetChoices(dataset);
      const rows = await fetchHfDatasetRows(dataset, { offset, length });
      return sendJson(res, 200, { ...meta, rows });
    } catch (err) {
      return sendJson(res, 500, { error: err.message });
    }
  }

  // 4. Evaluate an MRI scan via TypeSafe System One (Jev) or OpenJev Local Engines
  if (url.pathname === '/api/evaluate' && req.method === 'POST') {
    try {
      const raw = await readBody(req);
      const body = JSON.parse(raw || '{}');
      const apiKey = req.headers['x-typesafe-key'] || body.apiKey || process.env.TYPESAFE_API_KEY;
      const dataset = body.dataset || DEFAULT_HF_DATASET;
      const choices = body.choices;
      const engine = body.engine || 'jev';
      const visionModel = body.visionModel;
      const useExemplars = engine === 'jev-text' ? false : (body.useExemplars !== false);
      const confidenceThreshold = body.confidenceThreshold ?? 0.65;

      let result;
      if (body.row !== undefined && !body.image) {
        result = await isLesion.fromHF(dataset, {
          apiKey,
          engine,
          visionModel,
          offset: Number(body.row),
          choices,
          useExemplars,
          confidenceThreshold
        });
      } else if (body.image) {
        result = await isLesion(body.image, {
          apiKey,
          engine,
          visionModel,
          dataset,
          choices,
          useExemplars,
          confidenceThreshold,
          groundTruth: body.groundTruth || null
        });
      } else {
        return sendJson(res, 400, { error: 'Provide either "image" (URL/dataURI) or "row" index.' });
      }

      const { withDataset, withEngine, ...cleanResult } = result;
      return sendJson(res, 200, cleanResult);
    } catch (err) {
      return sendJson(res, 500, { error: err.message });
    }
  }

  // Serve static files from docs/
  let filePath = path.join(DOCS_DIR, url.pathname === '/' ? 'index.html' : url.pathname);
  if (!filePath.startsWith(DOCS_DIR)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    filePath = path.join(DOCS_DIR, 'index.html');
  }

  try {
    const ext = path.extname(filePath);
    const content = fs.readFileSync(filePath);
    res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream' });
    res.end(content);
  } catch {
    res.writeHead(404);
    res.end('Not Found');
  }
});

server.listen(PORT, () => {
  console.log(`\n  is-lesion studio running at http://localhost:${PORT}\n`);
});
