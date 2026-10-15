import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import isLesion, {
  isNormalLabel,
  buildChoiceCriteria,
  resolveHfDatasetChoices
} from '../src/index.js';

if (!process.env.TYPESAFE_API_KEY && fs.existsSync('.env')) {
  const m = fs.readFileSync('.env', 'utf8').match(/TYPESAFE_API_KEY=([^\s]+)/);
  if (m) process.env.TYPESAFE_API_KEY = m[1];
}
const TEST_API_KEY = process.env.TYPESAFE_API_KEY;

test('isNormalLabel detects non-lesion labels across HF dataset naming conventions', () => {
  assert.equal(isNormalLabel('notumor'), true);
  assert.equal(isNormalLabel('no-tumor'), true);
  assert.equal(isNormalLabel('no_tumor'), true);
  assert.equal(isNormalLabel('normal'), true);
  assert.equal(isNormalLabel('glioma'), false);
  assert.equal(isNormalLabel('meningioma'), false);
  assert.equal(isNormalLabel('pituitary'), false);
});

test('isLesion.engines() lists all supported cloud and local vision engines', () => {
  const engines = isLesion.engines();
  const ids = engines.map(e => e.id);
  assert.deepEqual(ids, ['jev', 'jev-text', 'clip-exemplar', 'clip-zeroshot', 'ollama', 'openjev']);
});

test('buildChoiceCriteria maps HF dataset ClassLabel choices to clinical descriptions and visual exemplars', () => {
  const criteria = buildChoiceCriteria(['glioma', 'notumor'], { glioma: 'ref-123' });
  assert.equal(typeof criteria.glioma, 'object');
  assert.equal(criteria.glioma.example, 'ref-123');
  assert.equal(typeof criteria.notumor, 'string');
  assert.ok(criteria.notumor.toLowerCase().includes('normal'));
});

test('resolveHfDatasetChoices fetches dynamic ClassLabel choices from Hugging Face', async () => {
  const choices = await isLesion.choices('Hemg/Brain-Tumor-MRI-Dataset');
  assert.deepEqual(choices, ['glioma', 'meningioma', 'notumor', 'pituitary']);

  const meta7 = await resolveHfDatasetChoices('RyanChen1209/Brain-Tumor-Classification-MRI');
  assert.equal(meta7.choices.length, 7);
  assert.ok(meta7.choices.includes('Glioblastoma'));
});

test('isLesion.fromHF evaluates live MRI slices across TypeSafe Jev and local OpenJev engines', async () => {
  // 1. Evaluate a glioma MRI scan via TypeSafe Jev (row 0)
  const gliomaRes = await isLesion.fromHF('Hemg/Brain-Tumor-MRI-Dataset', {
    row: 0,
    engine: 'jev',
    apiKey: TEST_API_KEY
  });
  assert.equal(gliomaRes.groundTruth, 'glioma');
  assert.equal(gliomaRes.lesion, true);
  assert.equal(gliomaRes.choice, 'glioma');
  assert.ok(gliomaRes.probability > 0.7);
  assert.ok(gliomaRes.box && gliomaRes.box.top >= 0 && gliomaRes.box.bottom <= 1);
  assert.ok(gliomaRes.severity.score > 1.5);

  // 2. Re-evaluate the same slice with local OpenJev engine via .withEngine('openjev')
  const localRes = await gliomaRes.withEngine('openjev');
  assert.equal(localRes.model, 'openjev:exemplar-fingerprint');
  assert.equal(typeof localRes.lesion, 'boolean');

  // 3. Evaluate a normal no-tumor MRI scan via TypeSafe Jev (row 5305)
  const normalRes = await isLesion.fromHF('Hemg/Brain-Tumor-MRI-Dataset', {
    row: 5305,
    engine: 'jev',
    apiKey: TEST_API_KEY
  });
  assert.equal(normalRes.groundTruth, 'notumor');
  assert.equal(normalRes.lesion, false);
  assert.equal(normalRes.choice, 'notumor');
  assert.equal(normalRes.box, null);
});
