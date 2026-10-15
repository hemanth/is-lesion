import fs from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';

const execFileAsync = promisify(execFile);

/**
 * Curated clinical descriptions for common Hugging Face MRI dataset labels.
 */
export const CLINICAL_DESCRIPTIONS = {
  glioma: 'Intra-axial glioma tumor inside the brain parenchyma with irregular infiltrative borders or surrounding edema.',
  glioma_tumor: 'Intra-axial glioma tumor inside the brain parenchyma with irregular infiltrative borders or surrounding edema.',
  brain_glioma: 'Intra-axial glioma tumor inside the brain parenchyma with irregular infiltrative borders or surrounding edema.',
  glioblastoma: 'High-grade aggressive glioblastoma mass with necrotic core, irregular ring enhancement, and strong mass effect.',
  meningioma: 'Extra-axial well-circumscribed homogeneous dural-based meningioma mass adjacent to the meninges.',
  meningioma_tumor: 'Extra-axial well-circumscribed homogeneous dural-based meningioma mass adjacent to the meninges.',
  brain_menin: 'Extra-axial well-circumscribed homogeneous dural-based meningioma mass adjacent to the meninges.',
  pituitary: 'Focal well-defined adenoma or mass located in the sellar or suprasellar pituitary region at the base of the brain.',
  pituitary_tumor: 'Focal well-defined adenoma or mass located in the sellar or suprasellar pituitary region at the base of the brain.',
  notumor: 'Healthy normal brain MRI scan with clear ventricles, symmetrical cerebral hemispheres, and no tumor or abnormal lesion.',
  'no-tumor': 'Healthy normal brain MRI scan with clear ventricles, symmetrical cerebral hemispheres, and no tumor or abnormal lesion.',
  no_tumor: 'Healthy normal brain MRI scan with clear ventricles, symmetrical cerebral hemispheres, and no tumor or abnormal lesion.',
  normal: 'Healthy normal brain MRI scan with clear ventricles, symmetrical cerebral hemispheres, and no tumor or abnormal lesion.',
  metastatic: 'Circumscribed metastatic lesion often at the gray-white matter junction with disproportionate vasogenic edema.',
  schwannoma: 'Extra-axial vestibular or cranial nerve schwannoma mass.',
  brain_tumor: 'Visible intracranial tumor mass or abnormal hyperintense lesion.'
};

/**
 * Determine whether a label represents a normal/benign non-tumor scan.
 */
export function isNormalLabel(label) {
  if (!label) return false;
  const norm = String(label).toLowerCase().replace(/[\s_-]+/g, '');
  return norm === 'notumor' || norm === 'normal' || norm === 'healthy' || norm === 'negative' || norm === 'none';
}

/**
 * Parse width, height, and format from PNG or JPEG buffer.
 */
export function getImageDimensions(buf) {
  if (!Buffer.isBuffer(buf) || buf.length < 24) return null;

  // PNG signature: 89 50 4E 47 0D 0A 1A 0A
  if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4E && buf[3] === 0x47) {
    return {
      format: 'png',
      width: buf.readUInt32BE(16),
      height: buf.readUInt32BE(20)
    };
  }

  // JPEG signature: FF D8
  if (buf[0] === 0xFF && buf[1] === 0xD8) {
    let i = 2;
    while (i + 9 < buf.length) {
      if (buf[i] !== 0xFF) {
        i++;
        continue;
      }
      const marker = buf[i + 1];
      if (marker === 0xFF) {
        i++;
        continue;
      }
      // SOF0..SOF15 markers (excluding DHT 0xC4, JPG 0xC8, DAC 0xCC)
      if (marker >= 0xC0 && marker <= 0xCF && marker !== 0xC4 && marker !== 0xC8 && marker !== 0xCC) {
        return {
          format: 'jpg',
          height: buf.readUInt16BE(i + 5),
          width: buf.readUInt16BE(i + 7)
        };
      }
      if (marker === 0xD8 || marker === 0xD9 || (marker >= 0xD0 && marker <= 0xD7)) {
        i += 2;
        continue;
      }
      const segLen = buf.readUInt16BE(i + 2);
      if (segLen < 2) break;
      i += 2 + segLen;
    }
  }

  return null;
}

/**
 * Ensure an image buffer meets TypeSafe System One size requirements:
 * - Width & height between 256 and 1568 px
 * - Total pixels (width * height) between 78,400 (280x280) and 2,458,624 (1568x1568)
 */
export async function ensureValidImageBuffer(buf) {
  const dims = getImageDimensions(buf);
  if (!dims) {
    return { buffer: buf, format: 'jpg', width: 512, height: 512 };
  }

  const totalPixels = dims.width * dims.height;
  const needsUpscale = dims.width < 280 || dims.height < 280 || totalPixels < 78400;
  const needsDownscale = dims.width > 1568 || dims.height > 1568 || totalPixels > 2458624;

  if (!needsUpscale && !needsDownscale) {
    return { buffer: buf, format: dims.format, width: dims.width, height: dims.height };
  }

  // On macOS, use sips to resize cleanly into [384..1024]
  const tmpId = randomUUID();
  const ext = dims.format === 'png' ? '.png' : '.jpg';
  const tmpIn = path.join(os.tmpdir(), `iscancer-in-${tmpId}${ext}`);
  const tmpOut = path.join(os.tmpdir(), `iscancer-out-${tmpId}.jpg`);

  try {
    await fs.writeFile(tmpIn, buf);
    const targetH = needsUpscale ? Math.max(384, dims.height) : Math.min(1024, dims.height);
    const targetW = needsUpscale ? Math.max(384, dims.width) : Math.min(1024, dims.width);
    await execFileAsync('sips', ['-s', 'format', 'jpeg', '-z', String(targetH), String(targetW), tmpIn, '--out', tmpOut]);
    const resizedBuf = await fs.readFile(tmpOut);
    const newDims = getImageDimensions(resizedBuf) || { format: 'jpg', width: targetW, height: targetH };
    return { buffer: resizedBuf, format: 'jpg', width: newDims.width, height: newDims.height };
  } catch {
    return { buffer: buf, format: dims.format, width: dims.width, height: dims.height };
  } finally {
    await fs.unlink(tmpIn).catch(() => {});
    await fs.unlink(tmpOut).catch(() => {});
  }
}

/**
 * Load an image from URL, local file path, Buffer, ArrayBuffer, or data URL
 * and return a prepared base64 asset ready for TypeSafe Jev.
 */
export async function loadImageAsset(input) {
  let rawBuf;

  if (Buffer.isBuffer(input)) {
    rawBuf = input;
  } else if (input instanceof ArrayBuffer) {
    rawBuf = Buffer.from(input);
  } else if (ArrayBuffer.isView(input)) {
    rawBuf = Buffer.from(input.buffer, input.byteOffset, input.byteLength);
  } else if (typeof input === 'string') {
    if (input.startsWith('data:image/')) {
      const commaIdx = input.indexOf(',');
      rawBuf = Buffer.from(input.slice(commaIdx + 1), 'base64');
    } else if (input.startsWith('http://') || input.startsWith('https://')) {
      const res = await fetch(input);
      if (!res.ok) throw new Error(`Failed to fetch image URL (${res.status}): ${input}`);
      rawBuf = Buffer.from(await res.arrayBuffer());
    } else {
      rawBuf = await fs.readFile(input);
    }
  } else if (input && typeof input === 'object' && input.src) {
    return loadImageAsset(input.src);
  } else {
    throw new Error('Unsupported image input: provide a file path, URL, data URI, or Buffer.');
  }

  const prepared = await ensureValidImageBuffer(rawBuf);
  const ref = `mri-${randomUUID()}`;
  return {
    ref,
    width: prepared.width,
    height: prepared.height,
    format: prepared.format,
    data_base64: prepared.buffer.toString('base64'),
    asset: {
      type: 'image',
      format: prepared.format,
      data_base64: prepared.buffer.toString('base64')
    }
  };
}
