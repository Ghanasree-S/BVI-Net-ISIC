import { OrganType, PredictionResult } from '../types';
import { ORGAN_CONFIGS } from '../data/organData';

const RASTER_DATA_URL = /^data:image\/(png|jpe?g|webp|bmp);base64,/i;

// The backend decodes images with OpenCV, which can't read SVG (the built-in
// sample cases) or arbitrary URLs -- rasterize anything else to a PNG data URL.
async function toRasterDataUrl(src: string): Promise<string> {
  // Raster images and raw .npy uploads (any non-SVG data URL) go to the backend as-is.
  if (RASTER_DATA_URL.test(src) || (src.startsWith('data:') && !src.startsWith('data:image/svg'))) return src;
  if (/\.npy$/i.test(src)) {
    // Bundled brain sample: fetch the raw 4-channel array and send it as base64.
    const buf = new Uint8Array(await (await fetch(src)).arrayBuffer());
    let bin = '';
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
    return `data:application/octet-stream;base64,${btoa(bin)}`;
  }
  const img = new Image();
  img.crossOrigin = 'anonymous';
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('Could not load image'));
    img.src = src;
  });
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth || 256;
  canvas.height = img.naturalHeight || 256;
  canvas.getContext('2d')!.drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/png');
}

/**
 * Runs real BVI-Net inference on a medical image.
 * Calls POST /predict (proxied by server.ts to the FastAPI backend) with
 * { organ_type, image }. Throws with the backend's message on failure --
 * no simulated fallback, so every mask shown comes from the trained model.
 */
export async function predictLesion(
  organType: OrganType,
  imageBase64OrUrl: string,
  fileName?: string
): Promise<PredictionResult> {
  const organConfig = ORGAN_CONFIGS[organType];
  const image = await toRasterDataUrl(imageBase64OrUrl);

  const response = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ organ_type: organType, image }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Prediction failed (HTTP ${response.status})`);
  }

  return {
    id: `pred_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
    organ_type: organType,
    status: organConfig.isAvailable ? 'production_ready' : 'experimental_checkpoint',
    // .npy MRI slices can't be shown by <img>; the backend returns a FLAIR preview.
    original_image_url: data.input_preview || image,
    mask_base64: data.mask_base64,
    metrics: data.metrics,
    model_info: data.model_info,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    fileName: fileName || `scan_${organType}_${Date.now()}.png`,
  };
}
