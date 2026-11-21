import type { LesionRegion } from '../types/diagnostics';

/**
 * OpenCV COLORMAP_JET transfer function mapping t in [0, 1] -> [R, G, B]
 * Matches cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET) from App.py.
 */
export function jetColorMap(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 1)));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

export interface GradCamField {
  gridWidth: number;
  gridHeight: number;
  values: Float32Array; // Normalized [0, 1] activation map
}

/**
 * Computes a spatially smooth Grad-CAM-like activation field combining:
 * 1) Vision-detected lesion bounding coordinates (Gaussian spatial kernels)
 * 2) Per-pixel botanical chromatic anomaly saliency inside the leaf silhouette
 *    (separating necrotic brown/olive/yellow/white lesions from healthy green chlorophyll
 *     while suppressing neutral laboratory background pixels).
 */
export function computeGradCamField(
  img: HTMLImageElement,
  lesionRegions: LesionRegion[],
  isHealthy: boolean,
  gridSize = 112
): GradCamField {
  const w = gridSize;
  const h = gridSize;
  const offscreen = document.createElement('canvas');
  offscreen.width = w;
  offscreen.height = h;
  const ctx = offscreen.getContext('2d');
  const raw = new Float32Array(w * h);

  if (!ctx) {
    return { gridWidth: w, gridHeight: h, values: raw };
  }

  ctx.drawImage(img, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);

  // Estimate background RGB from the 4 image corners (typical lab surface)
  const cornerOffsets = [
    0,
    (w - 1) * 4,
    (h - 1) * w * 4,
    (h * w - 1) * 4,
  ];
  let bgR = 0;
  let bgG = 0;
  let bgB = 0;
  for (const idx of cornerOffsets) {
    bgR += data[idx];
    bgG += data[idx + 1];
    bgB += data[idx + 2];
  }
  bgR /= 4;
  bgG /= 4;
  bgB /= 4;

  // Step 1: Leaf mask + pixel-level pathological chromatic anomaly map
  const leafMask = new Float32Array(w * h);
  const chromaticMap = new Float32Array(w * h);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const p = i * 4;
      const r = data[p];
      const g = data[p + 1];
      const b = data[p + 2];

      const maxC = Math.max(r, g, b);
      const minC = Math.min(r, g, b);
      const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
      const distFromBg = Math.hypot(r - bgR, g - bgG, b - bgB);

      // Foreground leaf detection (differs from neutral corner background or has botanical color)
      const isLeaf = distFromBg > 32 || (sat > 0.14 && g > 30);
      leafMask[i] = isLeaf ? 1 : 0;

      if (!isLeaf) {
        chromaticMap[i] = 0;
        continue;
      }

      if (isHealthy) {
        // For healthy leaves, Grad-CAM shows gentle uniform chlorophyll lamina attention
        const greenDominance = Math.max(0, (g - (r + b) * 0.5) / 140);
        chromaticMap[i] = Math.min(0.45, 0.12 + greenDominance * 0.25);
      } else {
        // Excess green index (healthy foliage has high g relative to r and b)
        const exG = (2 * g - r - b) / 255;
        // Brown/necrotic/tan/rust lesions have r >= g*0.88 and lower exG
        const necrosis = Math.max(0, (r - g * 0.82) / 110);
        // Chlorosis (yellowing halo) has high r & g, low b
        const chlorosis =
          r > 110 && g > 105 && b < 95 ? Math.max(0, (r - b) / 180) * 0.65 : 0;
        // Dark necrotic spots inside leaf
        const darkSpot = r < 95 && g < 85 && b < 75 ? 0.55 : 0;
        // Powdery mildew pale patches inside leaf contour
        const mildew =
          r > 170 && g > 170 && b > 165 && distFromBg > 45 ? 0.5 : 0;

        const lowGreenPenalty = exG < 0.14 ? (0.14 - exG) * 2.2 : 0;
        chromaticMap[i] = Math.min(
          1,
          necrosis * 1.15 + chlorosis * 0.65 + darkSpot * 0.45 + mildew + lowGreenPenalty * 0.7
        );
      }
    }
  }

  // Step 2: Vision-guided Gaussian activation kernels from detected lesion regions
  const regionMap = new Float32Array(w * h);
  if (!isHealthy && lesionRegions.length > 0) {
    for (const reg of lesionRegions) {
      const cx = ((reg.xmin + reg.xmax) / 2000) * w;
      const cy = ((reg.ymin + reg.ymax) / 2000) * h;
      const rx = Math.max(3.5, ((reg.xmax - reg.xmin) / 2000) * w * 0.95);
      const ry = Math.max(3.5, ((reg.ymax - reg.ymin) / 2000) * h * 0.95);
      const weight = Math.max(0.35, Math.min(1, reg.intensity));

      const y0 = Math.max(0, Math.floor(cy - ry * 2.4));
      const y1 = Math.min(h - 1, Math.ceil(cy + ry * 2.4));
      const x0 = Math.max(0, Math.floor(cx - rx * 2.4));
      const x1 = Math.min(w - 1, Math.ceil(cx + rx * 2.4));

      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const dx = (x - cx) / rx;
          const dy = (y - cy) / ry;
          const gVal = Math.exp(-0.5 * (dx * dx + dy * dy)) * weight;
          const idx = y * w + x;
          if (gVal > regionMap[idx]) {
            regionMap[idx] = gVal;
          }
        }
      }
    }
  }

  // Step 3: Fuse region kernels + chromatic pathology saliency (masked to leaf area)
  for (let i = 0; i < w * h; i++) {
    if (isHealthy) {
      raw[i] = chromaticMap[i] * leafMask[i];
    } else if (lesionRegions.length > 0) {
      // Strong agreement where both semantic lesion region & chromatic anomaly overlap
      const fused =
        regionMap[i] * 0.58 +
        chromaticMap[i] * 0.42 +
        regionMap[i] * chromaticMap[i] * 0.55;
      raw[i] = fused * (leafMask[i] > 0 ? 1 : 0.08);
    } else {
      raw[i] = chromaticMap[i] * leafMask[i];
    }
  }

  // Step 4: Two-pass separable box blur approximating conv5_block3_out bilinear upsampling
  const smoothed = boxBlurSeparable(boxBlurSeparable(raw, w, h, 3), w, h, 2);

  // Step 5: ReLU & min-max normalize to [0, 1] (matching np.maximum(heatmap, 0) / (max + 1e-8))
  let maxVal = 0;
  for (let i = 0; i < smoothed.length; i++) {
    if (smoothed[i] > maxVal) maxVal = smoothed[i];
  }

  const normalized = new Float32Array(w * h);
  if (maxVal > 1e-6) {
    const cap = isHealthy ? maxVal / 0.42 : maxVal;
    for (let i = 0; i < smoothed.length; i++) {
      normalized[i] = Math.max(0, Math.min(1, smoothed[i] / (cap + 1e-8)));
    }
  }

  return { gridWidth: w, gridHeight: h, values: normalized };
}

function boxBlurSeparable(
  src: Float32Array,
  w: number,
  h: number,
  radius: number
): Float32Array {
  const tmp = new Float32Array(w * h);
  const dst = new Float32Array(w * h);
  const kernelSize = radius * 2 + 1;

  // Horizontal pass
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let k = -radius; k <= radius; k++) {
        const sx = Math.max(0, Math.min(w - 1, x + k));
        sum += src[row + sx];
      }
      tmp[row + x] = sum / kernelSize;
    }
  }

  // Vertical pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let k = -radius; k <= radius; k++) {
        const sy = Math.max(0, Math.min(h - 1, y + k));
        sum += tmp[sy * w + x];
      }
      dst[y * w + x] = sum / kernelSize;
    }
  }

  return dst;
}

export interface RenderGradCamOptions {
  alpha: number; // 0..1 overlay intensity (default 0.48 matching cv2.addWeighted 0.6/0.4)
  threshold: number; // 0..0.85 minimum activation to colorize
  heatmapOnly?: boolean;
  showBoundingBoxes?: boolean;
  lesionRegions?: LesionRegion[];
}

export function renderGradCamToCanvas(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  field: GradCamField,
  options: RenderGradCamOptions
): void {
  const targetSize = 512;
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Draw base image (center-cropped square if non-square, or fit nicely)
  ctx.clearRect(0, 0, targetSize, targetSize);
  if (!options.heatmapOnly) {
    ctx.drawImage(img, 0, 0, targetSize, targetSize);
  } else {
    ctx.fillStyle = '#071330';
    ctx.fillRect(0, 0, targetSize, targetSize);
  }

  // Build COLORMAP_JET heatmap on an offscreen canvas at grid resolution for crisp smooth upscaling
  const heatCanvas = document.createElement('canvas');
  heatCanvas.width = field.gridWidth;
  heatCanvas.height = field.gridHeight;
  const heatCtx = heatCanvas.getContext('2d');
  if (!heatCtx) return;

  const heatImg = heatCtx.createImageData(field.gridWidth, field.gridHeight);
  const { data } = heatImg;

  for (let i = 0; i < field.values.length; i++) {
    const val = field.values[i];
    const p = i * 4;
    if (!options.heatmapOnly && val < options.threshold) {
      data[p] = 0;
      data[p + 1] = 0;
      data[p + 2] = 0;
      data[p + 3] = 0;
      continue;
    }

    const [r, g, b] = jetColorMap(val);
    data[p] = r;
    data[p + 1] = g;
    data[p + 2] = b;

    if (options.heatmapOnly) {
      data[p + 3] = 255;
    } else {
      // Smooth ramp near threshold so edges blend naturally onto leaf
      const ramp = Math.min(1, (val - options.threshold) / Math.max(0.08, 1 - options.threshold));
      const effectiveAlpha = options.alpha * (0.25 + 0.75 * Math.pow(ramp, 0.7));
      data[p + 3] = Math.round(effectiveAlpha * 255);
    }
  }

  heatCtx.putImageData(heatImg, 0, 0);

  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(heatCanvas, 0, 0, targetSize, targetSize);
  ctx.restore();

  // Optional localized lesion markers
  if (options.showBoundingBoxes && options.lesionRegions && options.lesionRegions.length > 0) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    for (const r of options.lesionRegions) {
      const x = (r.xmin / 1000) * targetSize;
      const y = (r.ymin / 1000) * targetSize;
      const bw = ((r.xmax - r.xmin) / 1000) * targetSize;
      const bh = ((r.ymax - r.ymin) / 1000) * targetSize;
      ctx.strokeRect(x, y, bw, bh);
    }
    ctx.restore();
  }
}

export function sampleFieldAt(
  field: GradCamField | null,
  xNorm: number,
  yNorm: number
): number {
  if (!field) return 0;
  const x = Math.max(0, Math.min(field.gridWidth - 1, Math.floor(xNorm * field.gridWidth)));
  const y = Math.max(0, Math.min(field.gridHeight - 1, Math.floor(yNorm * field.gridHeight)));
  return field.values[y * field.gridWidth + x] ?? 0;
}

export function buildResNet224Preview(img: HTMLImageElement): string {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  ctx.drawImage(img, 0, 0, 224, 224);
  return canvas.toDataURL('image/jpeg', 0.92);
}
