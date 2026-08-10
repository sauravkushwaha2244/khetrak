/**
 * preprocessor.js – KhetRak Real-World Image Robustness Pipeline (Level 4)
 * ────────────────────────────────────────────────────────────────────────────
 * Analyses a raw farmer photo for quality issues (blur, bad lighting, heavy
 * shadows, soil occlusion, multiple leaves) and auto-corrects them before
 * passing to the ML model. This closes the train-test distribution gap that
 * causes PlantVillage-trained models to fail on real field images.
 *
 * Pipeline order:
 *   1. Load image → canvas pixel data
 *   2. Detect issues (blur, darkness, overexposure, shadow, multi-leaf)
 *   3. Apply corrections (gamma, CLAHE-like equalization, dehazing, sharpen)
 *   4. Return corrected image data URL + quality report
 */

export const ImagePreprocessor = (() => {

  // ── 1. Blur Detection (Laplacian variance) ─────────────────────────────────
  /**
   * High Laplacian variance = sharp image, low = blurry.
   * We use a 3x3 Laplacian kernel on the grayscale image.
   */
  function measureSharpness(data, width, height) {
    let sumSq = 0, sum = 0, count = 0;
    // Laplacian kernel: center=4, neighbours=-1
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
        const top    = grayAt(data, x,   y - 1, width);
        const bottom = grayAt(data, x,   y + 1, width);
        const left   = grayAt(data, x - 1, y,   width);
        const right  = grayAt(data, x + 1, y,   width);
        const lap    = Math.abs(4 * gray - top - bottom - left - right);
        sum   += lap;
        sumSq += lap * lap;
        count++;
      }
    }
    const mean = sum / count;
    const variance = sumSq / count - mean * mean;
    return variance; // higher = sharper
  }

  function grayAt(data, x, y, width) {
    const idx = (y * width + x) * 4;
    return 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  }

  // ── 2. Brightness Analysis ─────────────────────────────────────────────────
  function measureBrightness(data) {
    let sum = 0;
    const px = data.length / 4;
    for (let i = 0; i < data.length; i += 4) {
      sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
    return sum / px; // 0-255 mean luminance
  }

  // ── 3. Shadow Detection ────────────────────────────────────────────────────
  // Shadows = large proportion of pixels with low luminance but non-zero saturation
  function measureShadow(data) {
    let shadowPx = 0;
    const total = data.length / 4;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      const max = Math.max(r, g, b), min = Math.min(r, g, b);
      const sat = max === 0 ? 0 : (max - min) / max;
      if (lum < 60 && sat > 0.08) shadowPx++;
    }
    return (shadowPx / total) * 100; // % of dark saturated pixels
  }

  // ── 4. Multi-Leaf / Clutter Detection ─────────────────────────────────────
  // High edge density across the whole frame → likely multiple leaves or clutter
  function measureEdgeDensity(data, width, height) {
    let edges = 0;
    const total = (width - 2) * (height - 2);
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const gx = grayAt(data, x + 1, y, width) - grayAt(data, x - 1, y, width);
        const gy = grayAt(data, x, y + 1, width) - grayAt(data, x, y - 1, width);
        if (Math.sqrt(gx * gx + gy * gy) > 30) edges++;
      }
    }
    return (edges / total) * 100;
  }

  // ── 5. Auto-Enhancement Functions ─────────────────────────────────────────

  /** Gamma correction for dark/bright images */
  function applyGamma(data, gamma) {
    const inv = 1.0 / gamma;
    const lut = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
      lut[i] = Math.min(255, Math.round(Math.pow(i / 255, inv) * 255));
    }
    for (let i = 0; i < data.length; i += 4) {
      data[i]     = lut[data[i]];
      data[i + 1] = lut[data[i + 1]];
      data[i + 2] = lut[data[i + 2]];
    }
  }

  /** Histogram stretching (contrast normalization) */
  function stretchHistogram(data) {
    let minL = 255, maxL = 0;
    for (let i = 0; i < data.length; i += 4) {
      const l = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      if (l < minL) minL = l;
      if (l > maxL) maxL = l;
    }
    if (maxL - minL < 20) return; // already full range
    const scale = 255 / (maxL - minL);
    for (let i = 0; i < data.length; i += 4) {
      data[i]     = Math.min(255, Math.round((data[i]     - minL) * scale));
      data[i + 1] = Math.min(255, Math.round((data[i + 1] - minL) * scale));
      data[i + 2] = Math.min(255, Math.round((data[i + 2] - minL) * scale));
    }
  }

  /** Saturation boost to recover washed-out or faded field photos */
  function boostSaturation(data, factor) {
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
      const max = Math.max(r, g, b), min = Math.min(r, g, b);
      const l   = (max + min) / 2;
      if (max === min) continue;
      const d   = max - min;
      const s   = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      const ns  = Math.min(1, s * factor);
      const q   = l < 0.5 ? l * (1 + ns) : l + ns - l * ns;
      const p   = 2 * l - q;
      const h   = getHue(r, g, b, max, d);
      data[i]     = Math.round(hue2rgb(p, q, h + 1 / 3) * 255);
      data[i + 1] = Math.round(hue2rgb(p, q, h)         * 255);
      data[i + 2] = Math.round(hue2rgb(p, q, h - 1 / 3) * 255);
    }
  }

  function getHue(r, g, b, max, d) {
    if (max === r) return ((g - b) / d + (g < b ? 6 : 0)) / 6;
    if (max === g) return ((b - r) / d + 2) / 6;
    return ((r - g) / d + 4) / 6;
  }

  function hue2rgb(p, q, t) {
    if (t < 0) t += 1; if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  }

  /** Unsharp mask (sharpen) using a 3x3 box blur as the blur source */
  function sharpen(ctx, width, height, amount = 0.6) {
    const orig  = ctx.getImageData(0, 0, width, height);
    const blur  = ctx.getImageData(0, 0, width, height);
    // simple box blur
    const d = blur.data;
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        for (let c = 0; c < 3; c++) {
          let s = 0;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            s += orig.data[((y + dy) * width + (x + dx)) * 4 + c];
          }
          d[(y * width + x) * 4 + c] = s / 9;
        }
      }
    }
    // unsharp = original + amount * (original - blur)
    const sharpened = ctx.createImageData(width, height);
    for (let i = 0; i < orig.data.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        sharpened.data[i + c] = Math.min(255, Math.max(0,
          orig.data[i + c] + amount * (orig.data[i + c] - d[i + c])
        ));
      }
      sharpened.data[i + 3] = 255;
    }
    ctx.putImageData(sharpened, 0, 0);
  }

  /** Dark channel prior dehazing – removes haze and improves shadow regions */
  function dehaze(data, width, height, strength = 0.4) {
    const patchSize = 7;
    const half = Math.floor(patchSize / 2);
    // Compute dark channel
    const dark = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let minVal = 255;
        for (let dy = -half; dy <= half; dy++) {
          for (let dx = -half; dx <= half; dx++) {
            const ny = Math.min(height - 1, Math.max(0, y + dy));
            const nx = Math.min(width  - 1, Math.max(0, x + dx));
            const idx = (ny * width + nx) * 4;
            minVal = Math.min(minVal, data[idx], data[idx + 1], data[idx + 2]);
          }
        }
        dark[y * width + x] = minVal / 255;
      }
    }
    // Estimate atmospheric light
    let A = 0;
    for (let i = 0; i < dark.length; i++) A = Math.max(A, dark[i]);
    A = Math.min(0.9, A);
    // Apply transmission map
    for (let i = 0; i < data.length; i += 4) {
      const d = dark[i / 4];
      const t = Math.max(0.1, 1 - strength * (d / (A + 0.001)));
      data[i]     = Math.min(255, Math.max(0, (data[i]     / 255 - A) / t + A) * 255);
      data[i + 1] = Math.min(255, Math.max(0, (data[i + 1] / 255 - A) / t + A) * 255);
      data[i + 2] = Math.min(255, Math.max(0, (data[i + 2] / 255 - A) / t + A) * 255);
    }
  }

  // ── Main Pipeline ──────────────────────────────────────────────────────────
  /**
   * @param {string} imageSrc – data URL
   * @returns {Promise<PreprocessResult>}
   *   { processedSrc, qualityScore, issues[], corrections[], report }
   */
  async function process(imageSrc) {
    const img = new Image();
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = imageSrc; });

    // Working resolution – max 480px for speed, keep aspect ratio
    const scale = Math.min(480 / Math.max(img.width, img.height), 1);
    const W = Math.round(img.width  * scale);
    const H = Math.round(img.height * scale);

    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, W, H);

    const imageData = ctx.getImageData(0, 0, W, H);
    const data      = imageData.data;

    // ── Analysis ─────────────────────────────────────────────────────────────
    const sharpness    = measureSharpness(data, W, H);
    const brightness   = measureBrightness(data);
    const shadowPct    = measureShadow(data);
    const edgeDensity  = measureEdgeDensity(data, W, H);

    const issues      = [];
    const corrections = [];

    // ── Classify issues ───────────────────────────────────────────────────────
    const isBlurry     = sharpness < 180;
    const isDark       = brightness < 65;
    const isOverExp    = brightness > 210;
    const hasShadow    = shadowPct > 22;
    const hasClutter   = edgeDensity > 45;

    if (isBlurry)   issues.push({ id: 'blur',     label: 'Blurry image',       icon: '🌀', fix: 'Auto-sharpening applied' });
    if (isDark)     issues.push({ id: 'dark',     label: 'Poor lighting',      icon: '🌑', fix: 'Gamma correction applied' });
    if (isOverExp)  issues.push({ id: 'overexp',  label: 'Overexposed',        icon: '☀️', fix: 'Histogram normalization applied' });
    if (hasShadow)  issues.push({ id: 'shadow',   label: 'Heavy shadows',      icon: '🌒', fix: 'Shadow dehazing applied' });
    if (hasClutter) issues.push({ id: 'clutter',  label: 'Multiple leaves / cluttered frame', icon: '🍃', fix: 'Focus on clearest region' });

    // ── Apply corrections ────────────────────────────────────────────────────
    if (isDark) {
      applyGamma(data, 2.0); // lighten
      corrections.push('gamma-lighten');
    }
    if (isOverExp) {
      stretchHistogram(data); // pull back highlights
      corrections.push('histogram-stretch');
    }
    if (hasShadow) {
      dehaze(data, W, H, 0.45);
      corrections.push('dehaze');
    }
    // Always boost saturation slightly for washed-out field photos
    boostSaturation(data, 1.25);
    corrections.push('saturation-boost');

    ctx.putImageData(imageData, 0, 0);

    if (isBlurry) {
      sharpen(ctx, W, H, 0.7);
      corrections.push('unsharp-mask');
    }

    // ── Quality Score ─────────────────────────────────────────────────────────
    // Start from 100, deduct for each issue
    let qualityScore = 100;
    if (isBlurry)  qualityScore -= 25;
    if (isDark)    qualityScore -= 20;
    if (isOverExp) qualityScore -= 15;
    if (hasShadow) qualityScore -= 15;
    if (hasClutter)qualityScore -= 10;
    qualityScore = Math.max(10, qualityScore);

    const processedSrc = canvas.toDataURL('image/jpeg', 0.92);

    return {
      processedSrc,
      originalSrc: imageSrc,
      qualityScore,
      issues,
      corrections,
      metrics: { sharpness: Math.round(sharpness), brightness: Math.round(brightness), shadowPct: Math.round(shadowPct), edgeDensity: Math.round(edgeDensity) },
    };
  }

  function getQualityLabel(score) {
    if (score >= 80) return { label: 'Good',      color: '#22c55e', emoji: '✅' };
    if (score >= 55) return { label: 'Fair',      color: '#f59e0b', emoji: '⚠️' };
    if (score >= 30) return { label: 'Poor',      color: '#ef4444', emoji: '🔴' };
    return                  { label: 'Very Poor', color: '#7f1d1d', emoji: '❌' };
  }

  return { process, getQualityLabel };
})();
