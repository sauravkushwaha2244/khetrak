/**
 * severity.js – KhetRak Severity Estimation Engine
 * ──────────────────────────────────────────────────────────────────────────────
 * Analyses a leaf image using canvas pixel data to estimate what percentage
 * of the leaf area shows disease symptoms.
 *
 * Approach:
 *   1. Extract pixels from the image via an offscreen canvas.
 *   2. Segment the leaf from the background (remove near-white/near-black pixels).
 *   3. Within leaf pixels, classify each as "healthy green" or "symptomatic"
 *      (brown, yellow, black, rust-coloured) using HSL thresholds.
 *   4. Severity % = symptomatic pixels / total leaf pixels × 100.
 *
 * Severity stages:
 *   Trace   0 – 10%   → preventive action only
 *   Mild   10 – 25%   → organic/biological treatment
 *   Moderate 25 – 50% → biological + chemical
 *   Severe 50 – 75%   → immediate chemical intervention
 *   Critical >75%     → salvage strategy + crop insurance advisory
 */

export const SeverityEngine = (() => {

  // ── HSL conversion ─────────────────────────────────────────────────────────
  function rgbToHsl(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;
    if (max === min) {
      h = s = 0;
    } else {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch (max) {
        case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
        case g: h = ((b - r) / d + 2) / 6; break;
        case b: h = ((r - g) / d + 4) / 6; break;
      }
    }
    return [h * 360, s * 100, l * 100];
  }

  // ── Pixel classification ───────────────────────────────────────────────────
  /**
   * Returns:
   *   'background'  – sky, soil, white background (exclude from analysis)
   *   'healthy'     – green leaf tissue
   *   'symptomatic' – diseased tissue (brown, yellow, rust, black spots)
   */
  function classifyPixel(r, g, b) {
    const [h, s, l] = rgbToHsl(r, g, b);

    // Background: very bright or very dark or very low saturation
    if (l > 88 || l < 6 || s < 8) return 'background';

    // Soil/dirt: dark brown tones that are likely background
    if (h >= 15 && h <= 40 && l < 28 && s < 35) return 'background';

    // Healthy green: hue 70–165°, reasonable saturation and lightness
    if (h >= 70 && h <= 165 && s >= 15 && l >= 12 && l <= 80) return 'healthy';

    // Yellow chlorosis: hue 45–70°, moderate saturation
    if (h >= 45 && h < 70 && s >= 20 && l >= 35) return 'symptomatic';

    // Brown / rust lesions: hue 15–45°
    if (h >= 15 && h < 45 && s >= 15 && l >= 18) return 'symptomatic';

    // Black necrotic spots: very dark with any hue
    if (l < 20 && s >= 5) return 'symptomatic';

    // Grey mold / powdery mildew: low saturation, mid lightness
    if (s < 20 && l >= 40 && l <= 75) return 'symptomatic';

    // Reddish purple (anthocyanin stress): hue 280–340°
    if ((h >= 280 || h <= 15) && s >= 20 && l >= 20) return 'symptomatic';

    return 'background'; // catch-all
  }

  // ── Main analysis function ─────────────────────────────────────────────────
  /**
   * @param {string} imageSrc – data URL or object URL
   * @returns {Promise<SeverityResult>}
   */
  async function analyse(imageSrc) {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = imageSrc;
    });

    // Downscale for performance (max 320px on longer side)
    const scale   = Math.min(320 / Math.max(img.width, img.height), 1);
    const W       = Math.round(img.width  * scale);
    const H       = Math.round(img.height * scale);

    const canvas  = document.createElement('canvas');
    canvas.width  = W;
    canvas.height = H;
    const ctx     = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, W, H);

    const { data } = ctx.getImageData(0, 0, W, H);

    let healthy     = 0;
    let symptomatic = 0;

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const cls = classifyPixel(r, g, b);
      if (cls === 'healthy')     healthy++;
      else if (cls === 'symptomatic') symptomatic++;
    }

    const leafPixels = healthy + symptomatic;
    if (leafPixels < 50) {
      // Couldn't detect a leaf – return unknown
      return { percentage: null, stage: 'unknown', healthy, symptomatic, leafPixels };
    }

    const percentage = (symptomatic / leafPixels) * 100;
    return {
      percentage: Math.round(percentage * 10) / 10,
      stage: getStage(percentage),
      healthy,
      symptomatic,
      leafPixels,
    };
  }

  // ── Severity stage mapping ─────────────────────────────────────────────────
  function getStage(pct) {
    if (pct <= 10)  return 'trace';
    if (pct <= 25)  return 'mild';
    if (pct <= 50)  return 'moderate';
    if (pct <= 75)  return 'severe';
    return 'critical';
  }

  const STAGE_META = {
    trace:    { label: 'Trace',    emoji: '🟢', color: '#22c55e', urgency: 'Monitor only'              },
    mild:     { label: 'Mild',     emoji: '🟡', color: '#84cc16', urgency: 'Preventive treatment'      },
    moderate: { label: 'Moderate', emoji: '🟠', color: '#f59e0b', urgency: 'Treat within 3–5 days'     },
    severe:   { label: 'Severe',   emoji: '🔴', color: '#ef4444', urgency: 'Immediate intervention'    },
    critical: { label: 'Critical', emoji: '💀', color: '#7f1d1d', urgency: 'Emergency – salvage mode'  },
    unknown:  { label: 'Unknown',  emoji: '❓', color: '#6b7280', urgency: 'Retake with clearer photo' },
  };

  function getStageMeta(stage) {
    return STAGE_META[stage] || STAGE_META.unknown;
  }

  return { analyse, getStage, getStageMeta };
})();


// ─── Trend Tracker ─────────────────────────────────────────────────────────────
export const TrendTracker = (() => {
  const STORE_KEY = 'khetrak-trends';

  function load() {
    return JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
  }
  function save(data) {
    localStorage.setItem(STORE_KEY, JSON.stringify(data));
  }

  /**
   * Add a severity reading for a crop+disease combination.
   * @param {string} cropKey  – e.g. "Millet_Downy Mildew"
   * @param {number} pct      – severity percentage
   */
  function addReading(cropKey, pct) {
    const data = load();
    if (!data[cropKey]) data[cropKey] = [];
    data[cropKey].push({ ts: Date.now(), pct });
    // Keep last 10 readings
    if (data[cropKey].length > 10) data[cropKey] = data[cropKey].slice(-10);
    save(data);
  }

  function getReadings(cropKey) {
    const data = load();
    return data[cropKey] || [];
  }

  /**
   * Compute trend direction from readings.
   * Returns 'worsening', 'improving', 'stable', or 'insufficient'.
   */
  function getTrend(cropKey) {
    const readings = getReadings(cropKey);
    if (readings.length < 2) return 'insufficient';
    const last  = readings[readings.length - 1].pct;
    const prev  = readings[readings.length - 2].pct;
    const delta = last - prev;
    if (delta >  5) return 'worsening';
    if (delta < -5) return 'improving';
    return 'stable';
  }

  return { addReading, getReadings, getTrend };
})();


// ─── Trend Chart (vanilla canvas) ─────────────────────────────────────────────
export function renderTrendChart(canvasEl, readings) {
  if (!canvasEl || readings.length === 0) return;

  const W   = canvasEl.width  = canvasEl.offsetWidth  || 320;
  const H   = canvasEl.height = canvasEl.offsetHeight || 120;
  const ctx = canvasEl.getContext('2d');

  // Background
  ctx.clearRect(0, 0, W, H);

  const PAD   = { top: 14, right: 16, bottom: 28, left: 36 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top  - PAD.bottom;

  const values = readings.map(r => r.pct);
  const maxV   = Math.max(...values, 25); // minimum scale of 25%
  const minV   = 0;

  // Grid lines
  ctx.strokeStyle = 'rgba(99,190,120,0.1)';
  ctx.lineWidth   = 1;
  [0, 25, 50, 75, 100].forEach(v => {
    if (v > maxV + 5) return;
    const y = PAD.top + plotH - (v / maxV) * plotH;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(W - PAD.right, y);
    ctx.stroke();
    // Label
    ctx.fillStyle = 'rgba(107,114,128,0.8)';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`${v}%`, PAD.left - 4, y + 3);
  });

  if (values.length < 2) {
    // Single point – show dot
    const x = PAD.left + plotW / 2;
    const y = PAD.top + plotH - (values[0] / maxV) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#22c55e';
    ctx.fill();
    return;
  }

  // Area fill
  const gradient = ctx.createLinearGradient(0, PAD.top, 0, H - PAD.bottom);
  gradient.addColorStop(0,   'rgba(34,197,94,0.25)');
  gradient.addColorStop(1,   'rgba(34,197,94,0.01)');

  const xs = values.map((_, i) => PAD.left + (i / (values.length - 1)) * plotW);
  const ys = values.map(v   => PAD.top  + plotH - (v / maxV) * plotH);

  ctx.beginPath();
  ctx.moveTo(xs[0], ys[0]);
  for (let i = 1; i < xs.length; i++) {
    // Smooth bezier
    const cpx = (xs[i - 1] + xs[i]) / 2;
    ctx.bezierCurveTo(cpx, ys[i - 1], cpx, ys[i], xs[i], ys[i]);
  }
  ctx.lineTo(xs[xs.length - 1], H - PAD.bottom);
  ctx.lineTo(xs[0], H - PAD.bottom);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(xs[0], ys[0]);
  for (let i = 1; i < xs.length; i++) {
    const cpx = (xs[i - 1] + xs[i]) / 2;
    ctx.bezierCurveTo(cpx, ys[i - 1], cpx, ys[i], xs[i], ys[i]);
  }
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth   = 2.5;
  ctx.stroke();

  // Dots + latest value label
  xs.forEach((x, i) => {
    ctx.beginPath();
    ctx.arc(x, ys[i], i === xs.length - 1 ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fillStyle   = '#22c55e';
    ctx.strokeStyle = '#060c14';
    ctx.lineWidth   = 2;
    ctx.fill();
    ctx.stroke();
  });

  // X-axis date labels (first, middle, last)
  ctx.fillStyle  = 'rgba(107,114,128,0.8)';
  ctx.font       = '8px Inter, sans-serif';
  ctx.textAlign  = 'center';
  const labelIdxs = [0, Math.floor(values.length / 2), values.length - 1];
  labelIdxs.forEach(i => {
    const d = new Date(readings[i].ts);
    const label = `${d.getDate()}/${d.getMonth() + 1}`;
    ctx.fillText(label, xs[i], H - PAD.bottom + 14);
  });
}
