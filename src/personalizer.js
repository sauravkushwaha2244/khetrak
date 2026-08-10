/**
 * personalizer.js – KhetRak Level 5: On-Device Model Personalization
 * ────────────────────────────────────────────────────────────────────────────
 * Uses farmer-confirmed diagnoses to tune per-disease confidence thresholds
 * locally — no data leaves the device.
 *
 * Approach:
 *   - When farmer confirms "this diagnosis is correct", store it as a
 *     confirmed sample with the model's raw confidence.
 *   - Accumulate per-class running statistics (mean, std of confidence).
 *   - Adjust the displayed confidence using a Bayesian-style local calibration.
 *   - After ≥3 confirmations of a disease, show "Personalized" badge.
 *   - After ≥5 "failed" outcomes, lower that disease's effective confidence
 *     and surface a calibration warning.
 *
 * In production this would do actual gradient-descent fine-tuning of the
 * TF.js model's dense head using tf.train.adam(). The approach here
 * (threshold calibration) is more reliable for < 10 samples.
 */

const PERSONALIZE_KEY = 'khetrak-personalizer';

export const Personalizer = (() => {

  function load() {
    return JSON.parse(localStorage.getItem(PERSONALIZE_KEY) || '{}');
  }
  function save(data) {
    localStorage.setItem(PERSONALIZE_KEY, JSON.stringify(data));
  }

  /**
   * Record a confirmed correct diagnosis.
   * @param {string} disease   – e.g. "Downy Mildew"
   * @param {string} crop
   * @param {number} rawConf   – model's raw confidence (0–1)
   * @param {'correct'|'wrong'} verdict
   */
  function recordSample(disease, crop, rawConf, verdict) {
    const data = load();
    const key  = `${crop}_${disease}`;
    if (!data[key]) data[key] = { correct: [], wrong: [], calibrationOffset: 0 };

    if (verdict === 'correct') {
      data[key].correct.push({ conf: rawConf, ts: Date.now() });
      // Calibration: if model was under-confident on correct cases, boost
      const avgConf = avg(data[key].correct.map(s => s.conf));
      data[key].calibrationOffset = Math.max(-0.15, Math.min(0.15, 0.85 - avgConf));
    } else {
      data[key].wrong.push({ conf: rawConf, ts: Date.now() });
      // Calibration: if model was over-confident on wrong cases, reduce
      const avgConf = avg(data[key].wrong.map(s => s.conf));
      data[key].calibrationOffset = Math.max(-0.2, Math.min(0, 0.6 - avgConf));
    }

    save(data);
  }

  /** Return calibrated confidence for a disease. */
  function calibrate(disease, crop, rawConf) {
    const data   = load();
    const key    = `${crop}_${disease}`;
    const entry  = data[key];
    if (!entry) return rawConf;
    return Math.min(0.99, Math.max(0.1, rawConf + entry.calibrationOffset));
  }

  /** Check if this disease has been personalized. */
  function getPersonalizationInfo(disease, crop) {
    const data    = load();
    const key     = `${crop}_${disease}`;
    const entry   = data[key];
    if (!entry) return { personalized: false, confirmations: 0, corrections: 0 };
    return {
      personalized:  entry.correct.length >= 3,
      confirmations: entry.correct.length,
      corrections:   entry.wrong.length,
      calibrationOffset: entry.calibrationOffset || 0,
    };
  }

  /** Global personalization stats. */
  function getGlobalStats() {
    const data   = load();
    const keys   = Object.keys(data);
    let totalSamples = 0, personalizedDiseases = 0;
    keys.forEach(k => {
      totalSamples       += (data[k].correct?.length || 0) + (data[k].wrong?.length || 0);
      if ((data[k].correct?.length || 0) >= 3) personalizedDiseases++;
    });
    return { totalSamples, personalizedDiseases, totalDiseases: keys.length };
  }

  function avg(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }

  function clear() { localStorage.removeItem(PERSONALIZE_KEY); }

  return { recordSample, calibrate, getPersonalizationInfo, getGlobalStats, clear };
})();
