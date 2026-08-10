/**
 * yield.js – KhetRak Level 5: Yield Loss Calculator
 * ────────────────────────────────────────────────────────────────────────────
 * Translates severity % into real economic impact in ₹.
 *
 * Formula:
 *   Yield loss % = base_loss(stage) + severity_scaling
 *   Quintals lost = (yield_loss% / 100) × typical_yield(crop) × area_acres
 *   ₹ loss = quintals_lost × current_mandi_price
 *
 * Typical yields (quintal/acre) from ICAR data:
 *   Millet:     4–8 q/acre  → use 6
 *   Pigeon Pea: 4–7 q/acre  → use 5
 *   Sorghum:    6–10 q/acre → use 8
 */

// Typical farm yield (quintals per acre, ICAR averages)
const TYPICAL_YIELD = {
  'Millet':     6,
  'Pigeon Pea': 5,
  'Sorghum':    8,
};

// Default mandi prices (₹/quintal) — editable by user
export const DEFAULT_MANDI_PRICES = {
  'Millet':     '2350',
  'Pigeon Pea': '7000',
  'Sorghum':    '3200',
};

// Yield loss % by severity stage (conservative mid-range from literature)
const STAGE_LOSS = {
  trace:    { min: 0,  max: 5,   typical: 2   },
  mild:     { min: 5,  max: 15,  typical: 10  },
  moderate: { min: 15, max: 35,  typical: 25  },
  severe:   { min: 35, max: 60,  typical: 48  },
  critical: { min: 60, max: 90,  typical: 72  },
  unknown:  { min: 0,  max: 0,   typical: 0   },
};

// Govt scheme eligibility thresholds
const SCHEME_THRESHOLDS = {
  PMFBY_CLAIM: 33,      // PM Fasal Bima Yojana – claim if >33% crop loss
  KRISHI_KENDRA: 25,    // Recommend visiting Krishi Kendra if >25%
  SDRF_RELIEF: 50,      // State Disaster Relief Fund if >50%
};

export const YieldCalculator = (() => {

  /**
   * Calculate estimated yield and economic loss.
   * @param {string} crop
   * @param {string} stage          – severity stage key
   * @param {number} severityPct    – pixel-level severity %
   * @param {number} areaAcres      – farmer's field area in acres
   * @param {number} mandiPrice     – current mandi price in ₹/quintal
   * @returns {YieldResult}
   */
  function calculate(crop, stage, severityPct, areaAcres, mandiPrice) {
    const stageLoss    = STAGE_LOSS[stage] || STAGE_LOSS.unknown;
    // Blend stage-based loss with actual pixel severity for better accuracy
    const pixelFactor  = severityPct !== null ? severityPct / 100 : 0.5;
    const blendedLoss  = stageLoss.typical * 0.6 + pixelFactor * (stageLoss.max - stageLoss.min) * 0.4 + stageLoss.min * 0.4;
    const yieldLossPct = Math.min(90, Math.round(blendedLoss));

    const typicalYield  = TYPICAL_YIELD[crop] || 6;
    const totalYield    = typicalYield * areaAcres;
    const quintalsLost  = (yieldLossPct / 100) * totalYield;
    const rupLoss       = quintalsLost * mandiPrice;
    const rupSaved      = ((stageLoss.max - yieldLossPct) / 100) * totalYield * mandiPrice;

    // Govt scheme eligibility
    const schemes = [];
    if (yieldLossPct >= SCHEME_THRESHOLDS.PMFBY_CLAIM)   schemes.push({ name: 'PM Fasal Bima Yojana', code: 'PMFBY',   desc: 'File insurance claim – eligible above 33% crop loss' });
    if (yieldLossPct >= SCHEME_THRESHOLDS.SDRF_RELIEF)   schemes.push({ name: 'SDRF Relief',          code: 'SDRF',    desc: 'State Disaster Relief Fund – eligible above 50% loss' });
    if (yieldLossPct >= SCHEME_THRESHOLDS.KRISHI_KENDRA) schemes.push({ name: 'Krishi Kendra Visit',  code: 'KK',      desc: 'Free extension advisory from block agriculture officer' });

    return {
      yieldLossPct,
      totalYield: Math.round(totalYield * 10) / 10,
      quintalsLost: Math.round(quintalsLost * 10) / 10,
      rupLoss:    Math.round(rupLoss),
      rupSaved:   Math.round(rupSaved),
      schemes,
      breakdown: {
        stageLossRange: `${stageLoss.min}–${stageLoss.max}%`,
        pixelSeverity:  severityPct !== null ? `${severityPct}%` : 'unknown',
        blended:        `${yieldLossPct}%`,
      },
    };
  }

  function formatINR(amount) {
    if (amount >= 100000) return `₹${(amount / 100000).toFixed(1)} lakh`;
    if (amount >= 1000)   return `₹${(amount / 1000).toFixed(1)}K`;
    return `₹${amount}`;
  }

  return { calculate, formatINR, DEFAULT_MANDI_PRICES, TYPICAL_YIELD };
})();
