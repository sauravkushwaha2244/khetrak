/**
 * feedback.js – KhetRak Level 5: Outcome Tracking & Follow-up System
 * ────────────────────────────────────────────────────────────────────────────
 * Tracks what happens AFTER treatment is applied.
 * - Stores pending follow-ups in localStorage
 * - Surfaces "Did it work?" cards after 7 days
 * - Aggregates outcome data for personalization
 */

const FEEDBACK_KEY   = 'khetrak-feedback';
const FOLLOW_UP_DAYS = 7;

export const FeedbackSystem = (() => {

  function load()       { return JSON.parse(localStorage.getItem(FEEDBACK_KEY) || '[]'); }
  function save(data)   { localStorage.setItem(FEEDBACK_KEY, JSON.stringify(data)); }

  /**
   * Register a new scan for follow-up.
   * @param {string} scanId   – timestamp ID from history entry
   * @param {string} disease  – disease name
   * @param {string} crop     – crop name
   * @param {string} stage    – severity stage
   * @param {string} treatment – treatment applied (organic/chemical/affordable)
   */
  function registerFollowUp(scanId, disease, crop, stage, treatment) {
    const data = load();
    if (data.find(f => f.scanId === scanId)) return; // already registered
    data.push({
      scanId,
      disease,
      crop,
      stage,
      treatment,
      registeredAt: Date.now(),
      followUpAt:   Date.now() + FOLLOW_UP_DAYS * 24 * 60 * 60 * 1000,
      outcome:      null,   // null | 'worked' | 'partial' | 'failed'
      notes:        '',
    });
    save(data);
  }

  /**
   * Record an outcome.
   * @param {string} scanId
   * @param {'worked'|'partial'|'failed'} outcome
   * @param {string} notes
   */
  function recordOutcome(scanId, outcome, notes = '') {
    const data = load();
    const entry = data.find(f => f.scanId === scanId);
    if (entry) {
      entry.outcome   = outcome;
      entry.notes     = notes;
      entry.resolvedAt = Date.now();
      save(data);
    }
  }

  /** Returns entries that are due for follow-up and still pending. */
  function getDueFollowUps() {
    const now  = Date.now();
    // For demo: anything > 1 minute old counts as "due" (real: 7 days)
    const DEMO_THRESHOLD = 60 * 1000;
    return load().filter(f => f.outcome === null && (now - f.registeredAt) > DEMO_THRESHOLD);
  }

  /** Returns all completed outcomes. */
  function getOutcomes() { return load().filter(f => f.outcome !== null); }

  /** Stats for a specific disease across all resolved cases. */
  function getDiseaseOutcomeStats(disease) {
    const resolved = getOutcomes().filter(f => f.disease === disease);
    if (resolved.length === 0) return null;
    return {
      total:   resolved.length,
      worked:  resolved.filter(f => f.outcome === 'worked').length,
      partial: resolved.filter(f => f.outcome === 'partial').length,
      failed:  resolved.filter(f => f.outcome === 'failed').length,
    };
  }

  /** Clear all feedback data */
  function clear() { localStorage.removeItem(FEEDBACK_KEY); }

  return { registerFollowUp, recordOutcome, getDueFollowUps, getOutcomes, getDiseaseOutcomeStats, clear };
})();


/**
 * FollowUpScheduler – checks for due follow-ups on app start and
 * renders a floating follow-up card if any are pending.
 */
export function initFollowUpChecker(onFollowUpDue) {
  const due = FeedbackSystem.getDueFollowUps();
  if (due.length > 0) {
    // Notify after a short delay so the main UI loads first
    setTimeout(() => onFollowUpDue(due[0]), 1500);
  }
}
