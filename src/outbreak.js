/**
 * outbreak.js – KhetRak Level 5: Geo-Clustered Disease Surveillance
 * ────────────────────────────────────────────────────────────────────────────
 * Aggregates disease reports by GPS location.
 * - Stores confirmed reports with coordinates
 * - Clusters reports within a 10 km radius
 * - Surfaces outbreak alerts to nearby farmers
 * - Seeded with realistic demo data across Indian agri-zones
 */

const OUTBREAK_KEY = 'khetrak-outbreaks';
const ALERT_RADIUS_KM = 10;

// ── Haversine distance ──────────────────────────────────────────────────────
function haversine(lat1, lon1, lat2, lon2) {
  const R   = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a   = Math.sin(dLat / 2) ** 2 +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ── Seeded demo outbreak data across India ──────────────────────────────────
const DEMO_OUTBREAKS = [
  // Rajasthan – Millet belt
  { lat: 26.912434, lon: 75.787271, crop: 'Millet',     disease: 'Downy Mildew',        reports: 12, severity: 'severe',   daysAgo: 3,  district: 'Jaipur' },
  { lat: 27.023000, lon: 74.219900, crop: 'Millet',     disease: 'Blast Disease',        reports: 7,  severity: 'moderate', daysAgo: 5,  district: 'Nagaur' },
  { lat: 26.450000, lon: 73.113000, crop: 'Millet',     disease: 'Ergot',                reports: 4,  severity: 'mild',     daysAgo: 8,  district: 'Jodhpur' },
  // Maharashtra – Pigeon Pea belt
  { lat: 19.997454, lon: 73.789803, crop: 'Pigeon Pea', disease: 'Fusarium Wilt',        reports: 18, severity: 'severe',   daysAgo: 2,  district: 'Nashik' },
  { lat: 18.520430, lon: 73.856743, crop: 'Pigeon Pea', disease: 'Sterility Mosaic',     reports: 9,  severity: 'moderate', daysAgo: 6,  district: 'Pune' },
  { lat: 17.688000, lon: 76.820000, crop: 'Pigeon Pea', disease: 'Phytophthora Blight',  reports: 5,  severity: 'moderate', daysAgo: 10, district: 'Latur' },
  // Karnataka – Sorghum belt
  { lat: 15.317277, lon: 75.713890, crop: 'Sorghum',    disease: 'Grain Mold',           reports: 14, severity: 'severe',   daysAgo: 1,  district: 'Dharwad' },
  { lat: 14.467000, lon: 75.920000, crop: 'Sorghum',    disease: 'Leaf Blight',          reports: 8,  severity: 'moderate', daysAgo: 4,  district: 'Chitradurga' },
  { lat: 16.203700, lon: 77.356600, crop: 'Sorghum',    disease: 'Covered Kernel Smut',  reports: 3,  severity: 'mild',     daysAgo: 12, district: 'Raichur' },
  // Gujarat – Millet
  { lat: 23.022505, lon: 72.571362, crop: 'Millet',     disease: 'Downy Mildew',         reports: 6,  severity: 'moderate', daysAgo: 7,  district: 'Ahmedabad' },
  // Andhra Pradesh – Sorghum
  { lat: 15.828126, lon: 78.037279, crop: 'Sorghum',    disease: 'Grain Mold',           reports: 11, severity: 'severe',   daysAgo: 2,  district: 'Kurnool' },
];

export const OutbreakSystem = (() => {

  function loadUserReports() {
    return JSON.parse(localStorage.getItem(OUTBREAK_KEY) || '[]');
  }

  function saveUserReports(data) {
    localStorage.setItem(OUTBREAK_KEY, JSON.stringify(data));
  }

  /**
   * Add a confirmed disease report at the user's location.
   */
  function addReport(lat, lon, crop, disease, severity) {
    const data = loadUserReports();
    data.push({ lat, lon, crop, disease, severity, ts: Date.now(), source: 'user' });
    saveUserReports(data);
  }

  /**
   * Request GPS and return { lat, lon } or null on failure.
   */
  async function getUserLocation() {
    return new Promise((resolve) => {
      if (!navigator.geolocation) { resolve(null); return; }
      navigator.geolocation.getCurrentPosition(
        (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
        ()     => resolve(null),
        { timeout: 6000 }
      );
    });
  }

  /**
   * Get all outbreaks within ALERT_RADIUS_KM of a given point.
   * Combines demo seed data + user reports.
   */
  function getNearbyOutbreaks(userLat, userLon) {
    const allReports = [
      ...DEMO_OUTBREAKS,
      ...loadUserReports().map(r => ({ ...r, reports: 1, daysAgo: Math.floor((Date.now() - r.ts) / 86400000), district: 'Your report' })),
    ];

    return allReports
      .map(r => ({ ...r, distanceKm: Math.round(haversine(userLat, userLon, r.lat, r.lon)) }))
      .filter(r => r.distanceKm <= ALERT_RADIUS_KM)
      .sort((a, b) => a.distanceKm - b.distanceKm);
  }

  /**
   * Get nationwide outbreak summary (for the Alerts tab).
   */
  function getNationwideOutbreaks() {
    return DEMO_OUTBREAKS
      .concat(loadUserReports().map(r => ({
        ...r, reports: 1, daysAgo: Math.floor((Date.now() - r.ts) / 86400000), district: 'Field report'
      })))
      .sort((a, b) => b.reports - a.reports);
  }

  /**
   * Cluster nearby reports – group by (crop + disease) within radius.
   */
  function clusterOutbreaks(outbreaks) {
    const clusters = {};
    outbreaks.forEach(r => {
      const key = `${r.crop}_${r.disease}`;
      if (!clusters[key]) clusters[key] = { ...r, locations: [r], totalReports: 0 };
      clusters[key].totalReports += r.reports;
      clusters[key].locations.push(r);
    });
    return Object.values(clusters).sort((a, b) => b.totalReports - a.totalReports);
  }

  /** Urgency level for UI colour coding */
  function getAlertLevel(reports, severity) {
    if (severity === 'severe'   || reports >= 10) return 'high';
    if (severity === 'moderate' || reports >= 5)  return 'medium';
    return 'low';
  }

  return { addReport, getUserLocation, getNearbyOutbreaks, getNationwideOutbreaks, clusterOutbreaks, getAlertLevel, DEMO_OUTBREAKS };
})();
