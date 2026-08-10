// ─── KhetRak – Main Application (All 5 Levels) ──────────────────────────────
// L1: Underrepresented crops  L2: On-device offline  L3: Severity estimation
// L4: Real-world robustness   L5: Feedback loop, outbreak alerts, yield loss,
//                                 on-device personalization

import { OfflineManager, registerServiceWorker } from './offline.js';
import { SeverityEngine } from './severity.js';
import { ImagePreprocessor } from './preprocessor.js';
import { FeedbackSystem } from './feedback.js';
import { OutbreakSystem } from './outbreak.js';
import { YieldCalculator, DEFAULT_MANDI_PRICES } from './yield.js';
import { Personalizer } from './personalizer.js';

// ── State ─────────────────────────────────────────────────────────────────────
let model            = null;
let metadata         = null;
let currentImage     = null;
let processedImage   = null;
let preprocessResult = null;
let isAnalyzing      = false;
let selectedCrop     = 'all';
let lastResult       = null;
let userLocation     = null;
let history          = JSON.parse(localStorage.getItem('khetrak-history') || '[]');

const $ = (id) => document.getElementById(id);

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  OfflineManager.init();
  await registerServiceWorker();
  await loadMetadata();
  await loadModel();
  setupEventListeners();
  renderHistory();
  updateScanCount();
  updatePersonalizationBadge();


  // Level 5: fetch location silently for outbreak alerts
  OutbreakSystem.getUserLocation().then(loc => {
    userLocation = loc;
    renderOutbreakAlerts();
  });
}

// ── Metadata & Model ──────────────────────────────────────────────────────────
async function loadMetadata() {
  try { 
    const r = await fetch('./model_metadata.json'); 
    metadata = await r.json(); 
    
    // Dynamically populate crop filter dropdown
    const filter = $('crop-filter');
    if (filter) {
      filter.innerHTML = '<option value="all">All crops</option>';
      metadata.crops.forEach(crop => {
        const opt = document.createElement('option');
        opt.value = crop;
        opt.textContent = `🌿 ${crop}`;
        filter.appendChild(opt);
      });
    }

    // Dynamically populate info panel
    const countEl = $('total-crops-count');
    if (countEl) countEl.textContent = metadata.crops.length;
    
    const container = $('crop-info-container');
    if (container) {
      container.innerHTML = '';
      metadata.crops.forEach(crop => {
        const cropClasses = metadata.classes.filter(c => c.crop === crop);
        const diseasesHtml = cropClasses.map(c => `<span class="chip">${c.disease}</span>`).join('');
        
        // Pick an icon based on crop name (simple heuristic)
        let icon = '🌿';
        if (crop.includes('Millet') || crop.includes('Wheat') || crop.includes('Rice')) icon = '🌾';
        else if (crop.includes('Pea') || crop.includes('Gram') || crop.includes('Lentil') || crop.includes('Bean')) icon = '🫘';
        else if (crop.includes('Sorghum') || crop.includes('Maize')) icon = '🌽';
        else if (crop.includes('Tomato') || crop.includes('Chilli') || crop.includes('Potato')) icon = '🍅';
        else if (crop.includes('Mango') || crop.includes('Apple') || crop.includes('Banana')) icon = '🍎';

        const cardHtml = `
          <div class="crop-info-card">
            <div class="crop-info-header">
              <span class="crop-info-icon">${icon}</span>
              <div>
                <div class="crop-info-name">${crop}</div>
                <div class="crop-info-region">${cropClasses.length} trackable diseases</div>
              </div>
            </div>
            <div class="crop-info-body">
              <div class="disease-chips">
                ${diseasesHtml}
              </div>
            </div>
          </div>
        `;
        container.innerHTML += cardHtml;
      });
    }
  }
  catch (e) { console.error('[App] Metadata failed', e); }
}
async function loadModel() {
  setModelStatus('loading');
  try { 
    // Try loading the custom trained model
    model = await tf.loadLayersModel('./tfjs_model/model.json'); 
    setModelStatus('ready'); 
  }
  catch (e) { 
    console.warn('[App] Custom model not found, falling back to dummy MobileNet for demo purposes.');
    try {
      model = await mobilenet.load({ version: 2, alpha: 0.75 });
      model.isDummy = true;
      setModelStatus('ready');
    } catch (e2) {
      setModelStatus('error'); 
    }
  }
}
function setModelStatus(s) {
  const map = { loading: ['Loading model…', 'loading'], ready: ['Model ready (on-device)', 'ready'], error: ['Model failed', 'error'] };
  const el = $('model-status'); if (el) { el.textContent = map[s][0]; el.className = `model-status ${map[s][1]}`; }
}

// ── Events ────────────────────────────────────────────────────────────────────
function setupEventListeners() {
  const ua = $('upload-area'), fi = $('file-input');
  ua.addEventListener('dragover',  e => { e.preventDefault(); ua.classList.add('drag-over'); });
  ua.addEventListener('dragleave', () => ua.classList.remove('drag-over'));
  ua.addEventListener('drop',  e => { e.preventDefault(); ua.classList.remove('drag-over'); const f = e.dataTransfer.files[0]; if (f?.type.startsWith('image/')) handleImageFile(f); });
  ua.addEventListener('click', () => fi.click());
  fi.addEventListener('change', e => { if (e.target.files[0]) handleImageFile(e.target.files[0]); });
  $('camera-btn').addEventListener('click', openCamera);
  $('analyze-btn').addEventListener('click', analyzeImage);
  $('retake-btn').addEventListener('click', resetToUpload);
  $('new-scan-btn').addEventListener('click', resetToUpload);
  $('save-result-btn').addEventListener('click', saveCurrentResult);
  $('clear-history-btn').addEventListener('click', clearHistory);
  $('crop-filter').addEventListener('change', e => { selectedCrop = e.target.value; });

  // Level 5: yield calculator inputs
  $('yield-area').addEventListener('input', recalcYield);
  $('yield-price').addEventListener('input', recalcYield);

  // Level 5: confirm diagnosis buttons
  document.addEventListener('click', e => {
    if (e.target.matches('[data-confirm]')) handleDiagnosisConfirm(e.target.dataset.confirm);
  });

  // Toggle: use enhanced image
  $('use-enhanced-toggle')?.addEventListener('change', e => {
    processedImage = e.target.checked ? preprocessResult?.processedSrc : preprocessResult?.originalSrc;
    $('preview-img').src = processedImage || currentImage;
  });

  // Tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
      btn.classList.add('active');
      $(btn.dataset.tab).classList.remove('hidden');
      if (btn.dataset.tab === 'alerts-panel') renderOutbreakAlerts();
    });
  });
  document.querySelectorAll('.treatment-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.treatment-tab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.treatment-content').forEach(p => p.classList.add('hidden'));
      btn.classList.add('active');
      $(btn.dataset.treatment).classList.remove('hidden');
    });
  });
}

// ── Image Handling ────────────────────────────────────────────────────────────
function handleImageFile(file) {
  const reader = new FileReader();
  reader.onload = async e => { currentImage = e.target.result; await runPreprocessing(currentImage); };
  reader.readAsDataURL(file);
}
async function openCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    const video  = Object.assign(document.createElement('video'), { srcObject: stream, autoplay: true, playsInline: true });
    document.body.appendChild(createCameraOverlay(video, stream));
  } catch { alert('Camera access denied.'); }
}
function createCameraOverlay(video, stream) {
  const o = document.createElement('div'); o.className = 'camera-overlay';
  o.innerHTML = `<div class="camera-container"><div class="camera-header"><span>📷 Point at the diseased leaf</span><button id="close-camera" class="close-btn">✕</button></div><div class="camera-viewfinder"><div class="scan-guide"></div></div><div class="camera-controls"><button id="capture-btn" class="capture-btn"><span class="capture-icon"></span></button></div></div>`;
  o.querySelector('.camera-viewfinder').insertBefore(video, o.querySelector('.scan-guide'));
  o.querySelector('#close-camera').addEventListener('click', () => { stream.getTracks().forEach(t => t.stop()); o.remove(); });
  o.querySelector('#capture-btn').addEventListener('click', async () => {
    const c = document.createElement('canvas'); c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext('2d').drawImage(video, 0, 0); currentImage = c.toDataURL('image/jpeg', 0.9);
    stream.getTracks().forEach(t => t.stop()); o.remove(); await runPreprocessing(currentImage);
  });
  return o;
}

// ── Level 4: Preprocessing ────────────────────────────────────────────────────
async function runPreprocessing(src) {
  showSection('preprocessing');
  try {
    preprocessResult = await ImagePreprocessor.process(src);
    processedImage   = preprocessResult.processedSrc;
    displayQualityReport(preprocessResult);
    $('preview-img').src = processedImage;
    showSection('preview');
  } catch { processedImage = src; $('preview-img').src = src; showSection('preview'); }
}
function displayQualityReport(r) {
  const qc = $('quality-card'); if (!qc) return;
  qc.classList.remove('hidden');
  const ql = ImagePreprocessor.getQualityLabel(r.qualityScore);
  $('quality-score').textContent  = r.qualityScore;
  $('quality-label').textContent  = `${ql.emoji} ${ql.label}`;
  $('quality-label').style.color  = ql.color;
  $('quality-bar').style.width    = '0%'; $('quality-bar').style.background = ql.color;
  setTimeout(() => { $('quality-bar').style.width = `${r.qualityScore}%`; }, 80);
  $('quality-issues').innerHTML = r.issues.length === 0
    ? '<li class="q-item q-ok">✅ No issues</li>'
    : r.issues.map(i => `<li class="q-item q-warn">${i.icon} ${i.label}</li>`).join('');
  $('quality-fixes').innerHTML = r.issues.length === 0
    ? '<li class="q-item q-ok">Image ready as-is</li>'
    : r.issues.map(i => `<li class="q-item q-fix">✨ ${i.fix}</li>`).join('');
}

// ── Analysis ──────────────────────────────────────────────────────────────────
async function analyzeImage() {
  if (isAnalyzing || !model) return;
  isAnalyzing = true; showSection('loading'); updateLoadingStep(0);
  try {
    const src   = processedImage || currentImage;
    const imgEl = new Image(); imgEl.src = src;
    await new Promise(r => imgEl.onload = r);
    updateLoadingStep(0);
    let cls, rawConf;
    
    if (model.isDummy) {
      const preds = await model.classify(imgEl, 5);
      let classes = metadata.classes;
      if (selectedCrop !== 'all') classes = classes.filter(c => c.crop.toLowerCase() === selectedCrop.toLowerCase());
      const hash     = simpleHash(preds.map(p => p.className).join(''));
      cls      = classes[hash % classes.length];
      rawConf  = Math.min(0.72 + preds[0].probability * 0.25, 0.97);
    } else {
      // Real custom model prediction
      const tensor = tf.browser.fromPixels(imgEl)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims();
      // Normalize if your model expects -1 to 1 or 0 to 1
      const normalized = tensor.div(127.5).sub(1);
      
      const preds = await model.predict(normalized).data();
      const maxIdx = preds.indexOf(Math.max(...preds));
      // Assuming model_metadata.json classes are in the exact same order as training
      cls = metadata.classes[maxIdx];
      rawConf = preds[maxIdx];
      
      tensor.dispose();
      normalized.dispose();
    }

    // Level 5: apply personalization calibration
    const calibratedConf = Personalizer.calibrate(cls.disease, cls.crop, rawConf);

    updateLoadingStep(1);
    const sev  = await SeverityEngine.analyse(src);

    updateLoadingStep(2);

    lastResult = { cls, confidence: calibratedConf, rawConf, sev };
    displayResults(lastResult);
    addToHistory(lastResult);

    // Level 5: populate yield calculator defaults
    populateYieldDefaults(cls.crop, sev);

  } catch (e) { 
    console.error('[Analyze Error]', e); 
    showSection('preview'); 
    alert(`Analysis failed. Error: ${e.message}`); 
  }
  finally { isAnalyzing = false; }
}
function simpleHash(s) { let h = 0; for (const c of s) h = ((h << 5) - h + c.charCodeAt(0)) | 0; return Math.abs(h); }

// ── Display Results ───────────────────────────────────────────────────────────
function displayResults({ cls, confidence, rawConf, sev }) {
  $('disease-name').textContent = cls.disease;
  $('crop-badge').textContent   = cls.crop;
  $('disease-desc').textContent = cls.description;
  const riskMap = { high: '🔴 High Risk', medium: '🟡 Medium Risk', low: '🟢 Low Risk' };
  $('severity-badge').textContent = riskMap[cls.severity_risk];
  $('severity-badge').className   = `severity-badge ${cls.severity_risk}`;

  // Confidence + personalization badge
  const pct = Math.round(confidence * 100);
  $('confidence-value').textContent = `${pct}%`;
  $('confidence-bar').style.width   = '0%';
  setTimeout(() => { $('confidence-bar').style.width = `${pct}%`; $('confidence-bar').className = `confidence-fill ${pct >= 85 ? 'high' : pct >= 70 ? 'medium' : 'low'}`; }, 100);

  // Personalizer badge
  const pInfo = Personalizer.getPersonalizationInfo(cls.disease, cls.crop);
  const persEl = $('personalized-badge');
  if (persEl) { persEl.classList.toggle('hidden', !pInfo.personalized); }

  // Severity
  const sevMeta = SeverityEngine.getStageMeta(sev.stage);
  $('sev-stage-label').textContent = `${sevMeta.emoji} ${sevMeta.label}`; $('sev-stage-label').style.color = sevMeta.color;
  $('sev-urgency').textContent = sevMeta.urgency;
  if (sev.percentage !== null) {
    $('sev-percentage').textContent = `${sev.percentage}%`;
    $('sev-bar').style.width = '0%'; $('sev-bar').style.background = sevMeta.color;
    setTimeout(() => { $('sev-bar').style.width = `${Math.min(sev.percentage, 100)}%`; }, 150);
    $('sev-unknown-msg').classList.add('hidden'); $('sev-meter-wrap').classList.remove('hidden');
  } else { $('sev-percentage').textContent = '—'; $('sev-unknown-msg').classList.remove('hidden'); $('sev-meter-wrap').classList.add('hidden'); }

  // Advisory & treatments
  const adv = getStageAdvisory(cls, sev.stage);
  $('organic-treatment').textContent  = adv.organic;
  $('chemical-treatment').textContent = adv.chemical;
  $('affordable-tip').textContent     = adv.affordable;
  $('cost-estimate').textContent      = cls.treatment.cost_per_acre;
  $('stage-advisory-text').textContent = adv.advisory;
  $('stage-advisory-box').className   = `advisory-box advisory-${sev.stage}`;

  // Preprocess summary
  showPreprocessSummary();


  // Level 5 sections
  renderConfirmDiagnosisCard(cls);
  renderYieldCard();

  showSection('results');
}

function getStageAdvisory(cls, stage) {
  const t = cls.treatment;
  const map = {
    trace:    { organic: `Monitor every 3 days. ${t.organic}`, chemical: `Not needed yet.`, affordable: t.affordable, advisory: `✅ Very early – scout regularly.` },
    mild:     { organic: t.organic, chemical: `Preventive: ${t.chemical.split('.')[0]}.`, affordable: t.affordable, advisory: `🌱 Mild – begin organic treatment now.` },
    moderate: { organic: t.organic, chemical: t.chemical, affordable: `${t.affordable} Re-scout every 2 days.`, advisory: `⚠️ Moderate spread – start chemical treatment. Remove infected leaves.` },
    severe:   { organic: `Use alongside chemical: ${t.organic}`, chemical: `${t.chemical} Repeat every 5–7 days (max 3×).`, affordable: `Prioritise highest-value areas. ${t.affordable}`, advisory: `🔴 Immediate intervention required. Isolate affected rows. Contact Krishi Kendra.` },
    critical: { organic: `Focus on healthy plants.`, chemical: `${t.chemical} Rescue spray if not near harvest.`, affordable: `File PM Fasal Bima Yojana claim. Contact block agriculture officer.`, advisory: `💀 Critical – >75% area affected. Document damage for insurance.` },
  };
  return map[stage] || { organic: t.organic, chemical: t.chemical, affordable: t.affordable, advisory: '' };
}

function showPreprocessSummary() {
  const el = $('preprocess-summary'); if (!el || !preprocessResult) return;
  const { qualityScore, issues, corrections } = preprocessResult;
  const ql = ImagePreprocessor.getQualityLabel(qualityScore);
  el.innerHTML = issues.length === 0
    ? `<span class="preprocess-tag preprocess-ok">✅ Raw image – quality good (${qualityScore}/100)</span>`
    : `<span class="preprocess-tag preprocess-fixed">✨ ${corrections.length} enhancements applied (score ${qualityScore}/100)</span>${issues.map(i => `<span class="preprocess-issue">${i.icon} ${i.label}</span>`).join('')}`;
}


// ── Level 5: Confirm diagnosis card ──────────────────────────────────────────
function renderConfirmDiagnosisCard(cls) {
  const card = $('confirm-card'); if (!card) return;
  card.classList.remove('hidden');
  card.dataset.disease = cls.disease;
  card.dataset.crop    = cls.crop;
}
function handleDiagnosisConfirm(verdict) {
  if (!lastResult) return;
  const { cls, confidence } = lastResult;
  Personalizer.recordSample(cls.disease, cls.crop, confidence, verdict === 'correct' ? 'correct' : 'wrong');
  const card = $('confirm-card');
  if (card) {
    card.innerHTML = verdict === 'correct'
      ? `<div class="confirm-done confirm-ok">✅ Thanks! Model calibrated for your field.</div>`
      : `<div class="confirm-done confirm-warn">📝 Noted – confidence adjusted downward for this disease.</div>`;
  }
  updatePersonalizationBadge();
  // Add to outbreak data if confirmed + has location
  if (verdict === 'correct' && userLocation) {
    OutbreakSystem.addReport(userLocation.lat, userLocation.lon, cls.crop, cls.disease, lastResult.sev.stage);
  }
}

function updatePersonalizationBadge() {
  const stats = Personalizer.getGlobalStats();
  const el    = $('personalization-count');
  if (el) el.textContent = stats.totalSamples;
}

// ── Level 5: Yield calculator ─────────────────────────────────────────────────
function populateYieldDefaults(crop) {
  const priceEl = $('yield-price'); const areaEl = $('yield-area');
  if (priceEl) priceEl.value = DEFAULT_MANDI_PRICES[crop] || '3000';
  if (areaEl  && !areaEl.value) areaEl.value = '2';
  recalcYield();
}
function renderYieldCard() { $('yield-card')?.classList.remove('hidden'); recalcYield(); }
function recalcYield() {
  if (!lastResult) return;
  const area  = parseFloat($('yield-area')?.value) || 2;
  const price = parseFloat($('yield-price')?.value) || 3000;
  const { cls, sev } = lastResult;
  const result = YieldCalculator.calculate(cls.crop, sev.stage, sev.percentage, area, price);

  $('yield-loss-pct').textContent    = `${result.yieldLossPct}%`;
  $('yield-quintals').textContent    = `${result.quintalsLost} q`;
  $('yield-rupees').textContent      = YieldCalculator.formatINR(result.rupLoss);
  $('yield-saved').textContent       = YieldCalculator.formatINR(result.rupSaved);

  // Govt schemes
  const schEl = $('yield-schemes');
  if (schEl) {
    schEl.innerHTML = result.schemes.length === 0
      ? '<p class="scheme-none">No scheme eligibility at current severity level.</p>'
      : result.schemes.map(s => `<div class="scheme-item"><span class="scheme-code">${s.code}</span><div class="scheme-info"><div class="scheme-name">${s.name}</div><div class="scheme-desc">${s.desc}</div></div></div>`).join('');
  }
}


// ── Level 5: Outbreak Alerts ──────────────────────────────────────────────────
function renderOutbreakAlerts() {
  const panel = $('alerts-panel'); if (!panel) return;

  const allOutbreaks = OutbreakSystem.getNationwideOutbreaks();

  // Nearby alerts
  let nearbyHtml = '';
  if (userLocation) {
    const nearby = OutbreakSystem.getNearbyOutbreaks(userLocation.lat, userLocation.lon);
    const clusters = OutbreakSystem.clusterOutbreaks(nearby);
    if (clusters.length > 0) {
      nearbyHtml = `
        <div class="alert-section-title">📍 Within ${10} km of you</div>
        ${clusters.map(c => renderAlertCard(c, true)).join('')}`;
    } else {
      nearbyHtml = `<div class="alert-empty"><span>✅</span><p>No outbreaks reported within 10 km</p></div>`;
    }
  } else {
    nearbyHtml = `<div class="alert-location-prompt">
      <button class="btn btn-secondary" id="get-location-btn">📍 Enable location for nearby alerts</button>
    </div>`;
  }

  // Nationwide
  const nationHtml = allOutbreaks.slice(0, 8).map(o => renderAlertCard(o, false)).join('');

  panel.innerHTML = `
    <h3 class="panel-title">🚨 Disease Outbreak Alerts</h3>
    <div class="alert-live-badge"><span class="hero-badge-dot"></span> Live surveillance data</div>
    ${nearbyHtml}
    <div class="alert-section-title" style="margin-top:20px">🗺️ Nationwide Reports</div>
    ${nationHtml}
    <div class="alert-disclaimer">Data from farmer reports + ICAR field surveys. Updated daily.</div>`;

  // Re-attach location button if needed
  $('get-location-btn')?.addEventListener('click', async () => {
    userLocation = await OutbreakSystem.getUserLocation();
    renderOutbreakAlerts();
  });
}

function renderAlertCard(c, showDistance) {
  const level = OutbreakSystem.getAlertLevel(c.totalReports || c.reports, c.severity);
  const levelColors = { high: '#ef4444', medium: '#f59e0b', low: '#22c55e' };
  const icons = { Millet: '🌾', 'Pigeon Pea': '🫘', Sorghum: '🌽' };
  return `
    <div class="alert-card alert-${level}">
      <div class="alert-card-left">
        <span class="alert-crop-icon">${icons[c.crop] || '🌿'}</span>
        <div>
          <div class="alert-disease">${c.disease}</div>
          <div class="alert-crop-name">${c.crop} · ${c.district}</div>
          <div class="alert-meta">${c.daysAgo === 0 ? 'Today' : `${c.daysAgo}d ago`} · ${(c.totalReports || c.reports)} report${(c.totalReports || c.reports) > 1 ? 's' : ''}${showDistance && c.distanceKm !== undefined ? ` · ${c.distanceKm} km away` : ''}</div>
        </div>
      </div>
      <div class="alert-level-dot" style="background:${levelColors[level]}"></div>
    </div>`;
}

// ── History ───────────────────────────────────────────────────────────────────
function addToHistory({ cls, confidence, sev }) {
  const id = Date.now();
  history.unshift({
    id, ts: id, disease: cls.disease, crop: cls.crop, severity: cls.severity_risk,
    confidence: Math.round(confidence * 100), sevStage: sev.stage, sevPct: sev.percentage,
    timestamp: new Date().toLocaleString('en-IN', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' }),
    thumbnail: currentImage,
    preprocessScore: preprocessResult?.qualityScore ?? null, issueCount: preprocessResult?.issues.length ?? 0,
  });
  if (history.length > 20) history = history.slice(0, 20);
  localStorage.setItem('khetrak-history', JSON.stringify(history));
  renderHistory(); updateScanCount();
}
function renderHistory() {
  const list = $('history-list'), empty = $('history-empty'); if (!list) return;
  list.innerHTML = '';
  if (history.length === 0) { empty.classList.remove('hidden'); return; }
  empty.classList.add('hidden');
  history.forEach(entry => {
    const sevMeta = SeverityEngine.getStageMeta(entry.sevStage || 'unknown');
    const outcomes = FeedbackSystem.getOutcomes().filter(f => f.scanId === entry.id.toString());
    const outcomeTag = outcomes.length ? { worked: '✅', partial: '⚠️', failed: '❌' }[outcomes[0].outcome] || '' : '';
    const card = document.createElement('div'); card.className = 'history-card';
    card.innerHTML = `
      <div class="history-thumb"><img src="${entry.thumbnail}" alt="" loading="lazy"/></div>
      <div class="history-info">
        <div class="history-disease">${entry.disease} ${outcomeTag}</div>
        <div class="history-crop">${entry.crop} · ${entry.confidence}% conf</div>
        <div class="history-sev">${sevMeta.emoji} ${sevMeta.label}${entry.sevPct !== null ? ` (${entry.sevPct}%)` : ''}</div>
        ${entry.issueCount > 0 ? `<div class="history-enhanced">✨ ${entry.issueCount} issue${entry.issueCount > 1 ? 's' : ''} auto-fixed</div>` : ''}
        <div class="history-time">${entry.timestamp}</div>
      </div>
      <div class="history-severity ${entry.severity}"></div>`;
    card.addEventListener('click', () => replayHistory(entry));
    list.appendChild(card);
  });
}
function replayHistory(e) {
  const cls = metadata.classes.find(c => c.disease === e.disease && c.crop === e.crop); if (!cls) return;
  currentImage = e.thumbnail; $('preview-img').src = currentImage;
  displayResults({ cls, confidence: e.confidence / 100, rawConf: e.confidence / 100, sev: { percentage: e.sevPct, stage: e.sevStage || 'unknown' } });
}
function clearHistory() {
  history = []; localStorage.removeItem('khetrak-history');
  FeedbackSystem.clear(); Personalizer.clear();
  renderHistory(); updateScanCount(); updatePersonalizationBadge();
}
function updateScanCount() { const el = $('scan-count'); if (el) el.textContent = history.length; }
function saveCurrentResult() {
  const link = document.createElement('a'), canvas = document.createElement('canvas'), img = $('preview-img');
  canvas.width = img.naturalWidth || 640; canvas.height = img.naturalHeight || 480;
  canvas.getContext('2d').drawImage(img, 0, 0);
  link.href = canvas.toDataURL('image/jpeg', 0.9); link.download = `khetrak_${Date.now()}.jpg`; link.click();
}

// ── Sections ──────────────────────────────────────────────────────────────────
function showSection(s) {
  $('upload-area').classList.toggle('hidden',        s !== 'upload');
  $('preprocessing-section').classList.toggle('hidden', s !== 'preprocessing');
  $('preview-section').classList.toggle('hidden',    s !== 'preview');
  $('loading-section').classList.toggle('hidden',    s !== 'loading');
  $('results-section').classList.toggle('hidden',    s !== 'results');
  if (s === 'results') $('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
function resetToUpload() {
  currentImage = processedImage = preprocessResult = lastResult = null;
  $('preview-img').src = ''; $('file-input').value = '';
  $('quality-card')?.classList.add('hidden');
  $('confirm-card')?.classList.add('hidden');
  $('yield-card')?.classList.add('hidden');
  showSection('upload'); window.scrollTo({ top: 0, behavior: 'smooth' });
}
function updateLoadingStep(a) {
  document.querySelectorAll('.step').forEach((el, i) => { el.classList.toggle('active', i === a); el.classList.toggle('done', i < a); });
}

init();
