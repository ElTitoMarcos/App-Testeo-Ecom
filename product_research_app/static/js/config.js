import * as api from './net.js';
import { LoadingHelpers } from './loading.js';

const SettingsCache = (() => {
  let cache = null;
  let inflight = null;

  const norm = (cfg) => {
    const weights = cfg?.weights || {};
    const order = (Array.isArray(cfg?.weights_order) && cfg.weights_order.length)
      ? cfg.weights_order.slice()
      : Object.keys(weights);
    const enabled = {};
    for (const k of order) enabled[k] = cfg?.weights_enabled?.[k] ?? true;
    return { order, weights, enabled };
  };

  const get = async () => {
    if (cache) return cache;
    if (inflight) return inflight;
    inflight = fetch('/config', { cache: 'no-store' })
      .then(r => r.json())
      .then(cfg => (cache = norm(cfg)))
      .finally(() => { inflight = null; });
    return inflight;
  };

  const set = (next) => { cache = norm(next); };

  return { get, set };
})();

const WEIGHT_FIELDS = [
  { key: 'price',        label: 'Price' },
  { key: 'rating',       label: 'Rating' },
  { key: 'units_sold',   label: 'Units Sold' },
  { key: 'revenue',      label: 'Revenue' },
  { key: 'desire',       label: 'Desire' },
  { key: 'competition',  label: 'Competition' },
  { key: 'oldness',      label: 'Oldness (antigüedad)' },
  { key: 'awareness',    label: 'Awareness' }
];
const WEIGHT_KEYS = WEIGHT_FIELDS.map(f => f.key);
const EXTREMES = {
  price:       { left: 'Más barato',       right: 'Más caro' },
  rating:      { left: 'Peor rating',      right: 'Mejor rating' },
  units_sold:  { left: 'Menos ventas',     right: 'Más ventas' },
  revenue:     { left: 'Menores ingresos', right: 'Mayores ingresos' },
  desire:      { left: 'Menor deseo',      right: 'Mayor deseo' },
  competition: { left: 'Menos competencia', right: 'Más competencia' },
  oldness:     { left: 'Más reciente',     right: 'Más antiguo' },
  awareness:   { left: 'Unaware',          right: 'Most aware' }
};
const ALIASES = { unitsSold: 'units_sold', orders: 'units_sold' };
function normalizeKey(k){ return ALIASES[k] || k; }
const metricDefs = WEIGHT_FIELDS;
const metricKeys = WEIGHT_KEYS;
let factors = [];
let userConfig = {};
let cacheState = { order: [], weights: {}, enabled: {} };

document.addEventListener('DOMContentLoaded', () => {
  SettingsCache.get().catch(() => {});
});

function defaultFactors(){
  return WEIGHT_FIELDS.map(f => ({ ...f, weight:50, enabled:true }));
}
let saveTimer=null;
let isInitialRender = true;
function markDirty(){
  clearTimeout(saveTimer);
  saveTimer=setTimeout(saveSettings,700);
}

function bindToggle(itemEl, field, state) {
  const toggle = itemEl.querySelector('.wt-enabled');
  if (!toggle) return;
  const setDisabled = (off) => {
    itemEl.classList.toggle('disabled', off);
    itemEl.querySelectorAll('input[type="range"], input[type="number"]').forEach(el => el.disabled = off);
  };
  toggle.checked = !!state.enabled[field];
  setDisabled(!toggle.checked);
  toggle.addEventListener('change', (e) => {
    const on = !!e.target.checked;
    state.enabled[field] = on;
    const factor = factors.find(f => f.key === field);
    if (factor) factor.enabled = on;
    setDisabled(!on);
    if(!isInitialRender) markDirty();
  });
}

function enhanceRangeWithFloat(rangeEl, itemEl){
  const THUMB = 16; // debe coincidir con el CSS del thumb
  const floatEl = itemEl.querySelector('.wi-float');
  if (!floatEl) return;

  const setPercent = () => {
    const min = +rangeEl.min || 0;
    const max = +rangeEl.max || 100;
    const val = +rangeEl.value || 0;
    const p = ((val - min) * 100) / (max - min);
    rangeEl.style.setProperty('--p', `${p}%`); // colorea el track
    return p;
  };

  const placeFloat = () => {
    const rect = rangeEl.getBoundingClientRect();
    const p = parseFloat(getComputedStyle(rangeEl).getPropertyValue('--p'));
    const usable = rect.width - THUMB;
    const x = (p / 100) * usable + THUMB / 2;
    floatEl.style.left = `${x}px`;
    floatEl.textContent = `peso: ${Math.round(rangeEl.value)}/100`;
  };

  const showFloat = () => { floatEl.classList.add('show'); };
  const hideFloatSoon = () => {
    clearTimeout(hideFloatSoon._t);
    hideFloatSoon._t = setTimeout(() => floatEl.classList.remove('show'), 700);
  };

  const updateAll = () => { setPercent(); placeFloat(); };

  rangeEl.addEventListener('input', () => { updateAll(); showFloat(); });
  rangeEl.addEventListener('pointerdown', () => { updateAll(); showFloat(); });
  ['pointerup','blur','mouseleave'].forEach(ev =>
    rangeEl.addEventListener(ev, hideFloatSoon)
  );

  updateAll();

  const toggle = itemEl.querySelector('.wt-enabled');
  if (toggle){
    toggle.addEventListener('change', () => {
      if (!toggle.checked) floatEl.classList.remove('show');
      else { updateAll(); showFloat(); hideFloatSoon(); }
    });
  }
}

function renderWeightsUI(state){
  const list = document.getElementById('weightsList');
  if(!list) return;
  if(state){
    cacheState = state;
    const fieldList = (typeof WEIGHT_FIELDS !== 'undefined' && Array.isArray(WEIGHT_FIELDS)) ? WEIGHT_FIELDS : [];
    const byKey = Object.fromEntries(fieldList.map(f => [f.key, f]));
    factors = cacheState.order
      .filter(k => byKey[k])
      .map(k => ({
        ...byKey[k],
        weight: (cacheState.weights[k] !== undefined && !isNaN(cacheState.weights[k])) ? Math.round(Number(cacheState.weights[k])) : 50,
        enabled: cacheState.enabled[k] !== undefined ? !!cacheState.enabled[k] : true
      }));
  } else {
    cacheState = {
      order: factors.map(f => f.key),
      weights: Object.fromEntries(factors.map(f => [f.key, f.weight])),
      enabled: Object.fromEntries(factors.map(f => [f.key, f.enabled !== false]))
    };
  }
  list.innerHTML = '';
  factors.forEach((f,idx) => {
    const priority = idx + 1;
    const li = document.createElement('li');
    li.className = 'weight-item';
    li.dataset.key = f.key;

    if (f.key === 'awareness') {
      li.innerHTML = `
        <div class="wi-head"><span class="wi-rank">#${priority}</span><span class="wi-title">Awareness</span><span class="wi-handle" aria-hidden="true">≡</span></div>
        <div class="wi-slider">
          <div class="segmented-range">
            <input id="awarenessSlider" class="weight-slider seg-awareness" type="range" min="0" max="100" step="1" />
            <div class="ticks" aria-hidden="true">
              <i style="left:20%"></i><i style="left:40%"></i><i style="left:60%"></i><i style="left:80%"></i>
            </div>
          </div>
          <div class="wi-float" aria-hidden="true"></div>
          <div class="awareness-labels">
            <span>Unaware</span>
            <span>Problem aware</span>
            <span>Solution aware</span>
            <span>Product aware</span>
            <span>Most aware</span>
          </div>
        </div>
        <div class="wi-meta">
          <span class="wi-min">${EXTREMES[f.key].left}</span>
          <div class="wi-center">
            <span class="wi-pill">peso: ${f.weight}/100</span>
            <label class="wi-toggle"><input type="checkbox" class="wt-enabled" aria-label="Activar métrica" /></label>
          </div>
          <span class="wi-max">${EXTREMES[f.key].right}</span>
        </div>`;
      const slider = li.querySelector('#awarenessSlider');
      const pill = li.querySelector('.wi-pill');
      const segs = Array.from(li.querySelectorAll('.awareness-labels span'));
      function updateAw(val){
        const v = Math.max(0, Math.min(100, parseInt(val,10) || 0));
        f.weight = v;
        cacheState.weights[f.key] = v;
        slider.value = v;
        pill.textContent = `peso: ${v}/100`;
        const idx = Math.min(4, Math.floor(v/20));
        segs.forEach((el,i)=>el.classList.toggle('active', i===idx));
      }
      updateAw(f.weight);
      slider.addEventListener('input', e => {
        updateAw(e.target.value);
        if(!isInitialRender) markDirty();
      });
      bindToggle(li, f.key, cacheState);
    } else {
      li.innerHTML = `
        <div class="wi-head"><span class="wi-rank">#${priority}</span><span class="wi-title">${f.label}</span><span class="wi-handle" aria-hidden="true">≡</span></div>
        <div class="wi-slider"><input id="weight-${f.key}" class="weight-range" type="range" min="0" max="100" step="1" value="${f.weight}"><div class="wi-float" aria-hidden="true"></div></div>
        <div class="wi-meta">
          <span class="wi-min">${EXTREMES[f.key].left}</span>
          <div class="wi-center">
            <span class="wi-pill">peso: ${f.weight}/100</span>
            <label class="wi-toggle"><input type="checkbox" class="wt-enabled" aria-label="Activar métrica" /></label>
          </div>
          <span class="wi-max">${EXTREMES[f.key].right}</span>
        </div>`;
      const range = li.querySelector('.weight-range');
      const pill = li.querySelector('.wi-pill');
      range.addEventListener('input', e => {
        const v = Math.max(0, Math.min(100, parseInt(e.target.value,10) || 0));
        f.weight = v;
        cacheState.weights[f.key] = v;
        range.value = v;
        pill.textContent = `peso: ${f.weight}/100`;
        if(!isInitialRender) markDirty();
      });
      bindToggle(li, f.key, cacheState);
    }
    list.appendChild(li);
    enhanceRangeWithFloat(li.querySelector('input[type="range"]'), li);
  });
  Sortable.create(list,{ handle:'.wi-handle', animation:150, onEnd:()=>{
    const orderKeys = Array.from(list.children).map(li=>li.dataset.key);
    factors.sort((a,b)=>orderKeys.indexOf(a.key)-orderKeys.indexOf(b.key));
    cacheState.order = orderKeys;
    renderWeightsUI();
    if(!isInitialRender) markDirty();
  }});
  const root = document.getElementById('weightsCard');
  if (root && !root.classList.contains('weights-section')) root.classList.add('weights-section');
  document.querySelector('.weights-section')?.classList.add('compact');
  isInitialRender = false;
  window.factors = factors;
}

function resetWeights(){
  const state = {
    order: WEIGHT_FIELDS.map(f => f.key),
    weights: Object.fromEntries(WEIGHT_FIELDS.map(f => [f.key, 50])),
    enabled: Object.fromEntries(WEIGHT_FIELDS.map(f => [f.key, true]))
  };
  renderWeightsUI(state);
  markDirty();
}

async function saveSettings(){
  const payload = {
    weights: Object.fromEntries(
      factors.map(f => [f.key, Math.max(0, Math.min(100, Math.round(Number(f.weight))))])
    ),
    order: factors.map(f => f.key),
    weights_enabled: Object.fromEntries(
      factors.map(f => [f.key, f.enabled !== false])
    ),
    weights_order: factors.map(f => f.key)
  };
  try{
    const res = await fetch('/api/config/winner-weights', {
      method:'PATCH',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json().catch(()=>null);
    if (data) SettingsCache.set(data);
    else SettingsCache.set({ weights: payload.weights, weights_order: payload.order, weights_enabled: payload.weights_enabled });
    if (typeof reloadProductsLight === 'function') reloadProductsLight();
    else if (typeof reloadProducts === 'function') reloadProducts();
  }catch(err){
    if (typeof toast !== 'undefined' && toast.error) toast.error('No se pudo guardar la configuración');
  }
}

const AWARE_MAP = {
  'unaware': 0,
  'problem aware': 0.25,
  'solution aware': 0.5,
  'product aware': 0.75,
  'most aware': 1
};
function awarenessValue(p){
  const s = (p.awareness_level || '').toString().trim().toLowerCase();
  return AWARE_MAP[s] ?? 0.5;
}

function stratifiedSample(list, n){
  const byCat = {};
  list.forEach(p=>{ const c = p.category || 'N/A'; (byCat[c] = byCat[c] || []).push(p); });
  const total = list.length;
  const sample = [];
  for(const c in byCat){
    const arr = byCat[c].slice().sort((a,b)=>{
      const ra = Number(a.revenue || (a.extras && a.extras['Revenue($)']) || 0);
      const rb = Number(b.revenue || (b.extras && b.extras['Revenue($)']) || 0);
      const ua = Number(a.units_sold || (a.extras && a.extras['Item Sold']) || 0);
      const ub = Number(b.units_sold || (b.extras && b.extras['Item Sold']) || 0);
      const sa = ra || ua;
      const sb = rb || ub;
      return sb - sa;
    });
    const k = Math.max(1, Math.round(n * arr.length / total));
    sample.push(...arr.slice(0,k));
  }
  return sample.slice(0,n);
}

async function adjustWeightsAI(ev){
  const btn = ev?.currentTarget || document.getElementById('btnAiWeights');
  const modal = btn?.closest('.modal') || document.querySelector('.config-modal.modal');
  const host = modal?.querySelector('.modal-progress-slot') || modal || document.querySelector('#inline-progress');
  const tracker = LoadingHelpers.start('Ajustando pesos con IA', { host });
  const num = v => { const n = Number(v); return Number.isFinite(n) ? n : 0; };
  const stratifiedSampleBy = (arr, key, n) => {
    if (!Array.isArray(arr) || arr.length <= n) return (arr || []).slice();
    const sorted = [...arr].sort((a,b) => num(b[key]) - num(a[key]));
    const out = [];
    for (let i=0; i<n; i++){
      const idx = Math.floor(i * (sorted.length - 1) / Math.max(1,(n - 1)));
      out.push(sorted[idx]);
    }
    return out;
  };

  try{
    const products = Array.isArray(window.allProducts) ? window.allProducts : [];
    if (!products.length){
      tracker.step(1);
      if (typeof toast !== 'undefined') toast.info('No hay productos cargados');
      return;
    }
    tracker.step(0.1);

    // Construir dataset con las 8 features del Winner Score
    const rows = products.map(p => {
      const ratingRaw = p.rating ?? (p.extras && p.extras.rating);
      const unitsRaw  = p.units_sold ?? (p.extras && (p.extras['Item Sold'] || p.extras['Orders']));
      const revRaw    = p.revenue ?? (p.extras && (p.extras['Revenue($)'] || p.extras['Revenue']));
      return {
        price:       num(p.price),
        rating:      num(ratingRaw),
        units_sold:  num(unitsRaw),
        revenue:     num(revRaw),
        desire:      num(p.desire_magnitude),
        competition: num(p.competition_level),
        oldness:     num(typeof computeOldnessDays === 'function' ? computeOldnessDays(p) : 0),
        awareness:   num(typeof awarenessValue === 'function' ? awarenessValue(p) : 0),
      };
    });

    // Target: revenue si >=50% lo tiene; si no, units_sold
    const revCount = rows.filter(r => r.revenue > 0).length;
    const targetName = (revCount >= Math.ceil(rows.length * 0.5)) ? 'revenue' : 'units_sold';
    tracker.step(0.22);

    // Enviar el mayor número posible en una sola llamada (controlado por presupuesto)
    const cfg = (window.userConfig && window.userConfig.aiCost) ? window.userConfig.aiCost : { costCapUSD: 0.25, estTokensPerItemIn: 300, estTokensPerItemOut: 80 };
    const estTokPerItem = (num(cfg.estTokensPerItemIn) + num(cfg.estTokensPerItemOut)) || 380;
    const pricePerK = 0.002; // estimación conservadora
    const maxByBudget = Math.max(30, Math.floor((cfg.costCapUSD || 0.25) / pricePerK * 1000 / estTokPerItem));
    const HARD_CAP = 500;
    const MAX = Math.min(HARD_CAP, Math.max(60, maxByBudget));

    let data_sample = rows.map(r => ({ ...r, target: r[targetName] }));
    if (data_sample.length > MAX) data_sample = stratifiedSampleBy(data_sample, targetName, MAX);

    const features = (typeof metricKeys !== 'undefined' && metricKeys.length) ? metricKeys : [
      'price','rating','units_sold','revenue','desire','competition','oldness','awareness'
    ];
    const payload = { features, target: targetName, data_sample };
    tracker.step(0.35);

    // 1 intento: GPT; fallback estadístico si falla
    let res = await fetch('/scoring/v2/auto-weights-gpt', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload),
      __hostEl: host,
      __skipLoadingHook: true
    });
    if (!res.ok){
      res = await fetch('/scoring/v2/auto-weights-stat', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(payload),
        __hostEl: host,
        __skipLoadingHook: true
      });
    }
    if (!res.ok) throw new Error('Auto-weights request failed');

    const out = await res.json(); // { weights:{}, weights_order:[...], method?... }
    const intWeights = (out && out.weights) ? out.weights : {};
    const newOrder = (out && Array.isArray(out.weights_order) && out.weights_order.length)
      ? out.weights_order.slice()
      : Object.keys(intWeights).sort((a,b) => (intWeights[b]||0) - (intWeights[a]||0));
    tracker.step(0.55);

    // Persistir en backend y refrescar UI con lo guardado
    const state = await SettingsCache.get();
    const resSave = await fetch('/api/config/winner-weights', {
      method:'PATCH',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ weights: intWeights, weights_order: newOrder, weights_enabled: state.enabled }),
      __hostEl: host,
      __skipLoadingHook: true
    });
    if (!resSave.ok) throw new Error('Persist weights failed');
    const saved = await resSave.json();
    SettingsCache.set(saved);
    // renderWeightsUI reinstates slider helpers like enhanceRangeWithFloat
    const fresh = await SettingsCache.get();
    if (typeof renderWeightsUI === 'function') renderWeightsUI(fresh);
    tracker.step(0.85);

    if (typeof toast !== 'undefined' && toast.success){
      const method = (out && out.method) ? out.method : 'gpt';
      toast.success(`Pesos ajustados por IA (${method}) con ${data_sample.length} muestras`);
    }
    tracker.step(1);
  }catch(err){
    tracker.step(1);
    if (typeof toast !== 'undefined' && toast.error){
      toast.error('No se pudo ajustar por IA. Revisa tu API Key o inténtalo más tarde.');
    }
  } finally {
    tracker.done();
  }
}


function showSettingsModalShell(){
  const list = document.getElementById('weightsList');
  if (list) list.innerHTML = '';
}

function revealSettingsModalContent(){ /* no-op */ }

async function hydrateSettingsModal(){
  try{
    isInitialRender = true;
    const state = await SettingsCache.get();
    renderWeightsUI(state);
    console.debug('hydrateSettingsModal -> weights/order aplicados:', state);
  }catch(err){
    /* silencioso */
  }
}

async function openConfigModal(){
  showSettingsModalShell();
  await hydrateSettingsModal();
  document.querySelector('#configModal')?.classList.add('ready');
  document.querySelector('#settings-modal')?.classList.add('ready');
  revealSettingsModalContent();
  const resetBtn = document.getElementById('btnReset');
  if (resetBtn) resetBtn.onclick = resetWeights;
  const aiBtn = document.getElementById('btnAiWeights');
  if (aiBtn) aiBtn.onclick = adjustWeightsAI;
}

window.openConfigModal = openConfigModal;
window.loadWeights = hydrateSettingsModal;
window.resetWeights = resetWeights;
window.adjustWeightsAI = adjustWeightsAI;
window.markDirty = markDirty;
window.metricKeys = metricKeys;
