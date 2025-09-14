import * as api from './net.js';

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

function setSliderValue(key, val){
  const slider = document.getElementById(`weight-${key}`);
  if (!slider) return;
  const v = Math.max(0, Math.min(100, parseInt(val,10) || 0));
  slider.value = v;
  const item = slider.closest('li.weight-item');
  const pill = item ? item.querySelector('.wi-pill') : null;
  if (pill) pill.textContent = `peso: ${v}/100`;
  const factor = factors.find(f => f.key === key);
  if (factor) factor.weight = v;
  cacheState.weights[key] = v;
}

function setToggleEnabled(key, enabled){
  const item = document.querySelector(`li.weight-item[data-key="${key}"]`);
  const toggle = item ? item.querySelector('.wt-enabled') : null;
  if (toggle){
    toggle.checked = enabled;
  }
  const factor = factors.find(f => f.key === key);
  if (factor) factor.enabled = enabled;
  cacheState.enabled[key] = enabled;
}

function reorderWeightsUI(order){
  const list = document.getElementById('weightsList');
  if (!list) return;
  const items = Array.from(list.children);
  const byKey = Object.fromEntries(items.map(el => [el.dataset.key, el]));
  const ordered = [];
  order.forEach(k => {
    const el = byKey[k];
    if (el) ordered.push(el);
  });
  list.innerHTML = '';
  ordered.forEach(el => list.appendChild(el));
  factors.sort((a,b) => order.indexOf(a.key) - order.indexOf(b.key));
  cacheState.order = factors.map(f => f.key);
  factors.forEach((f, idx) => {
    const el = byKey[f.key];
    const rank = el ? el.querySelector('.wi-rank') : null;
    if (rank) rank.textContent = `#${idx+1}`;
  });
}

function renderEffectiveBadges(eff){
  window.winnerWeightsEffective = eff;
  const effEl = document.getElementById('effectiveWeights');
  if (effEl) effEl.textContent = JSON.stringify(eff);
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
}

window.openConfigModal = openConfigModal;
window.loadWeights = hydrateSettingsModal;
window.resetWeights = resetWeights;
window.markDirty = markDirty;
window.metricKeys = metricKeys;
