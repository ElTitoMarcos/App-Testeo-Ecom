import * as api from './net.js';

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
let userConfig = {};

// ------------------------------------------------------------
// ConfigStore (cache + normalizador)
// ------------------------------------------------------------
const ConfigStore = (() => {
  let cache = null;
  let inflight = null;

  function normalize(cfg){
    const weights = cfg.weights || {};
    const order = Array.isArray(cfg.weights_order) && cfg.weights_order.length
      ? cfg.weights_order
      : Object.keys(weights);
    const enabled = {};
    for(const k of order) enabled[k] = cfg.weights_enabled?.[k] ?? true;
    return { order, weights, enabled };
  }

  async function get(){
    if (cache) return cache;
    if (inflight) return inflight;
    inflight = fetch('/config', { cache:'no-store' })
      .then(r => r.json())
      .then(cfg => (cache = normalize(cfg)))
      .finally(()=>{ inflight=null; });
    return inflight;
  }

  function set(next){ cache = normalize(next); }
  return { get, set };
})();

// Prefetch
// ------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  ConfigStore.get().catch(()=>{});
});

// ------------------------------------------------------------
// Estado y helpers
// ------------------------------------------------------------
let state = { order: [], weights: {}, enabled: {} };
let saveTimer = null;

function markDirty(){
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => saveSettings(state), 700);
}

function defaultState(){
  return {
    order: metricKeys.slice(),
    weights: Object.fromEntries(metricKeys.map(k => 50)),
    enabled: Object.fromEntries(metricKeys.map(k => true))
  };
}

function renderWeightsUI(nextState){
  state = { ...nextState };
  const list = document.getElementById('weightsList');
  if(!list) return;
  list.innerHTML='';
  list.classList.add('weights-section');
  list.classList.add('compact');

  state.order.forEach((key, idx) => {
    const def = metricDefs.find(f => f.key === key) || { key, label:key };
    const weight = Number(state.weights[key] ?? 50);
    const enabled = state.enabled[key] !== false;

    const li = document.createElement('li');
    li.className = 'weight-item';
    li.dataset.key = key;
    if(!enabled) li.classList.add('disabled');

    li.innerHTML = `
      <div class="wi-row wi-head">
        <span class="wi-rank">#${idx+1}</span>
        <span class="wi-title">${def.label}</span>
        <span class="wi-handle" aria-hidden="true">≡</span>
      </div>
      <div class="wi-row wi-slider"></div>
      <div class="wi-row wi-meta">
        <span class="wi-min">${EXTREMES[key]?.left || ''}</span>
        <span class="wi-pill">peso: ${weight}/100</span>
        <label class="wi-toggle"><input type="checkbox" class="wt-enabled" ${enabled?'checked':''}></label>
        <span class="wi-max">${EXTREMES[key]?.right || ''}</span>
      </div>`;

    const sliderRow = li.querySelector('.wi-slider');
    let slider;
    if(key === 'awareness'){
      sliderRow.innerHTML = `
        <div class="segmented-range">
          <input class="weight-range seg-awareness" type="range" min="0" max="100" step="1" />
          <div class="ticks" aria-hidden="true">
            <i style="left:20%"></i><i style="left:40%"></i><i style="left:60%"></i><i style="left:80%"></i>
          </div>
        </div>
        <div class="awareness-labels">
          <span>Unaware</span>
          <span>Problem aware</span>
          <span>Solution aware</span>
          <span>Product aware</span>
          <span>Most aware</span>
        </div>`;
      slider = sliderRow.querySelector('input[type="range"]');
      const labels = Array.from(sliderRow.querySelectorAll('.awareness-labels span'));
      const updateAw = v => {
        const val = Math.max(0, Math.min(100, parseInt(v,10)||0));
        state.weights[key] = val;
        slider.value = String(val);
        li.querySelector('.wi-pill').textContent = `peso: ${val}/100`;
        const seg = Math.min(4, Math.floor(val/20));
        labels.forEach((el,i)=>el.classList.toggle('active', i===seg));
      };
      slider.addEventListener('input', e => { updateAw(e.target.value); markDirty(); });
      updateAw(weight);
    } else {
      slider = document.createElement('input');
      slider.type = 'range';
      slider.min = '0';
      slider.max = '100';
      slider.step = '1';
      slider.className = 'weight-range';
      slider.value = String(weight);
      sliderRow.appendChild(slider);
      slider.addEventListener('input', e => {
        const val = Math.max(0, Math.min(100, parseInt(e.target.value,10)||0));
        state.weights[key] = val;
        li.querySelector('.wi-pill').textContent = `peso: ${val}/100`;
        markDirty();
      });
    }

    const toggle = li.querySelector('.wt-enabled');
    toggle.checked = enabled;
    toggle.addEventListener('change', e => {
      const on = e.target.checked;
      state.enabled[key] = on;
      for(const el of li.querySelectorAll('input[type="range"], input[type="number"]')){
        el.disabled = !on;
      }
      li.classList.toggle('disabled', !on);
      markDirty();
    });
    if(!enabled){
      for(const el of li.querySelectorAll('input[type="range"], input[type="number"]')) el.disabled = true;
    }

    list.appendChild(li);
  });

  Sortable.create(list,{ handle:'.wi-handle', animation:150, onEnd:()=>{
    state.order = Array.from(list.children).map(li=>li.dataset.key);
    renderWeightsUI(state);
    markDirty();
  }});
}

function resetWeights(){
  const next = defaultState();
  state = next;
  renderWeightsUI(state);
  markDirty();
}

async function saveSettings(curr){
  const payload = {
    weights: Object.fromEntries(Object.entries(curr.weights).map(([k,v])=>[k,Math.max(0,Math.min(100,Math.round(Number(v))))] )),
    order: curr.order.slice(),
    weights_enabled: { ...curr.enabled },
    weights_order: curr.order.slice()
  };
  try{
    await fetch('/api/config/winner-weights', {
      method:'PATCH',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    ConfigStore.set(payload);
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

async function adjustWeightsAI(){
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
    if (!products.length){ if (typeof toast !== 'undefined') toast.info('No hay productos cargados'); return; }

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

    const revCount = rows.filter(r => r.revenue > 0).length;
    const targetName = (revCount >= Math.ceil(rows.length * 0.5)) ? 'revenue' : 'units_sold';

    const cfg = (window.userConfig && window.userConfig.aiCost) ? window.userConfig.aiCost : { costCapUSD: 0.25, estTokensPerItemIn: 300, estTokensPerItemOut: 80 };
    const estTokPerItem = (num(cfg.estTokensPerItemIn) + num(cfg.estTokensPerItemOut)) || 380;
    const pricePerK = 0.002;
    const maxByBudget = Math.max(30, Math.floor((cfg.costCapUSD || 0.25) / pricePerK * 1000 / estTokPerItem));
    const HARD_CAP = 500;
    const MAX = Math.min(HARD_CAP, Math.max(60, maxByBudget));

    let data_sample = rows.map(r => ({ ...r, target: r[targetName] }));
    if (data_sample.length > MAX) data_sample = stratifiedSampleBy(data_sample, targetName, MAX);

    const features = metricKeys.length ? metricKeys : [
      'price','rating','units_sold','revenue','desire','competition','oldness','awareness'
    ];
    const payload = { features, target: targetName, data_sample };

    let res = await fetch('/scoring/v2/auto-weights-gpt', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)
    });
    if (!res.ok){
      res = await fetch('/scoring/v2/auto-weights-stat', {
        method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)
      });
    }
    if (!res.ok) throw new Error('Auto-weights request failed');

    const out = await res.json();
    const returned = (out && out.weights) ? out.weights : {};

    const intWeights = {};
    for (const k of features){
      let v = num(returned[k]);
      if (v > 0 && v <= 1) v = v * 100;
      v = Math.max(0, Math.min(100, Math.round(v)));
      intWeights[k] = v;
    }
    const newOrder = [...features].sort((a,b) => (intWeights[b]||0) - (intWeights[a]||0));

    state = {
      order: newOrder,
      weights: { ...state.weights, ...intWeights },
      enabled: { ...state.enabled }
    };
    renderWeightsUI(state);

    await fetch('/api/config/winner-weights', {
      method:'PATCH', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ weights:intWeights, order:newOrder, weights_enabled: state.enabled, weights_order:newOrder })
    });
    ConfigStore.set({ weights:intWeights, weights_order:newOrder, weights_enabled: state.enabled });

    if (typeof toast !== 'undefined' && toast.success){
      const method = (out && out.method) ? out.method : 'auto';
      toast.success(`Pesos ajustados por IA (${method}) con ${data_sample.length} muestras`);
    }
  }catch(err){
    if (typeof toast !== 'undefined' && toast.error){
      toast.error('No se pudo ajustar por IA. Revisa tu API Key o inténtalo más tarde.');
    }
  }
}

async function openSettingsModal(){
  const cfg = document.getElementById('config');
  const wcard = document.getElementById('weightsCard');
  const footer = document.getElementById('weightsFooter');
  if(!cfg || !wcard || !footer || !window.modalManager) return;
  const btn = document.getElementById('configBtn');

  const modal = document.createElement('div');
  modal.id = 'configModal';
  modal.className = 'modal config-modal';
  modal.setAttribute('role','dialog');
  modal.setAttribute('aria-modal','true');
  modal.setAttribute('aria-labelledby','configModalTitle');
  modal.innerHTML = '<header class="modal-header"><h3 id="configModalTitle">Configuración</h3><button type="button" class="modal-close" aria-label="Cerrar">✕</button></header><div class="modal-body"><div class="skeleton"></div></div>';
  const body = modal.querySelector('.modal-body');

  const cfgParent = cfg.parentElement;
  const wParent = wcard.parentElement;
  const fParent = footer.parentElement;

  window.modalManager.open(modal, {
    trigger: btn,
    onClose: () => {
      cfgParent.appendChild(cfg);
      wParent.appendChild(wcard);
      fParent.appendChild(footer);
      if (saveTimer) { clearTimeout(saveTimer); saveSettings(state); }
    }
  });

  const data = await ConfigStore.get();
  body.innerHTML = '';
  body.appendChild(cfg);
  body.appendChild(wcard);
  body.appendChild(footer);
  renderWeightsUI(data);
  const resetBtn = document.getElementById('btnReset');
  if (resetBtn) resetBtn.onclick = resetWeights;
  const aiBtn = document.getElementById('btnAiWeights');
  if (aiBtn) aiBtn.onclick = adjustWeightsAI;
}

window.openSettingsModal = openSettingsModal;
window.resetWeights = resetWeights;
window.adjustWeightsAI = adjustWeightsAI;
window.markDirty = markDirty;
window.metricKeys = metricKeys;
window.renderWeightsUI = renderWeightsUI;
