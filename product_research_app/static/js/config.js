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
const ALIASES = { unitsSold: 'units_sold', orders: 'units_sold' };
function normalizeKey(k){ return ALIASES[k] || k; }
const metricDefs = WEIGHT_FIELDS;
const metricKeys = WEIGHT_KEYS;

function defaultState(){
  return {
    order: WEIGHT_KEYS.slice(),
    weights: Object.fromEntries(WEIGHT_KEYS.map(k => [k,50])),
    enabled: Object.fromEntries(WEIGHT_KEYS.map(k => [k,true]))
  };
}
let settingsState = defaultState();

let saveTimer = null;
function markDirty(){
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveSettings,700);
}
function saveIfDirty(){
  if(saveTimer){
    clearTimeout(saveTimer);
    saveSettings();
  }
}

function renderWeightsUI(state){
  const list = document.getElementById('weightsList');
  if(!list) return;
  list.innerHTML='';
  state.order.forEach((key, idx)=>{
    const def = WEIGHT_FIELDS.find(f=>f.key===key);
    if(!def) return;
    const li = document.createElement('li');
    li.className = 'weight-item weight-card';
    li.dataset.key = key;
    if(!state.enabled[key]) li.classList.add('disabled');
    li.innerHTML = `
      <div class="priority-badge">#${idx+1}</div>
      <div class="content">
        <div class="label">${def.label}</div>
        <div class="controls">
          <input type="number" min="0" max="100" step="1" class="weight-input" value="${state.weights[key]}">
          <input type="checkbox" class="enable-toggle" ${state.enabled[key] ? 'checked' : ''}>
        </div>
      </div>
      <div class="drag-handle" title="Arrastra para reordenar">≡</div>
    `;
    const numInput = li.querySelector('.weight-input');
    const toggle = li.querySelector('.enable-toggle');
    numInput.addEventListener('input', e=>{
      const v = Math.max(0, Math.min(100, parseInt(e.target.value,10) || 0));
      state.weights[key] = v;
      numInput.value = v;
      markDirty();
    });
    toggle.addEventListener('change', ()=>{
      state.enabled[key] = toggle.checked;
      numInput.disabled = !toggle.checked;
      li.classList.toggle('disabled', !toggle.checked);
      markDirty();
    });
    numInput.disabled = !state.enabled[key];
    list.appendChild(li);
  });
  Sortable.create(list,{ handle:'.drag-handle', animation:150, onEnd:()=>{
    state.order = Array.from(list.children).map(li=>li.dataset.key);
    renderWeightsUI(state);
    markDirty();
  }});
}

function resetWeights(){
  settingsState = defaultState();
  renderWeightsUI(settingsState);
  markDirty();
}

async function saveSettings(){
  const payload = {
    weights: {...settingsState.weights},
    order: [...settingsState.order],
    weights_order: [...settingsState.order],
    weights_enabled: {...settingsState.enabled}
  };
  try{
    await fetch('/api/config/winner-weights', {
      method:'PATCH',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    if (typeof reloadProductsLight === 'function') reloadProductsLight();
    else if (typeof reloadProducts === 'function') reloadProducts();
  }catch(err){
    console.warn('saveSettings failed', err);
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

    const features = (typeof metricKeys !== 'undefined' && metricKeys.length) ? metricKeys : [
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

    settingsState.weights = { ...settingsState.weights, ...intWeights };
    settingsState.order = newOrder.filter(k => settingsState.weights[k] !== undefined);
    renderWeightsUI(settingsState);

    await fetch('/api/config/winner-weights', {
      method:'PATCH', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        weights: intWeights,
        order: newOrder,
        weights_order: newOrder,
        weights_enabled: { ...settingsState.enabled }
      })
    });

    if (typeof toast !== 'undefined' && toast.success){
      const method = (out && out.method) ? out.method : 'auto';
      toast.success(`Pesos ajustados por IA (${method}) con ${data_sample.length} muestras`);
    }
  }catch(err){
    console.error(err);
    if (typeof toast !== 'undefined' && toast.error){
      toast.error('No se pudo ajustar por IA. Revisa tu API Key o inténtalo más tarde.');
    }
  }
}

async function hydrateSettingsModal(){
  try{
    const res = await fetch('/api/config/winner-weights');
    const data = await res.json();

    const weights = (data && data.weights) ? data.weights : (data || {});
    const order = (data && data.weights_order && Array.isArray(data.weights_order)) ? data.weights_order
                : (data && Array.isArray(data.order) && data.order.length ? data.order : WEIGHT_KEYS.slice());
    const enabled = (data && data.weights_enabled) ? data.weights_enabled : Object.fromEntries(Object.keys(weights).map(k => [k,true]));

    const base = defaultState();
    base.order = order.filter(k => WEIGHT_KEYS.includes(k)).concat(WEIGHT_KEYS.filter(k=>!order.includes(k)));
    base.weights = { ...base.weights, ...Object.fromEntries(Object.entries(weights).map(([k,v]) => [k, Math.max(0, Math.min(100, Math.round(Number(v))))])) };
    base.enabled = { ...base.enabled, ...Object.fromEntries(Object.entries(enabled).map(([k,v]) => [k, !!v])) };
    settingsState = base;

    renderWeightsUI(settingsState);

    const resetBtn = document.getElementById('btnReset');
    if (resetBtn) resetBtn.onclick = resetWeights;
    const aiBtn = document.getElementById('btnAiWeights');
    if (aiBtn) aiBtn.onclick = adjustWeightsAI;
  }catch(err){
    console.error('Error loading weights', err);
  }
}

window.hydrateSettingsModal = hydrateSettingsModal;
window.resetWeights = resetWeights;
window.adjustWeightsAI = adjustWeightsAI;
window.markDirty = markDirty;
window.saveIfDirty = saveIfDirty;
window.metricKeys = metricKeys;
