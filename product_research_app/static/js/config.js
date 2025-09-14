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
let factors = [];
let userConfig = {};

function defaultFactors(){
  return WEIGHT_FIELDS.map(f => ({ ...f, weight:50, enabled:true }));
}
let saveTimer=null;
let isInitialRender = true;
function markDirty(){
  clearTimeout(saveTimer);
  saveTimer=setTimeout(saveSettings,700);
}

function renderWeightsUI(){
  const list = document.getElementById('weightsList');
  if(!list) return;
  list.innerHTML = '';
  factors.forEach((f,idx) => {
    const priority = idx+1;
    const li = document.createElement('li');
    li.className = 'weight-card';
    li.dataset.key = f.key;
    if(f.key === 'awareness'){
      li.innerHTML = `<div class="priority-badge">#${priority}</div><div class="content"><div class="label">Awareness</div>

    <div class="segmented-range">
      <input id="awarenessSlider" class="weight-slider seg-awareness" type="range" min="0" max="100" step="1" />
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
    </div>

    <div class="meta"><span class="weight-badge">peso: <span id="awarenessWeight"></span>/100</span></div>
  </div>
  <div class="drag-handle" title="Arrastra para reordenar">≡</div>`;
      const slider = li.querySelector('#awarenessSlider');
      const weightEl = li.querySelector('#awarenessWeight');
      const segs = Array.from(li.querySelectorAll('.awareness-labels span'));
      function updateAw(val){
        const v = Math.max(0, Math.min(100, parseInt(val,10) || 0));
        f.weight = v;
        slider.value = v;
        weightEl.textContent = v;
        const idx = Math.min(4, Math.floor(v/20));
        segs.forEach((el,i)=>el.classList.toggle('active', i===idx));
      }
      updateAw(f.weight);
      slider.addEventListener('input', e => {
        updateAw(e.target.value);
        if(!isInitialRender) markDirty();
      });
      const contentEl = li.querySelector('.content');
      const toggle = document.createElement('div');
      toggle.className = 'weight-toggle';
      toggle.innerHTML = '<input type="checkbox" class="wt-enabled" aria-label="Activar métrica" />';
      contentEl.appendChild(toggle);
      const cb = toggle.querySelector('input');
      cb.checked = f.enabled !== false;
      cb.addEventListener('change', e => {
        const on = !!e.target.checked;
        f.enabled = on;
        for (const el of li.querySelectorAll('input[type="range"], input[type="number"]')) {
          el.disabled = !on;
        }
        li.classList.toggle('disabled', !on);
        if(!isInitialRender) markDirty();
      });
      if(!cb.checked){
        for (const el of li.querySelectorAll('input[type="range"], input[type="number"]')) {
          el.disabled = true;
        }
        li.classList.add('disabled');
      }
    } else {
      li.innerHTML = `<div class="priority-badge">#${priority}</div><div class="content"><label for="weight-${f.key}" class="label">${f.label}</label><input id="weight-${f.key}" class="weight-range" type="range" min="0" max="100" step="1" value="${f.weight}"><div class="slider-extremes scale"><span class="extreme-left">${EXTREMES[f.key].left}</span><span class="extreme-right">${EXTREMES[f.key].right}</span></div><span class="weight-badge">peso: ${f.weight}/100</span></div><div class="drag-handle" aria-hidden>≡</div>`;
      const range = li.querySelector('.weight-range');
      range.addEventListener('input', e => {
        const v = Math.max(0, Math.min(100, parseInt(e.target.value,10) || 0));
        f.weight = v;
        range.value = v;
        li.querySelector('.weight-badge').textContent = `peso: ${f.weight}/100`;
        if(!isInitialRender) markDirty();
      });
      const contentEl = li.querySelector('.content');
      const toggle = document.createElement('div');
      toggle.className = 'weight-toggle';
      toggle.innerHTML = '<input type="checkbox" class="wt-enabled" aria-label="Activar métrica" />';
      contentEl.appendChild(toggle);
      const cb = toggle.querySelector('input');
      cb.checked = f.enabled !== false;
      cb.addEventListener('change', e => {
        const on = !!e.target.checked;
        f.enabled = on;
        for (const el of li.querySelectorAll('input[type="range"], input[type="number"]')) {
          el.disabled = !on;
        }
        li.classList.toggle('disabled', !on);
        if(!isInitialRender) markDirty();
      });
      if(!cb.checked){
        for (const el of li.querySelectorAll('input[type="range"], input[type="number"]')) {
          el.disabled = true;
        }
        li.classList.add('disabled');
      }
    }
    list.appendChild(li);
  });
  Sortable.create(list,{ handle:'.drag-handle', animation:150, onEnd:()=>{
    const orderKeys = Array.from(list.children).map(li=>li.dataset.key);
    factors.sort((a,b)=>orderKeys.indexOf(a.key)-orderKeys.indexOf(b.key));
    renderWeightsUI();
    if(!isInitialRender) markDirty();
  }});
  isInitialRender = false;
}

function resetWeights(){
  factors = defaultFactors();
  renderWeightsUI();
  markDirty();
}

async function saveSettings(){
  const payload = {
    // usar "weights" (no "winner_weights") y persistir el orden visible
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
    await fetch('/api/config/winner-weights', {
      method:'PATCH',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
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

    // 1 intento: GPT; fallback estadístico si falla
    let res = await fetch('/scoring/v2/auto-weights-gpt', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)
    });
    if (!res.ok){
      res = await fetch('/scoring/v2/auto-weights-stat', {
        method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)
      });
    }
    if (!res.ok) throw new Error('Auto-weights request failed');

    const out = await res.json(); // { weights:{k:0..1|0..100}, method?... }
    const returned = (out && out.weights) ? out.weights : {};

    // Normalizar a 0..100 enteros y ordenar por importancia
    const intWeights = {};
    for (const k of features){
      let v = num(returned[k]);
      if (v > 0 && v <= 1) v = v * 100;
      v = Math.max(0, Math.min(100, Math.round(v)));
      intWeights[k] = v;
    }
    const newOrder = [...features].sort((a,b) => (intWeights[b]||0) - (intWeights[a]||0));

    // Aplicar en UI
    if (Array.isArray(window.factors) && window.factors.length){
      const byKey = Object.fromEntries(window.factors.map(f => [f.key, f]));
      window.factors = newOrder.filter(k => byKey[k]).map(k => ({ ...byKey[k], weight: intWeights[k] ?? byKey[k].weight }));
      if (typeof renderWeightsUI === 'function') renderWeightsUI();
    }

    // Guardar {weights, order} y recargar desde servidor para reflejar lo persistido
    await fetch('/api/config/winner-weights', {
      method:'PATCH', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ weights: intWeights, order: newOrder })
    });
    if (typeof openConfigModal === 'function') await openConfigModal();

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


async function hydrateSettingsModal(){
  try{
    isInitialRender = true;
    const res = await fetch('/api/config/winner-weights');
    const data = await res.json(); // backend: { weights, order, weights_enabled? }

    const weights = (data && data.weights) ? data.weights : (data || {});
    const order   = (data && Array.isArray(data.order) && data.order.length)
      ? data.order
      : (typeof WEIGHT_KEYS !== 'undefined' ? WEIGHT_KEYS : Object.keys(weights));
    const enabled = (data && data.weights_enabled) ? data.weights_enabled : Object.fromEntries(Object.keys(weights).map(k => [k, true]));

    const fieldList = (typeof WEIGHT_FIELDS !== 'undefined' && Array.isArray(WEIGHT_FIELDS)) ? WEIGHT_FIELDS : [];
    const byKey = Object.fromEntries(fieldList.map(f => [f.key, f]));

    window.factors = order
      .filter(k => byKey[k])
      .map(k => ({
        ...byKey[k],
        weight: (weights[k] !== undefined && !isNaN(weights[k])) ? Math.round(Number(weights[k])) : 50,
        enabled: enabled[k] !== undefined ? !!enabled[k] : true
      }));

    renderWeightsUI();
    console.debug('hydrateSettingsModal -> weights/order aplicados:', { weights, order, enabled });
  }catch(err){
    /* silencioso */
  }
}

async function openConfigModal(){
  await hydrateSettingsModal();
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
