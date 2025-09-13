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
  return WEIGHT_FIELDS.map(f => ({ ...f, weight:50 }));
}
let saveTimer=null;
function markDirty(){
  clearTimeout(saveTimer);
  saveTimer=setTimeout(saveSettings,700);
}

function renderFactors(){
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
        markDirty();
      });
    } else {
      li.innerHTML = `<div class="priority-badge">#${priority}</div><div class="content"><label for="weight-${f.key}" class="label">${f.label}</label><input id="weight-${f.key}" class="weight-range" type="range" min="0" max="100" step="1" value="${f.weight}"><div class="slider-extremes scale"><span class="extreme-left">${EXTREMES[f.key].left}</span><span class="extreme-right">${EXTREMES[f.key].right}</span></div><span class="weight-badge">peso: ${f.weight}/100</span></div><div class="drag-handle" aria-hidden>≡</div>`;
      const range = li.querySelector('.weight-range');
      range.addEventListener('input', e => {
        const v = Math.max(0, Math.min(100, parseInt(e.target.value,10) || 0));
        f.weight = v;
        range.value = v;
        li.querySelector('.weight-badge').textContent = `peso: ${f.weight}/100`;
        markDirty();
      });
    }
    list.appendChild(li);
  });
  Sortable.create(list,{ handle:'.drag-handle', animation:150, onEnd:()=>{
    const orderKeys = Array.from(list.children).map(li=>li.dataset.key);
    factors.sort((a,b)=>orderKeys.indexOf(a.key)-orderKeys.indexOf(b.key));
    renderFactors();
    markDirty();
  }});
}

function resetWeights(){
  factors = defaultFactors();
  renderFactors();
  markDirty();
}

async function saveSettings(){
  const payload = {
    // El backend espera "weights" (0–100 enteros)
    weights: Object.fromEntries(
      factors.map(f => [f.key, Math.max(0, Math.min(100, Math.round(Number(f.weight))))])
    ),
    // Persistimos también el orden visible en la UI
    order: factors.map(f => f.key)
  };

  try{
    // Usa la misma librería "api" que ya importa config.js
    const res = await api.patch('/api/config/winner-weights', payload);
    if (typeof reloadProductsLight === 'function') {
      reloadProductsLight();
    } else if (typeof reloadProducts === 'function') {
      reloadProducts();
    }
  }catch(err){
    console.warn('saveSettings failed', err);
    if (typeof toast !== 'undefined' && toast && toast.error) {
      toast.error('No se pudo guardar la configuración');
    }
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
  // Helpers locales para robustez (no toques computeOldnessDays ni awarenessValue)
  const num = (v) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  };
  const stratifiedSampleBy = (arr, key, n) => {
    if (!Array.isArray(arr) || arr.length <= n) return (arr || []).slice();
    const sorted = [...arr].sort((a,b) => num(b[key]) - num(a[key]));
    const out = [];
    for (let i = 0; i < n; i++){
      const idx = Math.floor(i * (sorted.length - 1) / Math.max(1,(n - 1)));
      out.push(sorted[idx]);
    }
    return out;
  };

  try{
    const products = Array.isArray(window.allProducts) ? window.allProducts : [];
    if (!products.length){
      if (typeof toast !== 'undefined') toast.info('No hay productos cargados');
      return;
    }

    // Construimos dataset con las 8 features del Winner Score
    const rows = products.map(p => {
      const ratingRaw = p.rating ?? (p.extras && p.extras.rating);
      const unitsRaw  = p.units_sold ?? (p.extras && p.extras['Item Sold']);
      const revRaw    = p.revenue ?? (p.extras && p.extras['Revenue($)']);
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

    // Target: preferimos 'revenue' si está presente en ≥50% de filas, si no 'units_sold'
    const revCount = rows.filter(r => r.revenue > 0).length;
    const targetName = (revCount >= Math.ceil(rows.length * 0.5)) ? 'revenue' : 'units_sold';

    // data_sample: el backend espera un campo 'target' numérico en cada fila
    let data_sample = rows.map(r => ({ ...r, target: num(r[targetName]) }));

    // Para controlar coste: si hay muchísimos, muestrear de forma estratificada por el target
    const MAX = 150; // puedes subirlo si te compensa el coste
    if (data_sample.length > MAX){
      data_sample = stratifiedSampleBy(data_sample, targetName, MAX);
    }

    // Payload para el endpoint GPT (features -> las 8 claves del score)
    const features = (typeof metricKeys !== 'undefined' && metricKeys.length) ? metricKeys : [
      'price','rating','units_sold','revenue','desire','competition','oldness','awareness'
    ];
    const payload = { features, target: targetName, data_sample };

    // 1º intento: GPT
    let res = await fetch('/scoring/v2/auto-weights-gpt', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)
    });

    // Fallback: correlación estadística si GPT falla
    if (!res.ok){
      res = await fetch('/scoring/v2/auto-weights-stat', {
        method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)
      });
    }
    if (!res.ok) throw new Error('Auto-weights request failed');

    const data = await res.json(); // { weights: {k:0..1}, method:'gpt'|'stat' }
    const returned = data && data.weights ? data.weights : {};
    const newWeights = {};
    features.forEach(k => {
      let v = num(returned[k]);
      // Si viene 0..1, escalar a 0..100
      if (v > 0 && v <= 1) v = v * 100;
      if (v < 0) v = 0; if (v > 100) v = 100;
      newWeights[k] = Math.round(v);
    });

    // Aplicar en UI y persistir
    if (Array.isArray(window.factors) && window.factors.length){
      window.factors = window.factors.map(f => ({ ...f, weight: newWeights[f.key] ?? f.weight }));
      if (typeof renderFactors === 'function') renderFactors();
    }
    if (typeof saveSettings === 'function') await saveSettings();

    if (typeof toast !== 'undefined' && toast.success){
      const method = (data && data.method) ? data.method : 'auto';
      toast.success(`Pesos ajustados por IA (${method})`);
    }
  }catch(err){
    console.error(err);
    if (typeof toast !== 'undefined' && toast.error){
      toast.error('No se pudo ajustar por IA. Revisa tu API Key o inténtalo más tarde.');
    }
  }
}


async function openConfigModal(){
  try{
    const res = await fetch('/api/config/winner-weights');
    const data = await res.json();

    // El backend devuelve { weights, order, effective:{int:...} }
    const weights = (data && data.weights) ? data.weights : {};
    const order   = (data && Array.isArray(data.order) && data.order.length)
      ? data.order
      : (typeof WEIGHT_KEYS !== 'undefined' ? WEIGHT_KEYS : Object.keys(weights));

    // WEIGHT_FIELDS existe en este módulo; lo indexamos por key
    const byKey = Object.fromEntries((WEIGHT_FIELDS || []).map(f => [f.key, f]));
    const orderedKeys = (order && order.length) ? order : Object.keys(byKey);

    // Construimos factors respetando el orden persistido y aplicando los pesos guardados (fallback 50)
    window.factors = orderedKeys
      .filter(k => byKey[k]) // ignora claves desconocidas
      .map(k => ({
        ...byKey[k],
        weight: (weights[k] !== undefined && !isNaN(weights[k])) ? Math.round(Number(weights[k])) : 50
      }));

    renderFactors();

    const resetBtn = document.getElementById('btnReset');
    if (resetBtn) resetBtn.onclick = resetWeights;

    const aiBtn = document.getElementById('btnAiWeights');
    if (aiBtn) aiBtn.onclick = adjustWeightsAI;

  }catch(err){
    console.error('Error loading weights', err);
  }
}

window.openConfigModal = openConfigModal;
window.loadWeights = openConfigModal; // alias for legacy calls
window.resetWeights = resetWeights;
window.adjustWeightsAI = adjustWeightsAI;
window.markDirty = markDirty;
window.metricKeys = metricKeys;
