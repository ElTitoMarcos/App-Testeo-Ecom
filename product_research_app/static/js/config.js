import * as api from './net.js';

const WEIGHT_FIELDS = [
  { key: 'price',        label: 'Price' },
  { key: 'rating',       label: 'Rating' },
  { key: 'units_sold',   label: 'Units Sold' },
  { key: 'revenue',      label: 'Revenue' },
  { key: 'desire',       label: 'Desire' },
  { key: 'competition',  label: 'Competition' },
  { key: 'oldness',      label: 'Oldness (antigüedad)' }
];
const WEIGHT_KEYS = WEIGHT_FIELDS.map(f => f.key);
const EXTREMES = {
  price:       { left: 'Más barato',       right: 'Más caro' },
  rating:      { left: 'Peor rating',      right: 'Mejor rating' },
  units_sold:  { left: 'Menos ventas',     right: 'Más ventas' },
  revenue:     { left: 'Menores ingresos', right: 'Mayores ingresos' },
  desire:      { left: 'Menor deseo',      right: 'Mayor deseo' },
  competition: { left: 'Menos competencia', right: 'Más competencia' },
  oldness:     { left: 'Más reciente',     right: 'Más antiguo' }
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

function clampWeight(v){ return Math.max(0, Math.min(100, Math.round(v))); }
let saveTimer=null, dirty=false;
function markDirty(){ dirty=true; clearTimeout(saveTimer); saveTimer=setTimeout(saveIfDirty,700); }
function clearDirty(){ dirty=false; clearTimeout(saveTimer); }
async function saveIfDirty(){ if(!dirty) return; await saveSettings(); dirty=false; }

function renderFactors(){
  const list = document.getElementById('weightsList');
  if(!list) return;
  list.innerHTML = '';
  factors.forEach((f,idx) => {
    const priority = idx+1;
    const li = document.createElement('li');
    li.className = 'weight-card';
    li.dataset.key = f.key;
    li.innerHTML = `<div class="priority-badge">#${priority}</div><div class="content"><label for="weight-${f.key}" class="label">${f.label}</label><input id="weight-${f.key}" class="weight-range" type="range" min="0" max="100" step="1" value="${f.weight}"><div class="slider-extremes scale"><span class="extreme-left">${EXTREMES[f.key].left}</span><span class="extreme-right">${EXTREMES[f.key].right}</span></div><span class="weight-badge">peso: ${f.weight}/100</span></div><div class="drag-handle" aria-hidden>≡</div>`;
    const range = li.querySelector('.weight-range');
    range.addEventListener('input', e => {
      const v = parseInt(e.target.value,10);
      e.target.value = v;
      f.weight = v;
      li.querySelector('.weight-badge').textContent = `peso: ${f.weight}/100`;
      markDirty();
    });
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
    winner_weights: Object.fromEntries(factors.map(x=>[x.key, clampWeight(x.weight)])),
    winner_order: factors.map(x=>x.key)
  };
  try{
    await api.updateSettings(payload);
    clearDirty();
  }catch(err){
    console.error('Error saving weights',err);
  }
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
  try{
    let sample=stratifiedSample(allProducts||[],30);
    let payload=sample.map(p=>({
      price:p.price,
      rating:p.rating || (p.extras&&p.extras.rating),
      units_sold:p.units_sold||(p.extras&&p.extras['Item Sold'])||0,
      revenue:p.revenue||(p.extras&&p.extras['Revenue($)'])||0,
      desire:p.desire_magnitude,
      competition:p.competition_level,
      oldness:computeOldnessDays(p)
    }));
    let tokenEstimate=JSON.stringify(payload).length/4;
    const maxTokens=0.30/0.002*1000;
    while(tokenEstimate>maxTokens && sample.length>1){
      const ratio=maxTokens/tokenEstimate;
      const newN=Math.max(1,Math.floor(sample.length*ratio));
      sample=stratifiedSample(allProducts||[],newN);
      payload=sample.map(p=>({
        price:p.price,
        rating:p.rating || (p.extras&&p.extras.rating),
        units_sold:p.units_sold||(p.extras&&p.extras['Item Sold'])||0,
        revenue:p.revenue||(p.extras&&p.extras['Revenue($)'])||0,
        desire:p.desire_magnitude,
        competition:p.competition_level,
        oldness:computeOldnessDays(p)
      }));
      tokenEstimate=JSON.stringify(payload).length/4;
    }
    const cost=tokenEstimate/1000*0.002;
    toast.info(`Analizando ${sample.length} productos (~$${cost.toFixed(2)})`);
    const instruction='Devuelve SOLO un JSON plano con pesos 0-100 para estas 7 claves exactas, sin texto adicional:\n["price","rating","units_sold","revenue","desire","competition","oldness"]';
    const prompt=`Basado en estos productos ${JSON.stringify(payload)}\n${instruction}`;
    let res=await fetch('/custom_gpt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,response_format:{type:'json_object'}})});
    if(!res.ok){
      toast.error('No se pudo ajustar por IA. Revisa tu API Key o inténtalo más tarde.');
      return;
    }
    const data=await res.json();
    let raw=data.response||'';
    console.debug('IA raw:', raw);
    let obj;
    if(typeof raw==='string'){
      try{ obj=JSON.parse(raw); }
      catch(e){
        const m=raw.match(/\{[\s\S]*\}/);
        if(m){ try{ obj=JSON.parse(m[0]); } catch(e2){} }
      }
    }else if(typeof raw==='object'){
      obj=raw;
    }
    console.debug('IA parsed:', obj);
    if(!obj || typeof obj!=='object'){
      toast.error('No se pudo ajustar por IA. Revisa tu API Key o inténtalo más tarde.');
      return;
    }
    const newWeights={};
    for(const k of metricKeys){
      let v=Number(obj[k]);
      if(isNaN(v)){
        const current=factors.find(f=>f.key===k);
        v=current?current.weight:0;
      }
      if(v<0) v=0; if(v>100) v=100;
      newWeights[k]=Math.round(v);
    }
    factors=factors.map(f=>({ ...f, weight:newWeights[f.key] ?? f.weight }));
    renderFactors();
    markDirty();
    toast.success('Pesos ajustados por IA');
  }catch(err){
    console.error(err);
    toast.error('No se pudo ajustar por IA. Revisa tu API Key o inténtalo más tarde.');
  }
}

async function loadWeights(){
  try{
    const res = await fetch('/api/config/winner-weights');
    const data = await res.json();
    const weights = data.weights || {};
    const order = Array.isArray(data.order) && data.order.length ? data.order : WEIGHT_KEYS;
    const base = {};
    WEIGHT_FIELDS.forEach(f=>{ base[f.key] = { ...f, weight:50 }; });
    factors = order.filter(k=>base[k]).map(k=>({ ...base[k], weight: weights[k] !== undefined ? Math.round(weights[k]) : 50 }));
    renderFactors();
    clearDirty();
    const resetBtn=document.getElementById('btnReset');
    if(resetBtn) resetBtn.onclick=resetWeights;
    const aiBtn=document.getElementById('btnAiWeights');
    if(aiBtn) aiBtn.onclick=adjustWeightsAI;
  }catch(err){
    console.error('Error loading weights', err);
  }
}

window.loadWeights = loadWeights;
window.resetWeights = resetWeights;
window.adjustWeightsAI = adjustWeightsAI;
window.markDirty = markDirty;
window.saveIfDirty = saveIfDirty;
window.metricKeys = metricKeys;
