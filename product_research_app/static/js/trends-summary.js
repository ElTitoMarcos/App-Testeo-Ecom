import { fetchJson } from './net.js';
import { fmtInt, fmtPrice, fmtPct, fmtFloat2 } from './format.js';

function toISOFromDDMMYYYY(v) {
  const s = (v || '').trim();
  const m = s.match(/^(\d{2})\/(\d{2})\/(\d{4})$/);
  if (!m) return null;
  const [, dd, mm, yyyy] = m;
  return `${yyyy}-${mm}-${dd}`;
}
function formatDDMMYYYY(d) {
  const dd = String(d.getDate()).padStart(2, '0');
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const yyyy = d.getFullYear();
  return `${dd}/${mm}/${yyyy}`;
}

const container = document.getElementById('trendsSummary');
const btn = document.getElementById('trendsBtn');
const $desde = document.querySelector('#fecha-desde');
const $hasta = document.querySelector('#fecha-hasta');
const $btnAplicar = document.querySelector('#btn-aplicar-tendencias');
const metricButtons = document.querySelectorAll('#topCatCard .metric-btn');
const toggleLogBtn = document.getElementById('toggleLog');

document.addEventListener('DOMContentLoaded', () => {
  try {
    const today = new Date();
    const from = new Date(today);
    from.setDate(today.getDate() - 29);
    if ($desde && !$desde.value) $desde.value = formatDDMMYYYY(from);
    if ($hasta && !$hasta.value) $hasta.value = formatDDMMYYYY(today);
    fetchTrends();
  } catch (_) {}
});

if ($btnAplicar) {
  $btnAplicar.addEventListener('click', (ev) => {
    ev.preventDefault();
    fetchTrends();
  });
}

let currentMetric = 'revenue';
let scatterLog = false;
let currentData = null;
let prevData = null;
let revenueSpark, unitsSpark, topCatChart, scatterChart;

function showSkeleton() {
  document.getElementById('kpiGrid').innerHTML = '<div class="skeleton"></div>'.repeat(6);
}

async function fetchTrends() {
  showSkeleton();
  const url = new URL('/api/trends/summary', window.location.origin);
  const fromISO = $desde ? toISOFromDDMMYYYY($desde.value) : null;
  const toISO = $hasta ? toISOFromDDMMYYYY($hasta.value) : null;
  if (fromISO) url.searchParams.set('from', fromISO);
  if (toISO) url.searchParams.set('to', toISO);
  try {
    currentData = await fetchJson(url.toString());
    if (fromISO && toISO) {
      const start = new Date(fromISO);
      const end = new Date(toISO);
      const diff = end.getTime() - start.getTime();
      const prevFrom = new Date(start.getTime() - diff).toISOString().slice(0,10);
      const prevUrl = new URL('/api/trends/summary', window.location.origin);
      prevUrl.searchParams.set('from', prevFrom);
      prevUrl.searchParams.set('to', fromISO);
      prevData = await fetchJson(prevUrl.toString());
    }
    render();
  } catch (e) {
    // fetchJson already toasts
  }
}

function computeTotals(data) {
  return data.totals || {
    unique_products: data.categories.reduce((a,c)=>a+c.unique_products,0),
    units: data.categories.reduce((a,c)=>a+c.units,0),
    revenue: data.categories.reduce((a,c)=>a+c.revenue,0),
    avg_price: 0,
    avg_rating: 0,
    rev_per_unit: 0,
  };
}

function render() {
  const totals = computeTotals(currentData);
  const prevTotals = computeTotals(prevData);
  const deltaRev = prevTotals.revenue ? ((totals.revenue - prevTotals.revenue)/prevTotals.revenue)*100 : 0;
  const deltaUnits = prevTotals.units ? ((totals.units - prevTotals.units)/prevTotals.units)*100 : 0;
  const kpiGrid = document.getElementById('kpiGrid');
  kpiGrid.innerHTML = `
    <div class="kpi"><div class="kpi-value">${fmtInt(totals.unique_products)}</div><div class="kpi-label">Productos únicos</div></div>
    <div class="kpi"><div class="kpi-value">${fmtInt(totals.units)}</div><div class="kpi-label">Unidades</div><div class="kpi-delta" style="color:${deltaUnits>=0?'#4caf50':'#e53935'};">${fmtPct(deltaUnits)}</div></div>
    <div class="kpi"><div class="kpi-value">${fmtPrice(totals.revenue)}</div><div class="kpi-label">Ingresos</div><div class="kpi-delta" style="color:${deltaRev>=0?'#4caf50':'#e53935'};">${fmtPct(deltaRev)}</div></div>
    <div class="kpi"><div class="kpi-value">${fmtPrice(totals.rev_per_unit)}</div><div class="kpi-label">Rev/Unidad</div></div>
    <div class="kpi"><div class="kpi-value">${fmtPrice(totals.avg_price)}</div><div class="kpi-label">Precio medio</div></div>
    <div class="kpi"><div class="kpi-value">${fmtFloat2(totals.avg_rating)}</div><div class="kpi-label">Rating medio</div></div>`;
  renderCharts();
  renderTable();
}

function renderCharts() {
  const labels = currentData.timeseries.map(p=>p.date);
  const revData = currentData.timeseries.map(p=>p.revenue);
  const unitsData = currentData.timeseries.map(p=>p.units);
  const sparkOpts = {responsive:true, maintainAspectRatio:false, scales:{x:{display:false}, y:{display:false}}, elements:{line:{tension:0.3}, point:{radius:0}}, plugins:{legend:{display:false}}};
  if(revenueSpark) revenueSpark.destroy();
  revenueSpark = new Chart(document.getElementById('sparkRevenue'), {type:'line', data:{labels, datasets:[{data:revData,borderColor:'#42a5f5',fill:false}]}, options:sparkOpts});
  if(unitsSpark) unitsSpark.destroy();
  unitsSpark = new Chart(document.getElementById('sparkUnits'), {type:'line', data:{labels, datasets:[{data:unitsData,borderColor:'#66bb6a',fill:false}]}, options:sparkOpts});

  const top = currentData.categories.slice(0,10);
  const labelsCat = top.map(c=>c.category);
  const values = top.map(c=>c[currentMetric]);
  if(topCatChart) topCatChart.destroy();
  topCatChart = new Chart(document.getElementById('topCatChart'), {
    type:'bar',
    data:{labels:labelsCat, datasets:[{data:values, backgroundColor:'#42a5f5'}]},
    options:{indexAxis:'y', responsive:true, maintainAspectRatio:false, scales:{x:{grid:{display:false}, ticks:{callback:v=>fmtInt(v)}}, y:{grid:{display:false}}}, plugins:{legend:{display:false}, tooltip:{callbacks:{label:ctx=>fmtInt(ctx.parsed.x)}}}, maxBarThickness:24}
  });

  const scatterData = currentData.categories.map(c=>({x:c.avg_price, y:c.revenue, label:c.category, units:c.units, avg_price:c.avg_price, revenue:c.revenue, avg_rating:c.avg_rating}));
  if(scatterChart) scatterChart.destroy();
  scatterChart = new Chart(document.getElementById('priceRevChart'), {
    type:'scatter',
    data:{datasets:[{data:scatterData, backgroundColor:'#7e57c2'}]},
    options:{responsive:true, maintainAspectRatio:false, scales:{x:{type:scatterLog?'logarithmic':'linear'}, y:{}}, plugins:{legend:{display:false}, tooltip:{callbacks:{label:ctx=>{const d=ctx.raw; return `${d.label}\nIngresos: ${fmtPrice(d.revenue)}\nUnidades: ${fmtInt(d.units)}\nPrecio: ${fmtPrice(d.avg_price)}\nRating: ${fmtFloat2(d.avg_rating)}`;}}}}}
  });
}

function renderTable(){
  const tbl = document.getElementById('topCatTable');
  const rows = currentData.categories.slice(0,10);
  let html='<thead><tr><th>Cat.</th><th>Productos</th><th>Unidades</th><th>Ingresos</th><th>Precio</th><th>Rating</th></tr></thead><tbody>';
  rows.forEach(c=>{
    html+=`<tr><td>${c.category}</td><td>${fmtInt(c.unique_products)}</td><td>${fmtInt(c.units)}</td><td>${fmtPrice(c.revenue)}</td><td>${fmtPrice(c.avg_price)}</td><td>${fmtFloat2(c.avg_rating)}</td></tr>`;
  });
  html+='</tbody>';
  tbl.innerHTML = html;
}

btn?.addEventListener('click', () => {
  container.style.display = container.style.display === 'block' ? 'none' : 'block';
  if(container.style.display === 'block') fetchTrends();
});

metricButtons.forEach(btn => btn.addEventListener('click', e => {
  metricButtons.forEach(b=>b.classList.remove('active'));
  e.currentTarget.classList.add('active');
  currentMetric = e.currentTarget.dataset.metric;
  renderCharts();
}));

toggleLogBtn?.addEventListener('click', () => {
  scatterLog = !scatterLog;
  renderCharts();
});

export {};
