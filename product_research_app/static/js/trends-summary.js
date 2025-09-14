import { fmtInt, fmtPrice, fmtFloat2 } from './format.js';

function formatMoneyShort(n){
  const v = Number(n) || 0;
  const abs = Math.abs(v);
  if (abs >= 1e9) return (v/1e9).toFixed(1).replace('.0','') + 'B';
  if (abs >= 1e6) return (v/1e6).toFixed(1).replace('.0','') + 'M';
  if (abs >= 1e3) return (v/1e3).toFixed(1).replace('.0','') + 'K';
  return v.toLocaleString('es-ES', { maximumFractionDigits: 0 });
}
function toISOFromDDMMYYYY(v){ const s=(v||'').trim(); const m=s.match(/^(\d{2})\/(\d{2})\/(\d{4})$/); if(!m) return null; const[,dd,mm,yyyy]=m; return `${yyyy}-${mm}-${dd}`; }
function formatDDMMYYYY(d){ const dd=String(d.getDate()).padStart(2,'0'); const mm=String(d.getMonth()+1).padStart(2,'0'); const yyyy=d.getFullYear(); return `${dd}/${mm}/${yyyy}`; }
function shortPathLabel(path){ if(!path) return ''; const parts=String(path).split(' > '); const last=parts.slice(-2).join(' › '); return last.length>32 ? last.slice(0,29)+'…' : last; }

(function wireTrendsToggle(){
  const btn = document.querySelector('#btn-ver-tendencias');
  const secTrends = document.querySelector('#section-trends');
  const secProducts = document.querySelector('#section-products');
  if (!btn || !secTrends || !secProducts) return;

  function openTrends(){
    secTrends.hidden = false;
    secProducts.hidden = true;
    btn.classList.add('active');
    initDatesIfEmpty();
    fetchTrends();
  }
  function closeTrends(){
    secTrends.hidden = true;
    secProducts.hidden = false;
    btn.classList.remove('active');
  }

  btn.addEventListener('click', (e)=>{
    e.preventDefault();
    (secTrends.hidden ? openTrends : closeTrends)();
  });

  document.addEventListener('keydown', (e)=>{
    if (e.key === 'Escape' && !secTrends.hidden) closeTrends();
  });

  function initDatesIfEmpty(){
    const $desde = document.querySelector('#fecha-desde');
    const $hasta = document.querySelector('#fecha-hasta');
    const today = new Date();
    const from = new Date(today); from.setDate(today.getDate()-29);
    if ($desde && !$desde.value) $desde.value = formatDDMMYYYY(from);
    if ($hasta && !$hasta.value) $hasta.value = formatDDMMYYYY(today);
  }
})();

const $btnAplicar = document.querySelector('#btn-aplicar-tendencias');
if ($btnAplicar) {
  $btnAplicar.addEventListener('click', (ev)=>{
    ev.preventDefault();
    fetchTrends();
  });
}

async function fetchTrends(){
  try{
    const $desde = document.querySelector('#fecha-desde');
    const $hasta = document.querySelector('#fecha-hasta');
    const fISO = $desde ? toISOFromDDMMYYYY($desde.value) : null;
    const tISO = $hasta ? toISOFromDDMMYYYY($hasta.value) : null;
    const url = new URL('/api/trends/summary', window.location.origin);
    if (fISO) url.searchParams.set('from', fISO);
    if (tISO) url.searchParams.set('to', tISO);
    const res = await fetch(url.toString(), { credentials:'same-origin' });
    if (!res.ok) throw new Error('HTTP '+res.status);
    const json = await res.json();
    renderTrends(json);
    renderCategoriasTable(json);
  }catch(e){
    (window.toast?.error || alert).call(window.toast||window, 'No se pudieron cargar las tendencias.');
  }
}

function renderTrends(summary){
  if (!summary) return;
  renderTopCategoriesBar(summary);
  renderParetoHorizontal(summary);
}

function renderTopCategoriesBar(data){
  const list = Array.isArray(data.top_categories) ? data.top_categories.slice(0,10) : [];
  const labels = list.map(x => shortPathLabel(x.path));
  const values = list.map(x => x.revenue || 0);

  const ctx = document.getElementById('chart-top-categories');
  if (!ctx) return;
  if (ctx._chart) ctx._chart.destroy();

  ctx._chart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets:[{ data: values, borderWidth:0 }]},
    options: {
      indexAxis: 'y',
      maintainAspectRatio: false,
      plugins:{
        legend:{ display:false },
        tooltip:{ callbacks:{ label:(tt)=>`Ingresos: ${formatMoneyShort(tt.parsed.x)}` }}
      },
      scales:{
        x:{ grid:{ display:false }, ticks:{ callback:(v)=>formatMoneyShort(v) }},
        y:{ grid:{ display:false } }
      }
    }
  });
}

function renderParetoHorizontal(data){
  const src = Array.isArray(data.top_categories) ? [...data.top_categories] : [];
  src.sort((a,b)=> (b.revenue||0)-(a.revenue||0));
  const top = src.slice(0,10);
  const labels = top.map(x => shortPathLabel(x.path));
  const ingresos = top.map(x => x.revenue||0);
  const total = ingresos.reduce((s,n)=>s+n,0) || 1;
  let acc = 0;
  const pct = ingresos.map(v=>{ acc+=v; return +(acc/total*100).toFixed(1); });

  const ctx = document.getElementById('chart-pareto');
  if (!ctx) return;
  if (ctx._chart) ctx._chart.destroy();

  ctx._chart = new Chart(ctx, {
    data:{
      labels,
      datasets:[
        { type:'bar',  label:'Ingresos',    data:ingresos, xAxisID:'x',  borderWidth:0 },
        { type:'line', label:'% acumulado', data:pct,      xAxisID:'x1', tension:0.3, pointRadius:2 }
      ]
    },
    options:{
      indexAxis:'y',
      maintainAspectRatio:false,
      plugins:{
        legend:{ display:true },
        tooltip:{ callbacks:{ label:(tt)=> tt.datasetIndex===0 ? `Ingresos: ${formatMoneyShort(tt.parsed.x)}` : `% acumulado: ${tt.parsed.x}%` }}
      },
      scales:{
        y:{ grid:{ display:false }},
        x:{ grid:{ display:false }, ticks:{ callback:(v)=>formatMoneyShort(v) }},
        x1:{ position:'top', min:0, max:100, grid:{ display:false }, ticks:{ callback:(v)=> v + '%' }}
      }
    }
  });
}

(function enableSortableAndToggleRows(){
  const table = document.getElementById('tbl-categorias');
  if (!table) return;
  const thead = table.querySelector('thead');
  const tbody = table.querySelector('tbody');
  if (!thead || !tbody) return;

  const parseNumber = (s)=>{ const t=String(s).replace(/\./g,'').replace(/,/g,'.').replace(/[^\d.-]/g,''); const n=parseFloat(t); return isNaN(n)?0:n; };
  const getCell = (tr,idx)=> tr.children[idx]?.textContent?.trim() || '';

  thead.addEventListener('click',(e)=>{
    const th = e.target.closest('th[data-sort-key]');
    if (!th) return;
    const idx = Array.from(th.parentNode.children).indexOf(th);
    thead.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));
    const asc = !th.classList.contains('sort-asc');
    th.classList.add(asc ? 'sort-asc' : 'sort-desc');
    const numeric = ['Productos','Unidades','Ingresos','Precio','Rating'].includes(th.textContent.trim());
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a,b)=>{
      const va = getCell(a,idx); const vb = getCell(b,idx);
      if (numeric) return asc ? (parseNumber(va)-parseNumber(vb)) : (parseNumber(vb)-parseNumber(va));
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    });
    rows.forEach(r=>tbody.appendChild(r));
  });

  const btn = document.getElementById('btn-toggle-rows');
  if (btn){
    let expanded = false;
    btn.addEventListener('click',()=>{
      expanded = !expanded;
      btn.textContent = expanded ? 'Ver menos' : 'Ver más';
      const rows = table.querySelectorAll('tbody tr');
      rows.forEach((tr,i)=> tr.style.display = (!expanded && i>=10) ? 'none' : '');
    });
  }
})();

function renderCategoriasTable(data){
  const tbody = document.querySelector('#tbl-categorias tbody');
  if (!tbody) return;
  const rows = Array.isArray(data.top_categories) ? data.top_categories : (data.categories || []);
  let html = '';
  rows.forEach(c => {
    const productos = c.products_count || c.products || c.unique_products || 0;
    const unidades = c.units || 0;
    const ingresos = c.revenue || 0;
    const precio = c.avg_price || 0;
    const rating = c.avg_rating || 0;
    html += `<tr><td>${c.path || c.category || ''}</td><td>${fmtInt(productos)}</td><td>${fmtInt(unidades)}</td><td>${formatMoneyShort(ingresos)}</td><td>${fmtPrice(precio)}</td><td>${fmtFloat2(rating)}</td></tr>`;
  });
  tbody.innerHTML = html;
}

  const $desde = document.querySelector('#fecha-desde');
  const $hasta = document.querySelector('#fecha-hasta');
  try {
    const today = new Date();
    const from = new Date(today); from.setDate(today.getDate() - 29);
    if ($desde && !$desde.value) $desde.value = formatDDMMYYYY(from);
    if ($hasta && !$hasta.value) $hasta.value = formatDDMMYYYY(today);
  } catch(_) {}

  if (typeof fetchTrends === 'function') {
    fetchTrends();
  } else {
    (async function(){
      const url = new URL('/api/trends/summary', window.location.origin);
      const res = await fetch(url.toString(), { credentials:'same-origin' });
      if (res.ok) {
        const json = await res.json();
        if (typeof renderTrends === 'function') renderTrends(json);
      } else {
        (window.toast?.error || alert).call(window.toast||window, 'No se pudieron cargar las tendencias.');
      }
    })();
  }

  const firstChart = document.querySelector('#chart-top-categories, #card-top-categories');
  if (firstChart && typeof firstChart.scrollIntoView === 'function') {
    firstChart.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

document.addEventListener('click', function(e){
  const btn = e.target.closest('#btn-ver-tendencias, .btn-ver-tendencias, [data-action="show-trends"]');
  if (!btn) return;
  e.preventDefault();
  showTrendsSection();
});
export {};

