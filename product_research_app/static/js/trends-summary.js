import { fmtInt, fmtPrice, fmtFloat2 } from './format.js';

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

function formatMoney(v){
  if (v == null) return '0';
  const n = Number(v) || 0;
  return n.toLocaleString('es-ES', { maximumFractionDigits: 0 });
}

const $desde = document.querySelector('#fecha-desde');
const $hasta = document.querySelector('#fecha-hasta');
const $btnAplicar = document.querySelector('#btn-aplicar-tendencias');
const btnLog = document.getElementById('btn-log-trends');

let currentData = null;
let paretoLog = false;

if ($btnAplicar) {
  $btnAplicar.addEventListener('click', function(ev){
    ev.preventDefault();
    if (typeof fetchTrends === 'function') fetchTrends();
  });
}

btnLog?.addEventListener('click', (ev) => {
  ev.preventDefault();
  paretoLog = !paretoLog;
  renderPareto(currentData);
});

async function fetchTrends(){
  const $status = document.querySelector('#trends-status');
  try {
    if ($status) $status.textContent = 'Cargando...';
    const fISO = $desde ? toISOFromDDMMYYYY($desde.value) : null;
    const tISO = $hasta ? toISOFromDDMMYYYY($hasta.value) : null;
    const url = new URL('/api/trends/summary', window.location.origin);
    if (fISO) url.searchParams.set('from', fISO);
    if (tISO) url.searchParams.set('to', tISO);
    const res = await fetch(url.toString(), { credentials: 'same-origin' });
    if (!res.ok) throw new Error('HTTP '+res.status);
    const json = await res.json();
    currentData = json;
    renderTrends(json);
    renderCategoriasTable(json);
  } catch(e){
    (window.toast?.error || alert).call(window.toast||window, 'No se pudieron cargar las tendencias.');
  } finally {
    if ($status) $status.textContent = '';
  }
}

function renderTrends(summary){
  if(!summary) return;
  renderTopCategoriesBar(summary);
  renderPareto(summary);
}

function renderCategoriasTable(data){
  const tbody = document.querySelector('#tbl-categorias tbody');
  if(!tbody) return;
  const rows = [...(data.top_categories || data.categories || [])];
  let html = '';
  rows.forEach(c => {
    const productos = c.products_count || c.products || c.unique_products || 0;
    const unidades = c.units || 0;
    const ingresos = c.revenue || 0;
    const precio = c.avg_price || 0;
    const rating = c.avg_rating || 0;
    html += `<tr><td>${c.path || c.category || ''}</td><td>${fmtInt(productos)}</td><td>${fmtInt(unidades)}</td><td>${formatMoney(ingresos)}</td><td>${fmtPrice(precio)}</td><td>${fmtFloat2(rating)}</td></tr>`;
  });
  tbody.innerHTML = html;
}

function renderTopCategoriesBar(data) {
  const top = [...(data.top_categories || data.categories || [])].slice(0, 10);
  const labels = top.map(x => x.path || x.category);
  const values = top.map(x => x.revenue);

  const ctx = document.getElementById('chart-top-categories');
  if (!ctx) return;
  if (ctx._chart) { ctx._chart.destroy(); }

  ctx._chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ data: values, borderWidth: 0 }]
    },
    options: {
      indexAxis: 'y',
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (tt) => `Ingresos: ${formatMoney(tt.parsed.x)}`
          }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { callback: (v)=> formatMoney(v) } },
        y: { grid: { display: false } }
      }
    }
  });
}

function renderPareto(data) {
  if (!data) return;
  const src = [...(data.top_categories || data.categories || [])];
  src.sort((a,b) => (b.revenue||0) - (a.revenue||0));
  const top = src.slice(0, 10);

  const labels = top.map(x => x.path || x.category);
  const ingresos = top.map(x => x.revenue || 0);
  const total = ingresos.reduce((s,n)=>s+n, 0) || 1;
  let acc = 0;
  const acumuladoPct = ingresos.map(v => { acc += v; return +(acc/total*100).toFixed(1); });

  const ctx = document.getElementById('chart-pareto');
  if (!ctx) return;
  if (ctx._chart) { ctx._chart.destroy(); }

  ctx._chart = new Chart(ctx, {
    data: {
      labels,
      datasets: [
        {
          type: 'bar',
          label: 'Ingresos',
          data: ingresos,
          yAxisID: 'y',
          borderWidth: 0
        },
        {
          type: 'line',
          label: '% acumulado',
          data: acumuladoPct,
          yAxisID: 'y1',
          tension: 0.3,
          pointRadius: 2
        }
      ]
    },
    options: {
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true },
        tooltip: {
          callbacks: {
            label: (tt) => tt.datasetIndex === 0
              ? `Ingresos: ${formatMoney(tt.parsed.y)}`
              : `% acumulado: ${tt.parsed.y}%`
          }
        }
      },
      scales: {
        y:  { position: 'left', type: paretoLog ? 'logarithmic' : 'linear', grid: { display:false }, ticks: { callback: (v)=> formatMoney(v) } },
        y1: { position: 'right', grid: { display:false }, min: 0, max: 100, ticks: { callback: (v)=> v + '%' } },
        x:  { grid: { display:false } }
      }
    }
  });
}

(function enableSortableCategorias(){
  const table = document.getElementById('tbl-categorias');
  if (!table) return;
  const thead = table.querySelector('thead');
  const tbody = table.querySelector('tbody');
  if (!thead || !tbody) return;

  const parseNumber = (s) => {
    if (s == null) return NaN;
    const t = String(s).replace(/\./g,'').replace(/,/g,'.').replace(/[^\d.-]/g,'').trim();
    const n = parseFloat(t);
    return isNaN(n) ? NaN : n;
  };

  const getCellValue = (tr, idx) => tr.children[idx]?.textContent?.trim() || '';

  thead.addEventListener('click', (e) => {
    const th = e.target.closest('th[data-sort-key]');
    if (!th) return;
    const idx = Array.from(th.parentNode.children).indexOf(th);

    thead.querySelectorAll('th').forEach(h => h.classList.remove('sort-asc','sort-desc'));
    const asc = !th.classList.contains('sort-asc');
    th.classList.add(asc ? 'sort-asc' : 'sort-desc');

    const rows = Array.from(tbody.querySelectorAll('tr'));
    const numeric = ['Productos','Unidades','Ingresos','Precio','Rating']
      .includes(th.textContent.trim());

    rows.sort((a,b) => {
      const va = getCellValue(a, idx);
      const vb = getCellValue(b, idx);
      if (numeric) {
        const na = parseNumber(va);
        const nb = parseNumber(vb);
        return asc ? (na-nb) : (nb-na);
      }
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    });

    rows.forEach(r => tbody.appendChild(r));
  });
})();

function showTrendsSection(){
  const $trends = document.querySelector('#section-trends');
  const $list = document.querySelector('#section-products');
  if ($trends) $trends.hidden = false;
  if ($list) $list.hidden = true;

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

