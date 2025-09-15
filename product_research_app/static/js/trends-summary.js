import { fmtInt, fmtPrice, fmtFloat2 } from './format.js';

let priceIncomeChart;

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

function fmtMoney(v){
  const n = Number(v);
  if (!Number.isFinite(n) || n === 0) return '0';
  const abs = Math.abs(n);
  let divisor = 1;
  let suffix = '';
  if (abs >= 1e6) {
    divisor = 1e6;
    suffix = ' M';
  } else if (abs >= 1e3) {
    divisor = 1e3;
    suffix = ' K';
  }
  const scaled = n / divisor;
  let minimumFractionDigits = 0;
  let maximumFractionDigits = 0;
  if (divisor > 1) {
    minimumFractionDigits = 2;
    maximumFractionDigits = 2;
  } else if (Math.abs(scaled) < 10) {
    maximumFractionDigits = 2;
  } else if (Math.abs(scaled) < 100) {
    maximumFractionDigits = 1;
  }
  const formatted = scaled.toLocaleString('es-ES', {
    minimumFractionDigits,
    maximumFractionDigits,
  });
  return `${formatted}${suffix}`.trim();
}

function buildScatterData(allProducts){
  if (!Array.isArray(allProducts)) return [];
  const parseValue = (value) => {
    if (value == null) return null;
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value : null;
    }
    const text = String(value)
      .replace(/[€$]/g, '')
      .replace(/\s+/g, '')
      .replace(/\./g, '')
      .replace(/,/g, '.');
    const num = Number(text);
    return Number.isFinite(num) ? num : null;
  };
  return allProducts
    .map(item => {
      if (!item) return null;
      const priceRaw = item.price ?? item.avg_price ?? (item.extras && (item.extras['Avg. Unit Price($)'] ?? item.extras['Avg Unit Price($)'] ?? item.extras['Avg. Unit Price'] ?? item.extras.price));
      const revenueRaw = item.revenue ?? (item.extras && (item.extras['Revenue($)'] ?? item.extras['Revenue'] ?? item.extras.revenue));
      const price = parseValue(priceRaw);
      const revenue = parseValue(revenueRaw);
      if (!Number.isFinite(price) || !Number.isFinite(revenue) || price <= 0 || revenue <= 0) {
        return null;
      }
      const name = item.name || item.path || item.category || '';
      return { x: price, y: revenue, _name: name };
    })
    .filter(Boolean);
}

const $desde = document.querySelector('#fecha-desde');
const $hasta = document.querySelector('#fecha-hasta');
const $btnAplicar = document.querySelector('#btn-aplicar-tendencias');
let currentData = null;

if ($btnAplicar) {
  $btnAplicar.addEventListener('click', function(ev){
    ev.preventDefault();
    if (typeof fetchTrends === 'function') fetchTrends();
  });
}

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
  const products = Array.isArray(window.allProducts) ? window.allProducts : [];
  renderPriceIncomeScatter(products);
}

function renderCategoriasTable(data){
  const tbody = document.querySelector('#trendsTable tbody');
  if(!tbody) return;
  const rows = [...(data.top_categories || data.categories || [])];
  let html = '';
  rows.forEach(c => {
    const productos = c.products_count || c.products || c.unique_products || 0;
    const unidades = c.units || 0;
    const ingresos = c.revenue || 0;
    const precio = c.avg_price || 0;
    const rating = c.avg_rating || 0;
    const path = c.path || c.category || '';
    html += `<tr>`
      + `<td>${path}</td>`
      + `<td>${fmtInt(productos)}</td>`
      + `<td>${fmtInt(unidades)}</td>`
      + `<td>€ ${fmtMoney(ingresos)}</td>`
      + `<td>€ ${fmtPrice(precio)}</td>`
      + `<td>${fmtFloat2(rating)}</td>`
      + `</tr>`;
  });
  tbody.innerHTML = html;
  const thead = document.querySelector('#trendsTable thead');
  thead?.querySelectorAll('th[aria-sort]').forEach(th => th.removeAttribute('aria-sort'));
}

function renderTopCategoriesBar(data) {
  const top = [...(data.top_categories || data.categories || [])].slice(0, 10);
  const labels = top.map(x => x.path || x.category);
  const values = top.map(x => x.revenue);
  const ctx = document.getElementById('topCategoriesChart');
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
            label: (tt) => `Ingresos: ${fmtMoney(tt.parsed.x)}`
          }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { callback: (v)=> fmtMoney(v) } },
        y: { grid: { display: false } }
      }
    }
  });
}

function renderPriceIncomeScatter(products){
  const canvas = document.getElementById('priceIncomeScatter');
  if (!canvas) return;
  const points = buildScatterData(products);
  if (priceIncomeChart) {
    priceIncomeChart.destroy();
    priceIncomeChart = null;
  }
  if (!points.length) {
    return;
  }

  priceIncomeChart = new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [
        {
          data: points,
          label: '',
          pointBackgroundColor: '#6c8cff',
          pointBorderColor: '#6c8cff',
          pointRadius: 2.5,
          pointHoverRadius: 6,
          pointHitRadius: 8,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: 8 },
      plugins: {
        legend: { display: false },
        tooltip: {
          displayColors: false,
          callbacks: {
            title(items) {
              const raw = items?.[0]?.raw || {};
              const price = Number(raw.x);
              const revenue = Number(raw.y);
              return `€ ${fmtMoney(price)} · ${fmtMoney(revenue)}`;
            },
            label(ctx) {
              const raw = ctx.raw || {};
              return raw._name || '';
            }
          }
        }
      },
      scales: {
        x: {
          type: 'logarithmic',
          title: { display: true, text: 'Precio' },
          ticks: {
            callback(value) {
              const num = Number(value);
              return num > 0 ? `€ ${fmtMoney(num)}` : '';
            }
          },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          type: 'logarithmic',
          title: { display: true, text: 'Ingresos' },
          ticks: {
            callback(value) {
              const num = Number(value);
              return num > 0 ? `€ ${fmtMoney(num)}` : '';
            }
          },
          grid: { color: 'rgba(255,255,255,0.08)' }
        }
      }
    }
  });
}

const scheduleScatterUpdate = typeof queueMicrotask === 'function'
  ? queueMicrotask
  : (cb) => Promise.resolve().then(cb);

try {
  const initial = Array.isArray(window.allProducts) ? window.allProducts : [];
  let allProductsValue = initial;
  Object.defineProperty(window, 'allProducts', {
    configurable: true,
    get() {
      return allProductsValue;
    },
    set(value) {
      allProductsValue = value;
      scheduleScatterUpdate(() => {
        const arr = Array.isArray(value) ? value : [];
        renderPriceIncomeScatter(arr);
      });
    }
  });
  if (initial.length) {
    scheduleScatterUpdate(() => renderPriceIncomeScatter(initial));
  }
} catch (err) {
  // ignore if property cannot be redefined
}

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

  const firstChart = document.querySelector('#top-left, #topCategoriesChart');
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
