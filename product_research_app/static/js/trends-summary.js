import { LoadingHelpers } from './loading.js';

const fmt = {
  money: formatMoney,
  percent: (p) => (p * 100).toFixed(1).replace('.', ',') + '%'
};

// Opciones comunes y estables para charts
const chartOptsStable = {
  responsive: true,
  maintainAspectRatio: false,
  resizeDelay: 200,
  animation: { duration: 0 }
};

const charts = (window.__charts = window.__charts || {});

const $desde = document.querySelector('#fecha-desde');
const $hasta = document.querySelector('#fecha-hasta');
const $btnAplicar = document.querySelector('#btn-aplicar-tendencias');
const $status = document.querySelector('#trends-status');

if ($btnAplicar) {
  $btnAplicar.addEventListener('click', (ev) => {
    ev.preventDefault();
    fetchTrends(ev.currentTarget);
  });
}

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

function ensureDefaultDates() {
  try {
    const today = new Date();
    const from = new Date(today);
    from.setDate(today.getDate() - 29);
    if ($desde && !$desde.value) $desde.value = formatDDMMYYYY(from);
    if ($hasta && !$hasta.value) $hasta.value = formatDDMMYYYY(today);
  } catch (_) {}
}

async function fetchTrends(btn) {
  ensureDefaultDates();
  const host = document.querySelector('#progress-slot-global');
  const tracker = LoadingHelpers.start('Actualizando tendencias', { host });
  try {
    if ($status) $status.textContent = 'Cargando...';
    tracker.step(0.05);
    const fISO = $desde ? toISOFromDDMMYYYY($desde.value) : null;
    const tISO = $hasta ? toISOFromDDMMYYYY($hasta.value) : null;
    const url = new URL('/api/trends/summary', window.location.origin);
    if (fISO) url.searchParams.set('from', fISO);
    if (tISO) url.searchParams.set('to', tISO);
    tracker.step(0.25);
    const res = await fetch(url.toString(), {
      credentials: 'same-origin',
      __hostEl: host,
      __skipLoadingHook: true
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const json = await res.json();
    tracker.step(0.5);
    handleTrendsResponse(json, tracker);
    tracker.step(0.95);
  } catch (e) {
    (window.toast?.error || alert).call(window.toast || window, 'No se pudieron cargar las tendencias.');
    tracker.step(1);
  } finally {
    if ($status) $status.textContent = '';
    tracker.done();
  }
}

function handleTrendsResponse(summary, tracker) {
  if (!summary) return;
  const scope = computeTrendsScope();
  if (scope.categoriesAgg.length) {
    applyTrendsScope(scope, tracker);
    return;
  }
  const categoriesRaw = summary.categoriesAgg || summary.top_categories || summary.categories || [];
  const allProductsRaw =
    summary.products || summary.items || summary.all_products || summary.allProducts || [];
  renderTrends(categoriesRaw, allProductsRaw, tracker);
}

function toNumber(value) {
  if (value == null || value === '') return 0;
  if (typeof value === 'number') return Number.isFinite(value) ? value : 0;
  if (typeof value === 'string') {
    const text = value
      .replace(/[€$]/g, '')
      .replace(/\s+/g, '')
      .replace(/\./g, '')
      .replace(/,/g, '.');
    const num = Number(text);
    return Number.isFinite(num) ? num : 0;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : 0;
}

function normalizeCategories(list) {
  if (!Array.isArray(list)) return [];
  return list.map((item) => {
    const path = item.path || item.category || item.name || '';
    const name = item.name || path;
    const revenue = toNumber(item.revenue ?? item.total_revenue ?? item.sum_revenue ?? item.value);
    const units = toNumber(item.units ?? item.total_units ?? item.sum_units ?? item.quantity);
    const productsRaw = item.products ?? item.products_count ?? item.unique_products ?? item.count;
    const products = Number.isFinite(Number(productsRaw)) ? Number(productsRaw) : 0;
    const price = toNumber(item.price ?? item.avg_price ?? item.average_price);
    const rating = toNumber(item.rating ?? item.avg_rating ?? item.average_rating);
    return {
      ...item,
      path,
      name,
      revenue,
      units,
      products,
      price,
      rating
    };
  });
}

function normalizeProducts(list) {
  if (!Array.isArray(list)) return [];
  return list.map((item) => {
    const name = item.name || item.title || item.product_name || '';
    const revenue = toNumber(item.revenue ?? item.total_revenue ?? item.sales ?? item.turnover);
    const units = toNumber(item.units ?? item.quantity ?? item.total_units ?? item.sum_units);
    const unitsSold = toNumber(item.units_sold ?? item.unitsSold ?? item.sold_units ?? item.sales_units ?? units);
    const price = toNumber(item.price ?? item.avg_price ?? item.average_price);
    const rating = toNumber(item.rating ?? item.avg_rating ?? item.average_rating);
    return {
      ...item,
      name,
      revenue,
      units,
      units_sold: unitsSold,
      price,
      rating
    };
  });
}

function getVisibleProducts(){
  const visible = window.__visibleProducts;
  if (Array.isArray(visible) && visible.length) return visible;
  const all = window.__allProducts;
  return Array.isArray(all) ? all : [];
}

function buildCategoriesAgg(products){
  const agg = new Map();
  for (const p of Array.isArray(products) ? products : []){
    const key = p.category_path || p.category || p.path || 'Sin categoría';
    const row = agg.get(key) || { path:key, products:0, units:0, revenue:0, price:0, rating:0 };
    row.products += 1;
    row.units    += Number(p.units_sold || p.units || 0);
    row.revenue  += Number(p.revenue || 0);
    row.price    += Number(p.price || 0);
    row.rating   += Number(p.rating || 0);
    agg.set(key, row);
  }
  return [...agg.values()].map((r) => ({
    ...r,
    price : r.products ? r.price  / r.products : 0,
    rating: r.products ? r.rating / r.products : 0
  }));
}

function computeTrendsScope(){
  const base = getVisibleProducts();
  const products = normalizeProducts(base);
  const categoriesAgg = normalizeCategories(buildCategoriesAgg(products));
  const scope = { allProducts: products, categoriesAgg };
  window.__latestTrendsData = scope;
  return scope;
}

function renderTopCategoriesBar(categoriesAgg) {
  const canvas = document.getElementById('topCategoriesChart');
  if (!canvas) return;
  const rows = [...(Array.isArray(categoriesAgg) ? categoriesAgg : [])]
    .filter((x) => Number(x.revenue || 0) > 0)
    .sort((a, b) => Number(b.revenue || 0) - Number(a.revenue || 0))
    .slice(0, 10);
  const labels = rows.map((x) => x.path || x.category || x.name || '');
  const values = rows.map((x) => Number(x.revenue) || 0);
  if (!charts.leftChart) {
    charts.leftChart = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            data: values,
            borderWidth: 0
          }
        ]
      },
      options: {
        ...chartOptsStable,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (tt) => `Ingresos: ${formatMoney(tt.parsed.x)}`
            }
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { callback: (v) => formatMoney(v) } },
          y: { grid: { display: false } }
        }
      }
    });
    return;
  }

  const chart = charts.leftChart;
  chart.data.labels = labels;
  if (chart.data.datasets?.[0]) {
    chart.data.datasets[0].data = values;
  } else {
    chart.data.datasets = [{ data: values, borderWidth: 0 }];
  }
  chart.update('none');
}

function renderRightPareto(categoriesAgg) {
  const el = document.getElementById('paretoRevenueChart');
  if (!el) return;
  const rows = [...(Array.isArray(categoriesAgg) ? categoriesAgg : [])]
    .filter((r) => Number(r.revenue || 0) > 0)
    .sort((a, b) => Number(b.revenue || 0) - Number(a.revenue || 0))
    .slice(0, 15);
  const total = rows.reduce((s, r) => s + Number(r.revenue || 0), 0) || 1;
  let acc = 0;
  const labels = [];
  const bars = [];
  const cumu = [];
  for (const r of rows) {
    labels.push(r.path || r.name || '');
    bars.push(Number(r.revenue || 0));
    acc += Number(r.revenue || 0);
    cumu.push((acc / total) * 100);
  }

  if (!charts.paretoChart) {
    charts.paretoChart = new Chart(el.getContext('2d'), {
      data: {
        labels,
        datasets: [
          { type: 'bar', label: 'Ingresos', data: bars, yAxisID: 'y', borderWidth: 0 },
          {
            type: 'line',
            label: '% acumulado',
            data: cumu,
            yAxisID: 'y1',
            tension: 0.25,
            pointRadius: 0,
            pointHitRadius: 6
          }
        ]
      },
      options: {
        ...chartOptsStable,
        plugins: {
          legend: { display: true },
          tooltip: {
            callbacks: {
              label: (c) =>
                c.dataset.type === 'line'
                  ? `% acumulado: ${c.formattedValue}%`
                  : `Ingresos: ${formatMoney(c.raw)}`
            }
          }
        },
        scales: {
          y: { beginAtZero: true, ticks: { callback: (v) => formatMoney(v) } },
          y1: {
            beginAtZero: true,
            position: 'right',
            grid: { drawOnChartArea: false },
            min: 0,
            max: 100,
            ticks: { callback: (v) => v + '%' }
          },
          x: {
            ticks: { display: false },
            grid: { display: false }
          }
        }
      }
    });
    return;
  }

  const chart = charts.paretoChart;
  chart.data.labels = labels;
  if (chart.data.datasets?.[0]) {
    chart.data.datasets[0].data = bars;
  } else {
    chart.data.datasets = chart.data.datasets || [];
    chart.data.datasets[0] = { type: 'bar', label: 'Ingresos', data: bars, yAxisID: 'y', borderWidth: 0 };
  }
  if (chart.data.datasets?.[1]) {
    chart.data.datasets[1].data = cumu;
  } else {
    chart.data.datasets[1] = {
      type: 'line',
      label: '% acumulado',
      data: cumu,
      yAxisID: 'y1',
      tension: 0.25,
      pointRadius: 0,
      pointHitRadius: 6
    };
  }
  chart.update('none');
}

// Ajusta fillTrendsTable para usar SOLO categoriesAgg (sin productos)
function fillTrendsTable(categoriesAgg) {
  const tbody = document.querySelector('#trendsTable tbody');
  if (!tbody) return;
  const frag = document.createDocumentFragment();

  categoriesAgg.forEach((c) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${c.path || c.name || ''}</td>
      <td>${c.products ?? ''}</td>
      <td>${Number(c.units || 0).toLocaleString('es-ES')}</td>
      <td>${fmt.money(Number(c.revenue || 0))}</td>
      <td>${Number(c.price || 0).toFixed(2).replace('.', ',')}</td>
      <td>${Number(c.rating || 0).toFixed(2).replace('.', ',')}</td>
    `;
    frag.appendChild(tr);
  });

  tbody.replaceChildren(frag);
  const thead = document.querySelector('#trendsTable thead');
  thead?.querySelectorAll('th[aria-sort]').forEach((th) => th.removeAttribute('aria-sort'));
}

function applyTrendsScope(scope, tracker){
  const categoriesAgg = normalizeCategories(scope?.categoriesAgg || []);
  const allProducts = normalizeProducts(scope?.allProducts || []);
  window.__latestTrendsData = { categoriesAgg, allProducts };
  tracker?.step(0.7);
  renderTopCategoriesBar(categoriesAgg);
  renderRightPareto(categoriesAgg);
  tracker?.step(0.85);
  fillTrendsTable(categoriesAgg);
}

// Llama a esta función desde tu flujo principal tras obtener datos
export function renderTrends(categoriesAgg, allProducts, tracker) {
  const fallback = computeTrendsScope();
  const categories = Array.isArray(categoriesAgg) && categoriesAgg.length
    ? categoriesAgg
    : fallback.categoriesAgg;
  const products = Array.isArray(allProducts) && allProducts.length
    ? allProducts
    : fallback.allProducts;
  applyTrendsScope({ categoriesAgg: categories, allProducts: products }, tracker);
  tracker?.step(0.92);
}

// Toggle montado una sola vez
export function mountTrendsToggle(){
  if (window.__trendsToggleMounted) return;
  window.__trendsToggleMounted = true;

  const openClose = () => {
    const container = document.getElementById('section-trends');
    const sec1 = document.getElementById('trends');
    const sec2 = document.getElementById('trends-bottom');
    if (!sec1 || !sec2) return;
    const opening = sec1.hasAttribute('hidden');

    if (opening) {
      container?.removeAttribute('hidden');
      sec1.removeAttribute('hidden');
      sec2.removeAttribute('hidden');
      ensureDefaultDates();

      const scope = computeTrendsScope();
      applyTrendsScope(scope);

      requestAnimationFrame(() => {
        charts.paretoChart?.resize();
        charts.leftChart?.resize?.();
      });
    } else {
      container?.setAttribute('hidden', '');
      sec1.setAttribute('hidden', '');
      sec2.setAttribute('hidden', '');
    }
  };

  document.addEventListener('click', (ev) => {
    const btn = ev.target.closest('[data-action="toggle-trends"]');
    if (btn) openClose();
  }, { passive: true });

  document.addEventListener('visible-products-changed', () => {
    const trendsSection = document.getElementById('trends');
    if (!trendsSection || trendsSection.hasAttribute('hidden')) return;
    const scope = computeTrendsScope();
    applyTrendsScope(scope);
    requestAnimationFrame(() => {
      charts.paretoChart?.resize();
      charts.leftChart?.resize?.();
    });
  });
}

function formatMoney(v){
  v = Number(v||0);
  if (v >= 1e6) return '€ ' + (v/1e6).toFixed(2).replace('.', ',') + ' M';
  if (v >= 1e3) return '€ ' + (v/1e3).toFixed(1).replace('.', ',') + ' K';
  return '€ ' + v.toFixed(2).replace('.', ',');
}

window.computeTrendsScope = computeTrendsScope;

mountTrendsToggle();
