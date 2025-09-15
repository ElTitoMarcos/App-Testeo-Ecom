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

window.__charts = window.__charts || {};
let leftChart = window.__charts.leftChart || null;
let paretoChart = window.__charts.paretoChart || null;

const $desde = document.querySelector('#fecha-desde');
const $hasta = document.querySelector('#fecha-hasta');
const $btnAplicar = document.querySelector('#btn-aplicar-tendencias');
const $status = document.querySelector('#trends-status');

if ($btnAplicar) {
  $btnAplicar.addEventListener('click', (ev) => {
    ev.preventDefault();
    fetchTrends();
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

async function fetchTrends() {
  ensureDefaultDates();
  try {
    if ($status) $status.textContent = 'Cargando...';
    const fISO = $desde ? toISOFromDDMMYYYY($desde.value) : null;
    const tISO = $hasta ? toISOFromDDMMYYYY($hasta.value) : null;
    const url = new URL('/api/trends/summary', window.location.origin);
    if (fISO) url.searchParams.set('from', fISO);
    if (tISO) url.searchParams.set('to', tISO);
    const res = await fetch(url.toString(), { credentials: 'same-origin' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const json = await res.json();
    handleTrendsResponse(json);
  } catch (e) {
    (window.toast?.error || alert).call(window.toast || window, 'No se pudieron cargar las tendencias.');
  } finally {
    if ($status) $status.textContent = '';
  }
}

function handleTrendsResponse(summary) {
  if (!summary) return;
  const scope = resolveVisibleScope();
  if (scope.categoriesAgg.length) {
    renderTrends(scope.categoriesAgg, scope.allProducts);
    return;
  }
  const categoriesRaw = summary.categoriesAgg || summary.top_categories || summary.categories || [];
  renderTrends(categoriesRaw, getAllProductsSnapshot());
}

function getAllProductsSnapshot() {
  const arr = window.allProducts;
  if (Array.isArray(arr) && arr.length) return arr;
  const scope = resolveVisibleScope();
  return Array.isArray(scope.allProducts) ? scope.allProducts : [];
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

function renderTopCategoriesBar(categoriesAgg) {
  const canvas = document.getElementById('topCategoriesChart');
  if (!canvas) return;
  const rows = [...(Array.isArray(categoriesAgg) ? categoriesAgg : [])]
    .filter((x) => Number(x.revenue || 0) > 0)
    .sort((a, b) => Number(b.revenue || 0) - Number(a.revenue || 0))
    .slice(0, 10);
  const labels = rows.map((x) => x.path || x.category || x.name || '');
  const values = rows.map((x) => Number(x.revenue) || 0);

  if (leftChart) {
    leftChart.destroy();
  }

  leftChart = new Chart(canvas, {
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

  window.__charts.leftChart = leftChart;
}

function renderRightPareto(categoriesAgg) {
  const el = document.getElementById('paretoRevenueChart');
  if (!el) return;
  const ctx = el.getContext('2d');
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

  if (paretoChart) paretoChart.destroy();

  paretoChart = new Chart(ctx, {
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
        x: { ticks: { autoSkip: true, maxRotation: 0 } }
      }
    }
  });

  window.__charts.paretoChart = paretoChart;
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

// Llama a esta función desde tu flujo principal tras obtener datos
export function renderTrends(categoriesAgg, allProducts) {
  let normalized = normalizeCategories(Array.isArray(categoriesAgg) ? categoriesAgg : []);
  let productsSource = Array.isArray(allProducts) ? allProducts : null;

  if (!normalized.length || !productsSource?.length) {
    const scope = resolveVisibleScope();
    if (!normalized.length) {
      normalized = normalizeCategories(scope.categoriesAgg);
    }
    if (!productsSource || !productsSource.length) {
      productsSource = scope.allProducts;
    }
  }

  const products = normalizeProducts(productsSource || []);
  window.__latestTrendsData = { categoriesAgg: normalized, allProducts: products };
  renderTopCategoriesBar(normalized);
  renderRightPareto(normalized);
  fillTrendsTable(normalized);
}

// Toggle montado una sola vez
export function mountTrendsToggle(){
  if (window.__trendsToggleMounted) return;
  window.__trendsToggleMounted = true;

  document.addEventListener('click', (ev) => {
    const btn = ev.target.closest('[data-action="toggle-trends"]');
    if (!btn) return;

    const container = document.getElementById('section-trends');
    const sec1 = document.getElementById('trends');
    const sec2 = document.getElementById('trends-bottom');
    const opening = sec1?.hasAttribute('hidden');

    if (opening){
      container?.removeAttribute('hidden');
      sec1?.removeAttribute('hidden');
      sec2?.removeAttribute('hidden');
      ensureDefaultDates();

      // (Re)render con datos del ámbito visible
      const { categoriesAgg, allProducts } = resolveVisibleScope();
      window.__latestTrendsData = { categoriesAgg, allProducts };
      renderTopCategoriesBar(categoriesAgg);
      renderRightPareto(categoriesAgg);    // usa chartOptsStable
      fillTrendsTable(categoriesAgg);

      // fuerza cálculo de tamaño y pinta
      requestAnimationFrame(() => {
        paretoChart?.resize();
        leftChart?.resize?.();
      });
    }else{
      container?.setAttribute('hidden','');
      sec1?.setAttribute('hidden','');
      sec2?.setAttribute('hidden','');
    }
  }, { passive:true });
}

// === ÁMBITO DE DATOS: usar los productos actualmente visibles ===
function resolveVisibleScope(){
  // 1) Preferir el dataset que usa la tabla principal de productos
  // (ajusta nombres a lo que exista en el proyecto; fallbacks seguros)
  const visible = window.__visibleProducts
                || window.appState?.productsFiltered
                || window.appState?.currentView
                || window.appState?.products
                || window.__allProducts
                || [];

  // Recalcular agregados por categoría desde 'visible'
  const aggMap = new Map();
  for (const p of visible){
    const path = p.category_path || p.category || p.path || 'Sin categoría';
    const key = path;
    const row = aggMap.get(key) || { path:key, products:0, units:0, revenue:0, price:0, rating:0 };
    row.products += 1;
    row.units    += Number(p.units_sold || p.units || 0);
    row.revenue  += Number(p.revenue || 0);
    row.price    += Number(p.price || 0);
    row.rating   += Number(p.rating || 0);
    aggMap.set(key, row);
  }
  const categoriesAgg = [...aggMap.values()].map(r => ({
    ...r,
    price : r.products ? r.price  / r.products : 0,
    rating: r.products ? r.rating / r.products : 0
  }));

  return { categoriesAgg, allProducts: visible };
}

function formatMoney(v){
  v = Number(v||0);
  if (v >= 1e6) return '€ ' + (v/1e6).toFixed(2).replace('.', ',') + ' M';
  if (v >= 1e3) return '€ ' + (v/1e3).toFixed(1).replace('.', ',') + ' K';
  return '€ ' + v.toFixed(2).replace('.', ',');
}

mountTrendsToggle();

