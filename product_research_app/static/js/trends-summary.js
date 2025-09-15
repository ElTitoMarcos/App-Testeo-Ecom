const fmt = {
  money: (v) => {
    if (!isFinite(v)) return v;
    if (v >= 1e6) return '€ ' + (v / 1e6).toFixed(2).replace('.', ',') + ' M';
    if (v >= 1e3) return '€ ' + (v / 1e3).toFixed(1).replace('.', ',') + ' K';
    return '€ ' + v.toFixed(2).replace('.', ',');
  },
  percent: (p) => (p * 100).toFixed(1).replace('.', ',') + '%'
};

const chartOptsStable = {
  responsive: true,
  maintainAspectRatio: false,
  resizeDelay: 200,
  animation: { duration: 0 }
};

let topCategoriesChart = null;
let paretoChart = null;

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
  const categoriesRaw = summary.categoriesAgg || summary.top_categories || summary.categories || [];
  renderTrends(categoriesRaw, getAllProductsSnapshot());
}

function getAllProductsSnapshot() {
  const arr = window.allProducts;
  return Array.isArray(arr) ? arr : [];
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

function fmtMoney(v) {
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
    maximumFractionDigits
  });
  return `${formatted}${suffix}`.trim();
}

function renderTopCategoriesBar(categoriesAgg) {
  const canvas = document.getElementById('topCategoriesChart');
  if (!canvas) return;
  const top = (Array.isArray(categoriesAgg) ? categoriesAgg : []).slice(0, 10);
  const labels = top.map((x) => x.path || x.category || x.name || '');
  const values = top.map((x) => Number(x.revenue) || 0);

  if (topCategoriesChart) {
    topCategoriesChart.destroy();
    topCategoriesChart = null;
  }

  topCategoriesChart = new Chart(canvas, {
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
            label: (tt) => `Ingresos: ${fmtMoney(tt.parsed.x)}`
          }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { callback: (v) => fmtMoney(v) } },
        y: { grid: { display: false } }
      }
    }
  });
}

// Devuelve las top N categorías por ingresos con acumulado
function buildParetoData(categories, N = 15) {
  const rows = categories
    .map((c) => ({ name: c.path || c.name, revenue: Number(c.revenue || 0) }))
    .filter((r) => r.revenue > 0)
    .sort((a, b) => b.revenue - a.revenue)
    .slice(0, N);

  const total = rows.reduce((s, r) => s + r.revenue, 0) || 1;
  let acc = 0;
  const labels = [];
  const bars = [];
  const cumu = [];
  rows.forEach((r) => {
    labels.push(r.name);
    bars.push(r.revenue);
    acc += r.revenue;
    cumu.push((acc / total) * 100);
  });
  return { labels, bars, cumu };
}

function renderRightPareto(categoriesAgg) {
  const el = document.getElementById('paretoRevenueChart');
  if (!el) return;
  const ctx = el.getContext('2d');
  const { labels, bars, cumu } = buildParetoData(Array.isArray(categoriesAgg) ? categoriesAgg : [], 15);

  if (paretoChart) paretoChart.destroy();

  paretoChart = new Chart(ctx, {
    data: {
      labels,
      datasets: [
        {
          type: 'bar',
          label: 'Ingresos',
          data: bars,
          yAxisID: 'y',
          borderWidth: 0
        },
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
            label: (ctx) => {
              if (ctx.dataset.type === 'line') return '% acumulado: ' + ctx.formattedValue + '%';
              return 'Ingresos: ' + fmt.money(ctx.raw);
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            callback: (v) => fmt.money(v)
          }
        },
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
  const normalized = normalizeCategories(Array.isArray(categoriesAgg) ? categoriesAgg : []);
  const productsSource = Array.isArray(allProducts) ? allProducts : getAllProductsSnapshot();
  const products = normalizeProducts(productsSource);
  window.__latestTrendsData = { categoriesAgg: normalized, allProducts: products };
  renderTopCategoriesBar(normalized);
  renderRightPareto(normalized);
  fillTrendsTable(normalized);
}

export function mountTrendsToggle() {
  if (window.__trendsToggleMounted) return;
  window.__trendsToggleMounted = true;

  document.addEventListener(
    'click',
    (ev) => {
      const btn = ev.target.closest('[data-action="toggle-trends"]');
      if (!btn) return;

      const container = document.getElementById('section-trends');
      const sec1 = document.getElementById('trends');
      const sec2 = document.getElementById('trends-bottom');
      const opening = sec1 ? sec1.hasAttribute('hidden') : true;

      if (opening) {
        container?.removeAttribute('hidden');
        sec1?.removeAttribute('hidden');
        sec2?.removeAttribute('hidden');
        ensureDefaultDates();

        if (window.__latestTrendsData) {
          const data = window.__latestTrendsData.categoriesAgg || [];
          if (!topCategoriesChart) {
            renderTopCategoriesBar(data);
          }
          if (!paretoChart) {
            renderRightPareto(data);
          }
          fillTrendsTable(data);
        } else {
          fetchTrends();
        }

        requestAnimationFrame(() => {
          paretoChart?.resize();
          topCategoriesChart?.resize?.();
        });

        sec1?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } else {
        container?.setAttribute('hidden', '');
        sec1?.setAttribute('hidden', '');
        sec2?.setAttribute('hidden', '');
      }
    },
    { passive: true }
  );
}

mountTrendsToggle();

