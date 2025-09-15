import { fmtInt, fmtPrice, fmtFloat2 } from './format.js';

const echarts = window.echarts;
const toast = window.toast;

const state = {
  categories: [],
  granularity: '',
  loaded: false,
};

let loadPromise = null;

const btnVerTendencias = document.getElementById('btnVerTendencias');
const tendenciasPanel = document.getElementById('tendenciasPanel');
const trendsSection = document.getElementById('section-trends');

if (trendsSection?.hidden && tendenciasPanel && !tendenciasPanel.classList.contains('hidden')) {
  tendenciasPanel.classList.add('hidden');
}

function ensureTrendSectionVisible(shouldShow) {
  if (!trendsSection) return;
  trendsSection.hidden = !shouldShow;
}

const initiallyVisible = Boolean(tendenciasPanel && !tendenciasPanel.classList.contains('hidden') && !(trendsSection?.hidden ?? false));
if (btnVerTendencias) {
  btnVerTendencias.setAttribute('aria-controls', 'tendenciasPanel');
  btnVerTendencias.classList.toggle('active', initiallyVisible);
  btnVerTendencias.setAttribute('aria-expanded', initiallyVisible.toString());
}
ensureTrendSectionVisible(initiallyVisible);

(function attachTrendsToggle() {
  const run = () => {
    document.addEventListener('click', async (event) => {
      const trigger = event.target.closest('#btnVerTendencias, [data-action="toggle-trends"]');
      if (!trigger) return;
      event.preventDefault();

      const panel = document.getElementById('tendenciasPanel');
      if (!panel) {
        console.warn('[Trends] No se encontró #tendenciasPanel');
        return;
      }

      const hidden = panel.classList.toggle('hidden');

      document.querySelectorAll('#btnVerTendencias, [data-action="toggle-trends"]').forEach((btn) => {
        btn.classList.toggle('active', !hidden);
        if (btn.id === 'btnVerTendencias') {
          btn.setAttribute('aria-expanded', (!hidden).toString());
        }
      });

      ensureTrendSectionVisible(!hidden);

      if (!hidden) {
        if (typeof ensureTrendsData === 'function') {
          try {
            await ensureTrendsData();
          } catch (err) {
            console.debug('[Trends] ensureTrendsData:', err);
          }
        }
        try {
          window.leftChart && window.leftChart.resize();
          window.rightChart && window.rightChart.resize();
        } catch (err) {
          console.debug('[Trends] resize charts:', err);
        }
      }
    });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
  } else {
    run();
  }
})();

function ensureMetricChips(anchorSelector, metrics, onChange) {
  const anchor = document.querySelector(anchorSelector);
  if (!anchor) return;
  let bar = anchor.previousElementSibling;
  if (!bar || !bar.classList?.contains('metric-chips')) {
    bar = document.createElement('div');
    bar.className = 'metric-chips';
    Object.assign(bar.style, { display: 'flex', gap: '6px', margin: '0 0 6px 0' });
    anchor.parentNode?.insertBefore(bar, anchor);
  }
  bar.innerHTML = '';
  metrics.forEach((metric, index) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = metric[0].toUpperCase() + metric.slice(1);
    btn.className = 'chip';
    btn.dataset.metric = metric;
    btn.addEventListener('click', () => {
      bar.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      onChange(metric);
    });
    if (index === 0) btn.classList.add('active');
    bar.appendChild(btn);
  });
}

function initLeftChart(dataByCategory) {
  if (!echarts) return;
  const dom = document.getElementById('chart-left');
  if (!dom) return;
  const chart = echarts.getInstanceByDom(dom) || echarts.init(dom, null, { renderer: 'canvas' });
  window.leftChart = chart;

  const truncate = (s, n = 42) => (s?.length ?? 0) > n ? `${s.slice(0, n - 1)}…` : (s || '');
  let currentMetric = 'ingresos';

  const metricLabels = {
    ingresos: 'Ingresos',
    unidades: 'Unidades',
    precio: 'Precio',
    rating: 'Rating',
  };

  const metricFormatters = {
    ingresos: (v) => fmtInt(v),
    unidades: (v) => fmtInt(v),
    precio: (v) => fmtPrice(v),
    rating: (v) => fmtFloat2(v),
  };

  const getSeries = () => dataByCategory
    .map(row => ({
      name: row.categoria,
      full: row.categoria,
      value: Number(row[currentMetric]) || 0,
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 20);

  function render() {
    const rows = getSeries();
    const labels = rows.map(r => truncate(r.name));
    const values = rows.map(r => r.value);
    const formatter = metricFormatters[currentMetric] ?? (v => v);

    chart.setOption({
      backgroundColor: 'transparent',
      grid: { top: 40, right: 20, bottom: 10, left: 260, containLabel: false },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (points) => {
          const item = points?.[0];
          if (!item) return '';
          const row = rows[item.dataIndex];
          return `${row.full}<br/>${metricLabels[currentMetric]}: ${formatter(values[item.dataIndex])}`;
        },
      },
      xAxis: {
        type: 'value',
        axisLabel: { fontSize: 11, formatter: (value) => formatter(value) },
      },
      yAxis: {
        type: 'category',
        data: labels,
        axisLabel: { fontSize: 11, margin: 8 },
        axisTick: { show: false },
      },
      series: [
        {
          type: 'bar',
          data: values,
          barWidth: 14,
          emphasis: { focus: 'series' },
        },
      ],
    }, true);
  }

  ensureMetricChips('#chart-left', ['ingresos', 'unidades', 'precio', 'rating'], (metric) => {
    currentMetric = metric;
    render();
  });

  render();
  bindChartResize(dom, chart);
}

function initRightChart(points) {
  if (!echarts) return;
  const dom = document.getElementById('chart-right');
  if (!dom) return;
  const chart = echarts.getInstanceByDom(dom) || echarts.init(dom, null, { renderer: 'canvas' });
  window.rightChart = chart;

  const scatterData = points
    .map(point => ({
      categoria: point.categoria,
      precio: Number(point.precio) || 0,
      ingresos: Number(point.ingresos) || 0,
    }))
    .filter(point => Number.isFinite(point.precio) && Number.isFinite(point.ingresos));

  const option = {
    backgroundColor: 'transparent',
    grid: { top: 50, right: 20, bottom: 70, left: 60, containLabel: true },
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const data = params.data;
        if (!data) return '';
        return `${data.categoria}<br/>Precio: ${fmtPrice(data.precio)}<br/>Ingresos: ${fmtInt(data.ingresos)}`;
      },
    },
    xAxis: { type: 'value', name: 'Precio', axisLabel: { fontSize: 11, formatter: (value) => fmtPrice(value) } },
    yAxis: { type: 'value', name: 'Ingresos', axisLabel: { fontSize: 11, formatter: (value) => fmtInt(value) } },
    dataZoom: [
      { type: 'slider', xAxisIndex: 0, height: 18, bottom: 18 },
      { type: 'inside', xAxisIndex: 0 },
    ],
    series: [
      {
        type: 'scatter',
        symbolSize: 10,
        data: scatterData.map(item => ({ value: [item.precio, item.ingresos], ...item })),
        emphasis: { focus: 'series' },
      },
    ],
  };

  chart.setOption(option, true);
  document.querySelector('#temporalidad-helper')?.classList.add('visible');
  bindChartResize(dom, chart);
}

function bindChartResize(dom, chart) {
  if (!dom || !chart) return;
  if (dom.__resizeHandler) return;
  const handler = () => chart.resize();
  dom.__resizeHandler = handler;
  window.addEventListener('resize', handler);
}

function toNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function adaptCategories(list) {
  if (!Array.isArray(list)) return [];
  return list.map(item => {
    const categoria = item.path ?? item.category ?? '';
    const productos = toNumber(item.products_count ?? item.products ?? item.unique_products);
    const unidades = toNumber(item.units);
    const ingresos = toNumber(item.revenue);
    const precio = toNumber(item.avg_price ?? item.price);
    const rating = toNumber(item.avg_rating ?? item.rating);
    return { categoria, productos, unidades, ingresos, precio, rating };
  }).sort((a, b) => b.ingresos - a.ingresos);
}

function renderTrendTable(rows) {
  const table = document.getElementById('trendTable');
  if (!table) return;
  const tbody = table.querySelector('tbody');
  if (!tbody) return;

  tbody.textContent = '';
  const frag = document.createDocumentFragment();

  rows.forEach(row => {
    const tr = document.createElement('tr');

    const tdCategoria = document.createElement('td');
    tdCategoria.textContent = row.categoria;
    if (row.categoria?.length > 60) tdCategoria.title = row.categoria;
    tr.appendChild(tdCategoria);

    const tdProductos = document.createElement('td');
    tdProductos.textContent = fmtInt(row.productos);
    tdProductos.setAttribute('data-raw', String(row.productos));
    tr.appendChild(tdProductos);

    const tdUnidades = document.createElement('td');
    tdUnidades.textContent = fmtInt(row.unidades);
    tdUnidades.setAttribute('data-raw', String(row.unidades));
    tr.appendChild(tdUnidades);

    const tdIngresos = document.createElement('td');
    tdIngresos.textContent = fmtInt(row.ingresos);
    tdIngresos.setAttribute('data-raw', String(row.ingresos));
    tr.appendChild(tdIngresos);

    const tdPrecio = document.createElement('td');
    tdPrecio.textContent = fmtPrice(row.precio);
    tdPrecio.setAttribute('data-raw', String(row.precio));
    tr.appendChild(tdPrecio);

    const tdRating = document.createElement('td');
    tdRating.textContent = fmtFloat2(row.rating);
    tdRating.setAttribute('data-raw', String(row.rating));
    tr.appendChild(tdRating);

    frag.appendChild(tr);
  });

  tbody.appendChild(frag);

  table.querySelectorAll('th.sortable').forEach(th => {
    th.querySelector('.sort-caret')?.textContent = '↕';
    th.setAttribute('aria-sort', 'none');
    if (typeof th._resetSort === 'function') th._resetSort();
  });
}

function toSortKey(value) {
  if (typeof value === 'number') return value;
  const raw = String(value ?? '').trim();
  if (!raw) return '';
  const normalized = raw.replace(/\s+/g, '').replace(/\./g, '').replace(',', '.');
  const num = Number(normalized);
  return Number.isFinite(num) ? num : raw.toLowerCase();
}

(function makeTableSortable() {
  const table = document.getElementById('trendTable');
  if (!table) return;
  const tbody = table.querySelector('tbody');
  if (!tbody) return;

  table.querySelectorAll('th.sortable').forEach((th, index) => {
    let asc = true;
    th.setAttribute('aria-sort', 'none');
    th._resetSort = () => {
      asc = true;
      th.setAttribute('aria-sort', 'none');
    };

    th.addEventListener('click', () => {
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const direction = asc ? 1 : -1;

      rows.sort((a, b) => {
        const cellA = a.children[index];
        const cellB = b.children[index];
        const valA = toSortKey(cellA?.getAttribute('data-raw') ?? cellA?.textContent ?? '');
        const valB = toSortKey(cellB?.getAttribute('data-raw') ?? cellB?.textContent ?? '');
        if (typeof valA === 'number' && typeof valB === 'number') {
          return direction * (valA - valB);
        }
        return direction * String(valA).localeCompare(String(valB));
      });

      table.querySelectorAll('th.sortable').forEach(other => {
        if (other === th) return;
        other.querySelector('.sort-caret')?.textContent = '↕';
        other.setAttribute('aria-sort', 'none');
        if (typeof other._resetSort === 'function') other._resetSort();
      });

      th.querySelector('.sort-caret')?.textContent = asc ? '↑' : '↓';
      th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');

      rows.forEach(row => tbody.appendChild(row));
      asc = !asc;
    });
  });
})();

function formatTemporalDate(isoDate) {
  if (!isoDate) return '';
  try {
    const date = new Date(isoDate);
    if (Number.isNaN(date.getTime())) return '';
    const dd = String(date.getDate()).padStart(2, '0');
    const mm = String(date.getMonth() + 1).padStart(2, '0');
    const yyyy = date.getFullYear();
    return `${dd}/${mm}/${yyyy}`;
  } catch (err) {
    return '';
  }
}

function updateTemporalidad(summary) {
  const helper = document.getElementById('temporalidad-helper');
  if (!helper) return;

  const timeseries = Array.isArray(summary?.timeseries) ? summary.timeseries : [];
  const first = timeseries[0]?.date;
  const last = timeseries[timeseries.length - 1]?.date;
  const rangeText = first && last ? `${formatTemporalDate(first)} → ${formatTemporalDate(last)}` : '';
  const granularity = summary?.granularity;
  const granularityLabel = granularity === 'week' ? 'Semanal' : granularity === 'day' ? 'Diaria' : '';

  const parts = [];
  if (rangeText) parts.push(rangeText);
  if (granularityLabel) parts.push(`Granularidad: ${granularityLabel}`);
  helper.textContent = parts.join(' · ') || 'Temporalidad';
  helper.classList.add('visible');
}

async function fetchAndRenderTrends() {
  const response = await fetch('/api/trends/summary', { credentials: 'same-origin' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const summary = await response.json();
  const categories = adaptCategories(summary.categories ?? summary.top_categories ?? []);
  state.categories = categories;
  state.granularity = summary.granularity ?? '';
  state.loaded = true;

  renderTrendTable(categories);
  initLeftChart(categories);
  initRightChart(categories);
  updateTemporalidad(summary);
  return categories;
}

async function ensureTrendsData() {
  if (state.loaded) return state.categories;
  if (!loadPromise) {
    loadPromise = fetchAndRenderTrends().catch((err) => {
      state.loaded = false;
      const message = 'No se pudieron cargar las tendencias.';
      if (toast?.error) toast.error(message); else alert(message);
      throw err;
    }).finally(() => {
      loadPromise = null;
    });
  }
  return loadPromise;
}

if (tendenciasPanel && !tendenciasPanel.classList.contains('hidden') && !(trendsSection?.hidden ?? false)) {
  ensureTrendsData().catch(() => {});
}

export {};
