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
const echarts = window.echarts;
let topCategoriesChart = null;
let paretoChart = null;
let chartsResizeBound = false;

function bindChartsResize() {
  if (chartsResizeBound) return;
  window.addEventListener('resize', () => {
    if (topCategoriesChart) topCategoriesChart.resize();
    if (paretoChart) paretoChart.resize();
  });
  chartsResizeBound = true;
}

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
  const tbody = document.querySelector('#trendTable tbody');
  if (!tbody) return;
  const rows = [...(data.top_categories || data.categories || [])];
  const toNumber = (value) => {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
  };
  let html = '';
  rows.forEach(c => {
    const categoria = c.path || c.category || '';
    const productos = toNumber(c.products_count ?? c.products ?? c.unique_products ?? 0);
    const unidades = toNumber(c.units ?? 0);
    const ingresos = toNumber(c.revenue ?? 0);
    const precio = toNumber(c.avg_price ?? 0);
    const rating = toNumber(c.avg_rating ?? 0);
    html += `<tr>`
      + `<td>${categoria}</td>`
      + `<td data-raw="${productos}">${fmtInt(productos)}</td>`
      + `<td data-raw="${unidades}">${fmtInt(unidades)}</td>`
      + `<td data-raw="${ingresos}">${formatMoney(ingresos)}</td>`
      + `<td data-raw="${precio}">${fmtPrice(precio)}</td>`
      + `<td data-raw="${rating}">${fmtFloat2(rating)}</td>`
      + `</tr>`;
  });
  tbody.innerHTML = html;
  const table = document.getElementById('trendTable');
  if (table) {
    table.querySelectorAll('th.sortable .sort-caret').forEach(el => {
      el.textContent = '↕';
    });
    table.querySelectorAll('th.sortable').forEach(th => {
      if (typeof th._resetSort === 'function') th._resetSort();
    });
  }
}

function renderTopCategoriesBar(data) {
  if (!echarts) return;
  const top = [...(data.top_categories || data.categories || [])].slice(0, 10);
  const labels = top.map(x => x.path || x.category);
  const values = top.map(x => Number(x.revenue) || 0);
  const chartDom = document.getElementById('chart-left');
  if (!chartDom) return;

  if (!topCategoriesChart) {
    topCategoriesChart = echarts.init(chartDom, null, { renderer: 'canvas' });
  }

  const option = {
    backgroundColor: 'transparent',
    grid: { top: 30, right: 30, bottom: 20, left: 140, containLabel: true },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params) => {
        const item = params && params[0];
        if (!item) return '';
        return `${item.name}<br>${item.marker} Ingresos: ${formatMoney(item.value)}`;
      }
    },
    xAxis: {
      type: 'value',
      splitLine: { show: false },
      axisLabel: {
        fontSize: 12,
        formatter: (value) => formatMoney(value)
      }
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: labels,
      axisLabel: { interval: 0, fontSize: 12 },
      axisTick: { show: false }
    },
    series: [
      {
        name: 'Ingresos',
        type: 'bar',
        barWidth: '55%',
        data: values,
        itemStyle: { borderRadius: [0, 6, 6, 0] },
        emphasis: { focus: 'series' }
      }
    ]
  };

  topCategoriesChart.setOption(option, true);
  bindChartsResize();
}

function renderPareto(data) {
  if (!data || !echarts) return;
  const src = [...(data.top_categories || data.categories || [])];
  src.sort((a, b) => (Number(b.revenue) || 0) - (Number(a.revenue) || 0));
  const top = src.slice(0, 10);

  const labels = top.map(x => x.path || x.category);
  const ingresos = top.map(x => Number(x.revenue) || 0);
  const total = ingresos.reduce((sum, value) => sum + value, 0) || 1;
  let acc = 0;
  const acumuladoPct = ingresos.map(v => {
    acc += v;
    const pct = (acc / total) * 100;
    return Number.isFinite(pct) ? +pct.toFixed(1) : 0;
  });

  const paretoDom = document.getElementById('chart-right');
  if (!paretoDom) return;

  if (!paretoChart) {
    paretoChart = echarts.init(paretoDom, null, { renderer: 'canvas' });
  }

  const barData = ingresos.map(v => (paretoLog && v <= 0 ? null : v));
  const yAxisLeft = {
    type: paretoLog ? 'log' : 'value',
    name: 'Ingresos',
    axisLabel: {
      fontSize: 12,
      formatter: (value) => formatMoney(value)
    }
  };
  if (!paretoLog) yAxisLeft.min = 0;

  const paretoOption = {
    backgroundColor: 'transparent',
    grid: { top: 50, right: 40, bottom: 90, left: 60, containLabel: true },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params) => {
        if (!Array.isArray(params) || params.length === 0) return '';
        const lines = [params[0].name];
        params.forEach(item => {
          if (item.seriesName === 'Ingresos') {
            lines.push(`${item.marker} ${item.seriesName}: ${formatMoney(item.value)}`);
          } else {
            const pct = Number(item.value) || 0;
            lines.push(`${item.marker} ${item.seriesName}: ${pct.toLocaleString('es-ES', { maximumFractionDigits: 1 })}%`);
          }
        });
        return lines.join('<br>');
      }
    },
    legend: { top: 8, textStyle: { fontSize: 12 } },
    dataZoom: [
      { type: 'slider', xAxisIndex: 0, height: 16, bottom: 50 },
      { type: 'inside', xAxisIndex: 0 }
    ],
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { interval: 0, rotate: 28, fontSize: 11, margin: 12 },
      axisTick: { alignWithLabel: true }
    },
    yAxis: [
      yAxisLeft,
      {
        type: 'value',
        name: '% acumulado',
        min: 0,
        max: 100,
        position: 'right',
        axisLabel: { formatter: '{value} %', fontSize: 12 }
      }
    ],
    series: [
      { name: 'Ingresos', type: 'bar', barWidth: '55%', data: barData, emphasis: { focus: 'series' } },
      { name: '% acumulado', type: 'line', yAxisIndex: 1, smooth: true, symbolSize: 6, lineStyle: { width: 3 }, data: acumuladoPct }
    ]
  };

  paretoChart.setOption(paretoOption, true);
  bindChartsResize();
}

// BEGIN: TABLE SORT
(function makeTableSortable() {
  const table = document.getElementById('trendTable');
  if (!table) return;

  const tbody = table.querySelector('tbody');
  const getCell = (row, idx) => row.children[idx];
  const getValue = (cell) => cell?.getAttribute('data-raw') ?? cell?.innerText?.trim() ?? '';

  const toComparable = (v) => {
    const s = String(v).replace(/\./g, '').replace(',', '.'); // 1.234,56 -> 1234.56
    const n = Number(s);
    return Number.isFinite(n) ? n : String(v).toLowerCase();
  };

  table.querySelectorAll('th.sortable').forEach((th, idx) => {
    let asc = true;
    th.addEventListener('click', () => {
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => {
        const va = toComparable(getValue(getCell(a, idx)));
        const vb = toComparable(getValue(getCell(b, idx)));
        if (typeof va === 'number' && typeof vb === 'number') return asc ? va - vb : vb - va;
        return asc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
      });
      table.querySelectorAll('th.sortable .sort-caret').forEach(el => el.textContent = '↕');
      const caret = th.querySelector('.sort-caret'); if (caret) caret.textContent = asc ? '↑' : '↓';
      rows.forEach(r => tbody.appendChild(r));
      asc = !asc;
    });
    th._resetSort = () => { asc = true; };
  });
})();
// END: TABLE SORT

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

  const firstChart = document.querySelector('#chart-left, #card-top-categories');
  if (firstChart && typeof firstChart.scrollIntoView === 'function') {
    firstChart.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

document.addEventListener('click', function(e){
  const btn = e.target.closest('#btnVerTendencias, #btn-ver-tendencias, .btn-ver-tendencias, [data-action="show-trends"]');
  if (!btn) return;
  e.preventDefault();
  showTrendsSection();
});
export {};
