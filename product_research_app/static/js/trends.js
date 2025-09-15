const hasDataTables = () => {
  const table = window.productsTable;
  return table && typeof table.rows === 'function' && typeof table.rows().data === 'function';
};

function getVisibleProductIds() {
  if (hasDataTables()) {
    try {
      return window.productsTable
        .rows({ filter: 'applied' })
        .data()
        .toArray()
        .map((row) => row.id);
    } catch (error) {
      console.warn('No se pudieron leer los IDs desde DataTables:', error);
    }
  }

  return Array.from(document.querySelectorAll('#productTable tbody tr[data-id]'))
    .filter((tr) => tr.offsetParent !== null)
    .map((tr) => tr.getAttribute('data-id'));
}

function toBulletedText(items) {
  if (!items) return '- Sin datos';
  const list = Array.isArray(items) ? items : [items];
  const filtered = list
    .map((value) => (value == null ? '' : String(value).trim()))
    .filter((value) => value.length > 0);
  if (!filtered.length) return '- Sin datos';
  return filtered.map((value) => `- ${value}`).join('\n');
}

let leftChart;
let rightChart;

function renderTrendsCharts(data) {
  if (typeof Chart === 'undefined') return;
  const leftCanvas = document.getElementById('chartLeft');
  const rightCanvas = document.getElementById('chartRight');
  if (!leftCanvas || !rightCanvas) return;

  const leftCtx = leftCanvas.getContext('2d');
  const rightCtx = rightCanvas.getContext('2d');

  const revenueByCategory = data?.revenue_by_category || { labels: [], values: [] };
  const trendOverTime = data?.trend_over_time || { labels: [], values: [] };

  if (leftChart) leftChart.destroy();
  if (rightChart) rightChart.destroy();

  leftChart = new Chart(leftCtx, {
    type: 'bar',
    data: {
      labels: Array.isArray(revenueByCategory.labels) ? revenueByCategory.labels : [],
      datasets: [
        {
          label: 'Ingresos',
          data: Array.isArray(revenueByCategory.values) ? revenueByCategory.values : [],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
    },
  });

  rightChart = new Chart(rightCtx, {
    type: 'line',
    data: {
      labels: Array.isArray(trendOverTime.labels) ? trendOverTime.labels : [],
      datasets: [
        {
          label: 'Tendencia',
          data: Array.isArray(trendOverTime.values) ? trendOverTime.values : [],
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
    },
  });

  setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
}

async function generateTrends() {
  const visibleIds = getVisibleProductIds();
  const payload = { ids: visibleIds };

  try {
    const response = await fetch('/api/trends', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    const panel = document.getElementById('trendsPanel');
    panel?.classList.remove('hidden');

    renderTrendsCharts(data);

    const insights = document.getElementById('insightsBox');
    if (insights) {
      const sections = [];
      if (data?.top_categories) {
        sections.push(`Top categorías por ingresos:\n${toBulletedText(data.top_categories)}`);
      }
      if (data?.top_products) {
        sections.push(`Productos top:\n${toBulletedText(data.top_products)}`);
      }
      if (data?.notes) {
        sections.push(`Notas:\n${toBulletedText(data.notes)}`);
      }
      insights.textContent = sections.length ? sections.join('\n\n') : '- Sin datos';
    }
  } catch (error) {
    console.error('Error generando tendencias', error);
    if (window.toast?.error) {
      window.toast.error('No se pudieron generar las tendencias.');
    }
  }
}

const trendsButton = document.getElementById('btnTrends');
trendsButton?.addEventListener('click', () => {
  const panel = document.getElementById('trendsPanel');
  if (panel?.classList.contains('hidden')) {
    panel.classList.remove('hidden');
    setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
  }
  generateTrends();
});

window.addEventListener('trends:refresh', () => {
  const panel = document.getElementById('trendsPanel');
  if (panel && panel.classList.contains('hidden')) return;
  generateTrends();
});

window.generateTrends = generateTrends;
