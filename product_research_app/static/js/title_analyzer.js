import { fetchJson } from '/static/js/net.js';

const fileInput = document.getElementById('taFile');
const analyzeBtn = document.getElementById('taAnalyzeBtn');
const table = document.getElementById('taTable');
const tbody = table.querySelector('tbody');
const summaryDiv = document.getElementById('taSummary');

let detailDialog = document.getElementById('taDetailDialog');
if (!detailDialog) {
  detailDialog = document.createElement('dialog');
  detailDialog.id = 'taDetailDialog';
  const pre = document.createElement('pre');
  pre.id = 'taDetailText';
  const closeBtn = document.createElement('button');
  closeBtn.textContent = 'Cerrar';
  closeBtn.addEventListener('click', () => detailDialog.close());
  detailDialog.appendChild(pre);
  detailDialog.appendChild(closeBtn);
  document.body.appendChild(detailDialog);
}

analyzeBtn?.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) {
    if (window.toast) toast.error('Selecciona un archivo');
    return;
  }
  let resp;
  try {
    if (file.name.endsWith('.json')) {
      const text = await file.text();
      const data = JSON.parse(text);
      resp = await fetchJson('/api/analyze/titles', {
        method: 'POST',
        body: JSON.stringify(data)
      });
    } else {
      const form = new FormData();
      form.append('file', file);
      resp = await fetchJson('/api/analyze/titles', {
        method: 'POST',
        body: form
      });
    }
  } catch (err) {
    return;
  }
  const items = resp.items || [];
  renderTable(items);
  renderSummary(items);
});

function renderTable(items) {
  tbody.innerHTML = '';
  if (!items.length) {
    table.style.display = 'none';
    summaryDiv.style.display = 'none';
    return;
  }
  table.style.display = '';
  items.forEach(item => {
    const tr = document.createElement('tr');

    const tdTitle = document.createElement('td');
    tdTitle.textContent = item.title || '';
    tr.appendChild(tdTitle);

    const tdSignals = document.createElement('td');
    ['value','claims','materials','compat'].forEach(key => {
      (item.signals && item.signals[key] || []).forEach(val => {
        const span = document.createElement('span');
        span.className = 'chip';
        span.textContent = val;
        tdSignals.appendChild(span);
      });
    });
    tr.appendChild(tdSignals);

    const tdScore = document.createElement('td');
    const score = item.titleScore || 0;
    const badge = document.createElement('span');
    badge.className = 'badge ' + scoreClass(score);
    badge.textContent = score.toFixed(2);
    tdScore.appendChild(badge);
    if (item.flags?.seo_bloat) {
      const f = document.createElement('span');
      f.className = 'ta-flag';
      f.title = 'Título muy largo';
      f.textContent = '📏';
      tdScore.appendChild(f);
    }
    if (item.flags?.ip_risk) {
      const f = document.createElement('span');
      f.className = 'ta-flag';
      f.title = 'Riesgo de marca';
      f.textContent = '⚠️';
      tdScore.appendChild(f);
    }
    tr.appendChild(tdScore);

    const tdSummary = document.createElement('td');
    const summaryText = item.summary?.text || '';
    tdSummary.textContent = summaryText;
    tr.appendChild(tdSummary);

    const tdCopy = document.createElement('td');
    const btn = document.createElement('button');
    btn.textContent = 'Copiar resumen';
    btn.addEventListener('click', () => navigator.clipboard.writeText(summaryText));
    tdCopy.appendChild(btn);
    tr.appendChild(tdCopy);

    const tdDetail = document.createElement('td');
    const btnDetail = document.createElement('button');
    btnDetail.className = 'ta-detail-btn';
    btnDetail.textContent = '🔍';
    btnDetail.title = 'Ver análisis detallado';
    btnDetail.addEventListener('click', async () => {
      try {
        const resp = await fetchJson('/api/analyze/product_detail', {
          method: 'POST',
          body: JSON.stringify(item),
          headers: { 'Content-Type': 'application/json' }
        });
        const pre = document.getElementById('taDetailText');
        if (pre) pre.textContent = resp.detail || '';
        detailDialog.showModal();
      } catch (err) {
        if (window.toast) toast.error('No se pudo obtener el detalle');
      }
    });
    tdDetail.appendChild(btnDetail);
    tr.appendChild(tdDetail);

    tbody.appendChild(tr);
  });
}

function scoreClass(score){
  if (score >= 1) return 'score-green';
  if (score >= 0.5) return 'score-amber';
  return 'score-red';
}

function renderSummary(items){
  const termCounts = {};
  const prices = [];
  items.forEach(it => {
    ['value','claims','materials','compat'].forEach(k => {
      (it.signals && it.signals[k] || []).forEach(v => {
        termCounts[v] = (termCounts[v] || 0) + 1;
      });
    });
    if (typeof it.price === 'number') prices.push(it.price);
  });
  const topTerms = Object.entries(termCounts).sort((a,b) => b[1]-a[1]).slice(0,5);
  summaryDiv.innerHTML = '';
  const termsEl = document.createElement('div');
  termsEl.innerHTML = '<strong>Top términos:</strong> ' + (topTerms.map(([t,c]) => `${t} (${c})`).join(', ') || '—');
  summaryDiv.appendChild(termsEl);

  if (prices.length) {
    prices.sort((a,b)=>a-b);
    const q25 = quantile(prices,0.25);
    const q75 = quantile(prices,0.75);
    const med = quantile(prices,0.5);
    const stats = document.createElement('div');
    stats.innerHTML = `<strong>Precio Q25–Q75:</strong> ${q25.toFixed(2)} - ${q75.toFixed(2)} | <strong>Mediana:</strong> ${med.toFixed(2)}`;
    summaryDiv.appendChild(stats);
    const canvas = document.createElement('canvas');
    canvas.width = 400; canvas.height = 200;
    summaryDiv.appendChild(canvas);
    drawHist(prices, canvas);
  }
  summaryDiv.style.display = 'block';
}

function quantile(arr,q){
  const pos = (arr.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return arr[base + 1] !== undefined ? arr[base] + rest * (arr[base + 1] - arr[base]) : arr[base];
}

function drawHist(prices, canvas){
  const ctx = canvas.getContext('2d');
  const bins = 5;
  const min = prices[0];
  const max = prices[prices.length - 1];
  const step = (max - min) / bins || 1;
  const counts = new Array(bins).fill(0);
  prices.forEach(p => {
    let idx = Math.floor((p - min) / step);
    if (idx >= bins) idx = bins - 1;
    counts[idx]++;
  });
  const maxCount = Math.max(...counts);
  const w = canvas.width / bins;
  counts.forEach((c,i) => {
    const h = maxCount ? (c / maxCount) * (canvas.height - 20) : 0;
    ctx.fillStyle = '#0077cc';
    ctx.fillRect(i*w, canvas.height - h, w - 2, h);
  });
}
