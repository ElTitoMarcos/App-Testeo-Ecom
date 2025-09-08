import { fetchJson } from '/static/js/net.js';

const EC_COST_PER_1K_TOKENS = {
  'gpt-4o-mini': 0.15,
  'gpt-4o': 0.6,
  'gpt-4': 30,
  'gpt-3.5-turbo': 0.5
};
const EC_BA_MAX_BATCH = 100;
const STORAGE_KEY = 'ec-ba-batch-v1';

const btnBatch = document.getElementById('btn-ba-batch');
if (btnBatch) btnBatch.addEventListener('click', openBatchModal);

checkSavedBatch();

let state = null;

function checkSavedBatch() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {
    const st = JSON.parse(raw);
    if (st.pendingIds && st.pendingIds.length) {
      toast.info('Hay un lote BA pausado. ¿Reanudar?', {
        actionText: 'Reanudar',
        onAction: () => resumeBatch(st),
        actionText2: 'Descartar',
        onAction2: () => localStorage.removeItem(STORAGE_KEY),
        duration: 10000
      });
    }
  } catch (e) {
    localStorage.removeItem(STORAGE_KEY);
  }
}

function openBatchModal() {
  if (selection.size === 0) return;
  let ids = Array.from(selection);
  if (ids.length > EC_BA_MAX_BATCH) {
    toast.info(`Máximo ${EC_BA_MAX_BATCH} productos. Se tomarán los primeros ${EC_BA_MAX_BATCH}.`);
    ids = ids.slice(0, EC_BA_MAX_BATCH);
  }
  const products = ids.map(id => (window.products || []).find(p => String(p.id) === id)).filter(Boolean);
  const box = document.createElement('div');
  box.style.padding = '20px';
  box.innerHTML = `<h3>BA (GPT) Lote</h3><p>N seleccionados: ${products.length}</p>`;

  const modelLabel = document.createElement('label');
  modelLabel.textContent = 'Modelo: ';
  const modelSel = document.createElement('select');
  ['gpt-4o-mini', 'gpt-4o', 'gpt-4', 'gpt-3.5-turbo'].forEach(m => {
    const o = document.createElement('option');
    o.value = m; o.textContent = m;
    if (m === 'gpt-4o-mini') o.selected = true;
    modelSel.appendChild(o);
  });
  modelLabel.appendChild(modelSel);
  box.appendChild(modelLabel);

  const concLabel = document.createElement('label');
  concLabel.textContent = ' Concurrencia: ';
  const concSel = document.createElement('select');
  for (let i = 1; i <= 5; i++) {
    const o = document.createElement('option');
    o.value = i; o.textContent = i;
    if (i === 2) o.selected = true;
    concSel.appendChild(o);
  }
  concLabel.appendChild(concSel);
  box.appendChild(concLabel);

  const limitLabel = document.createElement('label');
  limitLabel.textContent = ' Límite ítems: ';
  const limitInput = document.createElement('input');
  limitInput.type = 'number';
  limitInput.min = 1;
  limitInput.max = products.length;
  limitInput.value = Math.min(products.length, 10);
  limitLabel.appendChild(limitInput);
  box.appendChild(limitLabel);

  const incLabel = document.createElement('label');
  const incCb = document.createElement('input');
  incCb.type = 'checkbox';
  incCb.checked = true;
  incLabel.appendChild(incCb);
  incLabel.appendChild(document.createTextNode(' Incluir imagen si hay URL'));
  box.appendChild(incLabel);

  const costLabel = document.createElement('label');
  costLabel.textContent = ' Límite coste (€): ';
  const costInput = document.createElement('input');
  costInput.type = 'number';
  costInput.min = '0';
  costInput.step = '0.01';
  costLabel.appendChild(costInput);
  box.appendChild(costLabel);

  const estDiv = document.createElement('div');
  estDiv.style.marginTop = '8px';
  box.appendChild(estDiv);

  function updateEstimate() {
    const model = modelSel.value;
    const n = Math.min(parseInt(limitInput.value) || 0, products.length);
    const items = products.slice(0, n);
    let total = 0;
    items.forEach(p => { total += estimateCost(p, model); });
    estDiv.textContent = `Coste estimado: €${total.toFixed(2)}`;
  }
  modelSel.onchange = updateEstimate;
  limitInput.oninput = updateEstimate;
  updateEstimate();

  const actions = document.createElement('div');
  actions.style.marginTop = '15px';
  actions.style.display = 'flex';
  actions.style.gap = '8px';
  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancelar';
  const runBtn = document.createElement('button');
  runBtn.textContent = 'Iniciar lote';
  actions.appendChild(cancelBtn);
  actions.appendChild(runBtn);
  box.appendChild(actions);

  const handle = window.modalManager.open(box, { returnFocus: btnBatch });
  cancelBtn.onclick = () => handle.close();
  runBtn.onclick = () => {
    handle.close();
    const limit = Math.min(parseInt(limitInput.value) || 0, products.length);
    const queue = products.slice(0, limit).map(p => ({
      product: p,
      original: {
        desire: p.desire,
        desire_magnitude: p.desire_magnitude,
        awareness_level: p.awareness_level,
        competition_level: p.competition_level
      },
      estimatedCost: estimateCost(p, modelSel.value)
    }));
    const config = {
      model: modelSel.value,
      concurrency: parseInt(concSel.value) || 1,
      includeImage: incCb.checked,
      costLimit: costInput.value ? parseFloat(costInput.value) : null
    };
    startBatch(queue, config);
  };
}

function estimateTokens(product) {
  return Math.ceil(JSON.stringify(product).length * 1.3 + 500);
}

function estimateCost(product, model) {
  const tokens = estimateTokens(product);
  const rate = EC_COST_PER_1K_TOKENS[model] || 0;
  return tokens / 1000 * rate;
}

function startBatch(queue, config) {
  if (!queue.length) return;
  state = {
    queue,
    total: queue.length,
    processed: 0,
    ok: 0,
    ko: 0,
    logs: [],
    done: [],
    failed: [],
    costSoFar: 0,
    config,
    start: Date.now(),
    paused: false,
    canceled: false
  };
  saveState();
  showProgressBar();
  updateUI();
  startWorkers();
  if (btnBatch) btnBatch.disabled = true;
}

function resumeBatch(saved) {
  const products = window.products || [];
  const queue = [];
  saved.pendingIds.forEach(id => {
    const p = products.find(pp => String(pp.id) === String(id));
    if (p) queue.push({
      product: p,
      original: {
        desire: p.desire,
        desire_magnitude: p.desire_magnitude,
        awareness_level: p.awareness_level,
        competition_level: p.competition_level
      },
      estimatedCost: estimateCost(p, saved.config.model)
    });
  });
  let costSoFar = 0;
  (saved.doneIds || []).forEach(id => {
    const p = products.find(pp => String(pp.id) === String(id));
    if (p) costSoFar += estimateCost(p, saved.config.model);
  });
  (saved.failedIds || []).forEach(f => {
    const p = products.find(pp => String(pp.id) === String(f.id));
    if (p) costSoFar += estimateCost(p, saved.config.model);
  });
  state = {
    queue,
    total: (saved.doneIds ? saved.doneIds.length : 0) + (saved.failedIds ? saved.failedIds.length : 0) + queue.length,
    processed: (saved.doneIds ? saved.doneIds.length : 0) + (saved.failedIds ? saved.failedIds.length : 0),
    ok: saved.doneIds ? saved.doneIds.length : 0,
    ko: saved.failedIds ? saved.failedIds.length : 0,
    logs: [],
    done: saved.doneIds || [],
    failed: saved.failedIds || [],
    costSoFar,
    config: saved.config,
    start: Date.now(),
    paused: false,
    canceled: false
  };
  showProgressBar();
  updateUI();
  startWorkers();
  if (btnBatch) btnBatch.disabled = true;
}

function showProgressBar() {
  if (document.getElementById('baBatchBar')) return;
  const bar = document.createElement('div');
  bar.id = 'baBatchBar';
  bar.style.position = 'fixed';
  bar.style.left = '0';
  bar.style.right = '0';
  bar.style.bottom = '0';
  bar.style.background = '#fff';
  bar.style.borderTop = '1px solid #ccc';
  bar.style.padding = '8px';
  bar.style.zIndex = '1000';
  bar.innerHTML = `<div id="baBatchStats"></div>
  <div style="height:8px;background:#ddd;margin:6px 0;"><div id="baBatchProg" style="height:100%;width:0%;background:#3b82f6;"></div></div>
  <div id="baBatchEta" style="font-size:12px;"></div>
  <div style="display:flex;gap:8px;margin-top:6px;">
    <button id="baBatchPause">Pausar</button>
    <button id="baBatchCancel">Cancelar</button>
    <button id="baBatchView">Ver resultados</button>
  </div>
  <details id="baBatchLast"><summary>Últimos 5</summary><ul></ul></details>`;
  document.body.appendChild(bar);
  state.statsEl = document.getElementById('baBatchStats');
  state.progEl = document.getElementById('baBatchProg');
  state.etaEl = document.getElementById('baBatchEta');
  state.pauseBtn = document.getElementById('baBatchPause');
  state.cancelBtn = document.getElementById('baBatchCancel');
  state.resultsBtn = document.getElementById('baBatchView');
  state.lastList = bar.querySelector('#baBatchLast ul');
  state.pauseBtn.onclick = togglePause;
  state.cancelBtn.onclick = cancelBatch;
  state.resultsBtn.onclick = showResultsPanel;
}

function updateUI() {
  if (!state) return;
  if (state.statsEl) state.statsEl.textContent = `Progreso: ${state.processed}/${state.total} (éxitos ${state.ok}, fallos ${state.ko})`;
  if (state.progEl) state.progEl.style.width = state.total ? ((state.processed / state.total) * 100).toFixed(1) + '%' : '0%';
  if (state.etaEl) {
    const elapsed = (Date.now() - state.start) / 1000;
    const rate = state.processed ? elapsed / state.processed : 0;
    const remain = state.total - state.processed;
    const eta = rate * remain;
    state.etaEl.textContent = `ETA: ${isFinite(eta) ? eta.toFixed(1) : '0'}s`;
  }
  updateLastUI();
  if (state.canceled && state.queue.length === 0 && state.processed === state.total) finishBatch();
  if (!state.canceled && state.queue.length === 0 && state.processed === state.total) finishBatch();
}

function togglePause() {
  if (!state) return;
  state.paused = !state.paused;
  if (state.pauseBtn) state.pauseBtn.textContent = state.paused ? 'Reanudar' : 'Pausar';
  if (!state.paused) startWorkers();
  saveState();
}

function cancelBatch() {
  if (!state) return;
  state.canceled = true;
  state.queue.length = 0;
  saveState();
  updateUI();
}

function addLog(id, status, message) {
  state.logs.unshift({ id, status, message });
  state.logs = state.logs.slice(0, 5);
  updateLastUI();
}

function updateLastUI() {
  if (!state || !state.lastList) return;
  state.lastList.innerHTML = '';
  state.logs.forEach(l => {
    const li = document.createElement('li');
    li.textContent = `${l.id} ${l.status}${l.message ? ' (' + l.message + ')' : ''}`;
    state.lastList.appendChild(li);
  });
}

async function worker() {
  while (true) {
    if (!state || state.canceled) return;
    if (state.paused) { await wait(500); continue; }
    const item = state.queue.shift();
    if (!item) return;
    if (state.config.costLimit && state.costSoFar + item.estimatedCost > state.config.costLimit) {
      state.paused = true;
      state.queue.unshift(item);
      saveState();
      updateUI();
      toast.info('Se alcanzará el límite de coste. ¿Continuar?', {
        actionText: 'Continuar',
        onAction: () => { state.config.costLimit = null; state.paused = false; startWorkers(); },
        actionText2: 'Cancelar',
        onAction2: () => cancelBatch(),
        duration: 10000
      });
      return;
    }
    await processItem(item);
    state.processed++;
    saveState();
    updateUI();
    await wait(200 + Math.random() * 300);
  }
}

function startWorkers() {
  const n = state ? state.config.concurrency : 0;
  for (let i = 0; i < n; i++) worker();
}

async function processItem(item) {
  const product = item.product;
  const payload = {
    id: product.id,
    name: product.name,
    category: product.category,
    price: product.price,
    rating: product.rating,
    units_sold: product.units_sold,
    revenue: product.revenue,
    conversion_rate: product.conversion_rate,
    launch_date: product.launch_date,
    date_range: product.date_range,
    image_url: state.config.includeImage ? product.image_url : null,
    desire: product.desire,
    desire_magnitude: product.desire_magnitude,
    awareness_level: product.awareness_level,
    competition_level: product.competition_level
  };
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const resp = await fetchJson('/api/ba/insights', {
        method: 'POST',
        body: JSON.stringify({ product: payload, model: state.config.model })
      });
      state.costSoFar += item.estimatedCost;
      applyUpdates(product, item, resp.grid_updates);
      state.ok++;
      state.done.push(product.id);
      addLog(product.id, 'ok');
      return;
    } catch (e) {
      if ((e.status === 429 || e.status === 503) && attempt < 2) {
        addLog(product.id, 'retry');
        const delay = Math.min(30000, Math.pow(2, attempt) * 1000 + Math.random() * 1000);
        await wait(delay);
        continue;
      }
      state.costSoFar += item.estimatedCost;
      state.ko++;
      const msg = e.message || (e.status ? 'Error ' + e.status : 'Error');
      state.failed.push({ id: product.id, message: msg });
      addLog(product.id, 'fallo', msg);
      return;
    }
  }
}

function applyUpdates(product, item, updates) {
  const keys = ['desire', 'desire_magnitude', 'awareness_level', 'competition_level'];
  const applied = {};
  keys.forEach(k => {
    const nv = updates[k];
    if (nv === undefined) return;
    if (product[k] !== item.original[k]) {
      toast.info(`Se detectaron cambios locales en ${k} del producto ${product.id}; ¿aplicar GPT?`, {
        actionText: 'Aplicar GPT',
        onAction: () => {
          product[k] = nv;
          renderTable();
          fetchJson(`/products/${product.id}`, { method: 'PUT', body: JSON.stringify({ [k]: nv }) }).catch(() => {});
        }
      });
    } else {
      product[k] = nv;
      applied[k] = nv;
    }
  });
  if (Object.keys(applied).length) {
    renderTable();
    fetchJson(`/products/${product.id}`, { method: 'PUT', body: JSON.stringify(applied) }).catch(() => {});
  }
}

function saveState() {
  if (!state) return;
  const data = {
    pendingIds: state.queue.map(it => it.product.id),
    doneIds: state.done,
    failedIds: state.failed,
    config: state.config,
    timestamp: Date.now()
  };
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); } catch (e) {}
}

function finishBatch() {
  localStorage.removeItem(STORAGE_KEY);
  if (state.pauseBtn) state.pauseBtn.disabled = true;
  if (state.cancelBtn) {
    state.cancelBtn.textContent = 'Cerrar';
    state.cancelBtn.onclick = () => {
      const bar = document.getElementById('baBatchBar');
      if (bar) document.body.removeChild(bar);
    };
  }
  if (btnBatch) btnBatch.disabled = selection.size === 0;
}

function showResultsPanel() {
  if (!state) return;
  const box = document.createElement('div');
  box.style.padding = '20px';
  box.style.maxWidth = '800px';
  const elapsed = ((Date.now() - state.start) / 1000).toFixed(1);
  box.innerHTML = `<h3>Resultados lote BA</h3><p>${state.ok} ok / ${state.ko} fallos de ${state.total} en ${elapsed}s</p>`;
  const doneTable = document.createElement('table');
  let html = '<thead><tr><th>ID</th><th>Awareness</th><th>Desire Mag</th><th>Competition</th><th>Desire</th></tr></thead><tbody>';
  state.done.forEach(id => {
    const p = (window.products || []).find(pp => String(pp.id) === String(id));
    if (p) html += `<tr><td>${p.id}</td><td>${p.awareness_level || ''}</td><td>${p.desire_magnitude || ''}</td><td>${p.competition_level || ''}</td><td>${(p.desire || '').slice(0,40)}</td></tr>`;
  });
  html += '</tbody>';
  doneTable.innerHTML = html;
  box.appendChild(doneTable);
  if (state.failed.length) {
    const failTable = document.createElement('table');
    let html2 = '<thead><tr><th>ID</th><th>Motivo</th></tr></thead><tbody>';
    state.failed.forEach(f => { html2 += `<tr><td>${f.id}</td><td>${f.message}</td></tr>`; });
    html2 += '</tbody>';
    failTable.innerHTML = html2;
    box.appendChild(failTable);
  }
  window.modalManager.open(box, { returnFocus: state.resultsBtn });
}

function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

export {}

