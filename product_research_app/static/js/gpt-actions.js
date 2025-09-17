import { post } from './net.js';

const ENDPOINTS = {
  consulta: '/api/gpt/consulta',
  pesos: '/api/gpt/pesos',
  tendencias: '/api/gpt/tendencias',
  imputacion: '/api/gpt/imputacion',
  desire: '/api/gpt/desire'
};

const DEFAULT_PROMPTS = {
  consulta: 'Analiza el conjunto de productos y entrega hallazgos accionables.',
  pesos: 'Ajusta los pesos del winner score usando los agregados proporcionados.',
  tendencias: 'Realiza un análisis profundo de tendencias y oportunidades.',
  imputacion: 'Imputa los campos faltantes con valores plausibles.',
  desire: 'Resume el nivel de desire/comprabilidad de los productos.'
};

let lastHighlightedIds = new Set();
let lastWarnings = [];
let lastResponse = null;
let pendingImputations = null;

function safeCloneProducts(list) {
  if (!Array.isArray(list)) return [];
  if (typeof structuredClone === 'function') {
    try { return structuredClone(list); } catch (err) { /* continue */ }
  }
  try {
    return JSON.parse(JSON.stringify(list));
  } catch (err) {
    return list.map(item => {
      const copy = {};
      for (const key in item) copy[key] = item[key];
      return copy;
    });
  }
}

function getVisibleRowIds() {
  const rows = Array.from(document.querySelectorAll('#productTable tbody tr'));
  const ids = [];
  for (const row of rows) {
    if (!row || row.offsetParent === null || row.style.display === 'none') continue;
    const cb = row.querySelector('input.rowCheck');
    if (!cb || !cb.dataset.id) continue;
    ids.push(cb.dataset.id);
  }
  return ids;
}

function resolveGroupId(value) {
  if (value === undefined || value === null || value === '' || value === -1) return null;
  return String(value);
}

function resolveTimeWindow(value) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

export function collectContext({ includeProducts = true } = {}) {
  const products = includeProducts && Array.isArray(window.products) ? window.products : [];
  const context = {
    group_id: resolveGroupId(window.currentGroupFilter),
    time_window: resolveTimeWindow(window.currentTimeWindow),
    products: includeProducts ? safeCloneProducts(products) : []
  };
  const visible = getVisibleRowIds();
  if (visible.length) context.visible_ids = visible;
  return context;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function formatInline(text) {
  if (!text) return '';
  let out = escapeHtml(text);
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  return out;
}

function renderMarkdown(text) {
  if (!text) return '';
  const lines = String(text).split(/\r?\n/);
  const parts = [];
  let inList = false;
  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    if (!line) {
      if (inList) {
        parts.push('</ul>');
        inList = false;
      }
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      if (!inList) {
        parts.push('<ul>');
        inList = true;
      }
      parts.push(`<li>${formatInline(line.replace(/^[-*]\s+/, ''))}</li>`);
      continue;
    }
    if (inList) {
      parts.push('</ul>');
      inList = false;
    }
    if (/^#{1,6}\s+/.test(line)) {
      const level = Math.min(6, line.match(/^#+/)[0].length);
      parts.push(`<h${level}>${formatInline(line.replace(/^#{1,6}\s+/, ''))}</h${level}>`);
    } else {
      parts.push(`<p>${formatInline(line)}</p>`);
    }
  }
  if (inList) parts.push('</ul>');
  return parts.join('\n');
}

function extractHighlightIds(data) {
  if (!data) return [];
  const ids = new Set();
  const refs = data.refs;
  if (Array.isArray(refs)) {
    for (const entry of refs) {
      const id = entry && (entry.id || entry.product_id || entry.productId);
      if (id !== undefined && id !== null && id !== '') ids.add(String(id));
    }
  } else if (refs && typeof refs === 'object') {
    if (Array.isArray(refs.product_ids)) {
      refs.product_ids.forEach(id => {
        if (id !== undefined && id !== null && id !== '') ids.add(String(id));
      });
    }
    if (Array.isArray(refs.ids)) {
      refs.ids.forEach(id => {
        if (id !== undefined && id !== null && id !== '') ids.add(String(id));
      });
    }
  }
  return Array.from(ids);
}

function highlightRows(ids) {
  const table = document.getElementById('productTable');
  if (!table) return;
  for (const prev of lastHighlightedIds) {
    const row = table.querySelector(`input.rowCheck[data-id="${CSS.escape(prev)}"]`);
    const tr = row ? row.closest('tr') : null;
    if (tr) tr.classList.remove('gpt-highlight');
  }
  lastHighlightedIds = new Set();
  ids.forEach(id => {
    const row = table.querySelector(`input.rowCheck[data-id="${CSS.escape(id)}"]`);
    const tr = row ? row.closest('tr') : null;
    if (tr) {
      tr.classList.add('gpt-highlight');
      lastHighlightedIds.add(id);
    }
  });
}

function ensurePanel() {
  const panel = document.getElementById('gptPanel');
  if (panel) panel.classList.remove('hidden');
  return panel;
}

function updateWarnings(panel, response) {
  const wrap = panel?.querySelector('#gptWarnings');
  if (!wrap) return;
  const list = wrap.querySelector('ul');
  if (!list) return;
  list.innerHTML = '';
  const warnings = [];
  if (response && response.ok === false) {
    warnings.push('El modelo no devolvió un resultado completo.');
  }
  const arr = Array.isArray(response?.warnings) ? response.warnings : [];
  for (const item of arr) {
    if (!item) continue;
    warnings.push(String(item));
  }
  lastWarnings = warnings;
  if (!warnings.length) {
    wrap.classList.add('hidden');
    return;
  }
  warnings.forEach(text => {
    const li = document.createElement('li');
    li.textContent = text;
    list.appendChild(li);
  });
  wrap.classList.remove('hidden');
}

function updateRisks(panel, data) {
  const container = panel?.querySelector('#gptRisks');
  if (!container) return;
  const list = container.querySelector('ul');
  if (!list) return;
  list.innerHTML = '';
  const riesgosRaw = data?.riesgos;
  let riesgos = [];
  if (Array.isArray(riesgosRaw)) riesgos = riesgosRaw;
  else if (typeof riesgosRaw === 'string' && riesgosRaw.trim()) riesgos = [riesgosRaw.trim()];
  if (!riesgos.length) {
    container.classList.add('hidden');
    return;
  }
  riesgos.forEach(item => {
    const li = document.createElement('li');
    li.textContent = String(item);
    list.appendChild(li);
  });
  container.classList.remove('hidden');
}

function renderWeightsBlock(container, data) {
  if (!data?.weights) return;
  const block = document.createElement('section');
  block.className = 'gpt-block gpt-weights';
  const title = document.createElement('h4');
  title.textContent = 'Pesos sugeridos';
  block.appendChild(title);
  const list = document.createElement('ul');
  const entries = Object.entries(data.weights).sort((a, b) => (Number(b[1]) || 0) - (Number(a[1]) || 0));
  entries.forEach(([key, value]) => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="gpt-metric">${formatInline(key)}</span><span class="gpt-value">${Number(value).toFixed(2)}</span>`;
    list.appendChild(li);
  });
  block.appendChild(list);
  if (Array.isArray(data.weights_order) && data.weights_order.length) {
    const order = document.createElement('p');
    order.className = 'gpt-order';
    order.innerHTML = `<strong>Orden sugerido:</strong> ${data.weights_order.join(', ')}`;
    block.appendChild(order);
  }
  container.appendChild(block);
  document.dispatchEvent(new CustomEvent('gpt-weights-suggestion', { detail: data }));
}

function renderDesireBlock(container, results) {
  const entries = Object.entries(results || {});
  if (!entries.length) return;
  const block = document.createElement('section');
  block.className = 'gpt-block gpt-desire';
  block.innerHTML = '<h4>Resumen de desire</h4>';
  const list = document.createElement('div');
  list.className = 'gpt-grid';
  entries.forEach(([id, info]) => {
    const card = document.createElement('article');
    card.className = 'gpt-item';
    const header = document.createElement('header');
    header.innerHTML = `<span class="gpt-id">#${formatInline(id)}</span>`;
    card.appendChild(header);
    if (info && typeof info === 'object') {
      if (info.summary || info.text) {
        const p = document.createElement('p');
        p.innerHTML = formatInline(info.summary || info.text);
        card.appendChild(p);
      }
      if (Array.isArray(info.bullets) && info.bullets.length) {
        const ul = document.createElement('ul');
        info.bullets.forEach(line => {
          const li = document.createElement('li');
          li.textContent = String(line);
          ul.appendChild(li);
        });
        card.appendChild(ul);
      }
      if (info.desire) {
        const desireTag = document.createElement('div');
        desireTag.className = 'gpt-tag';
        desireTag.textContent = String(info.desire);
        card.appendChild(desireTag);
      }
    } else {
      const p = document.createElement('p');
      p.textContent = String(info);
      card.appendChild(p);
    }
    list.appendChild(card);
  });
  block.appendChild(list);
  container.appendChild(block);
}

function renderGenericResults(container, results) {
  const entries = Object.entries(results || {});
  if (!entries.length) return;
  const block = document.createElement('section');
  block.className = 'gpt-block gpt-results';
  block.innerHTML = '<h4>Resultados</h4>';
  const table = document.createElement('table');
  table.className = 'gpt-table';
  const tbody = document.createElement('tbody');
  entries.forEach(([id, detail]) => {
    const tr = document.createElement('tr');
    const tdId = document.createElement('td');
    tdId.innerHTML = formatInline(id);
    const tdDetail = document.createElement('td');
    if (detail && typeof detail === 'object') {
      const lines = [];
      for (const [k, v] of Object.entries(detail)) {
        if (v === undefined || v === null || v === '') continue;
        lines.push(`<strong>${formatInline(k)}:</strong> ${formatInline(String(v))}`);
      }
      tdDetail.innerHTML = lines.length ? lines.join('<br>') : '';
    } else {
      tdDetail.innerHTML = formatInline(String(detail));
    }
    tr.appendChild(tdId);
    tr.appendChild(tdDetail);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  block.appendChild(table);
  container.appendChild(block);
}

function renderImputationBlock(container, imputed) {
  const entries = Object.entries(imputed || {});
  if (!entries.length) return;
  pendingImputations = imputed;
  const block = document.createElement('section');
  block.className = 'gpt-block gpt-imputed';
  block.innerHTML = '<h4>Imputaciones sugeridas</h4>';
  const list = document.createElement('div');
  list.className = 'gpt-table gpt-imputed-list';
  entries.forEach(([id, fields]) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'gpt-imputed-item';
    const title = document.createElement('div');
    title.className = 'gpt-id';
    title.textContent = `#${id}`;
    wrapper.appendChild(title);
    const ul = document.createElement('ul');
    for (const [k, v] of Object.entries(fields || {})) {
      const li = document.createElement('li');
      li.innerHTML = `<strong>${formatInline(k)}:</strong> ${formatInline(String(v))}`;
      ul.appendChild(li);
    }
    wrapper.appendChild(ul);
    list.appendChild(wrapper);
  });
  block.appendChild(list);
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'gpt-apply-btn';
  btn.textContent = 'Aplicar imputaciones';
  btn.addEventListener('click', () => applyImputations(btn));
  block.appendChild(btn);
  container.appendChild(block);
}

async function applyImputations(button) {
  if (!pendingImputations) return;
  const entries = Object.entries(pendingImputations);
  if (!entries.length) return;
  let okCount = 0;
  const failures = [];
  const prev = button ? button.textContent : null;
  if (button) {
    button.disabled = true;
    button.textContent = 'Aplicando…';
  }
  for (const [id, fields] of entries) {
    const numId = Number(id);
    if (!Number.isFinite(numId)) {
      failures.push(`ID ${id} inválido`);
      continue;
    }
    try {
      const payload = Object.assign({}, fields, { source: 'gpt-imputacion' });
      const res = await fetch(`/products/${numId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || res.statusText || 'Error desconocido');
      }
      okCount += 1;
    } catch (err) {
      console.error('applyImputations', err);
      failures.push(`ID ${id}: ${err.message}`);
    }
  }
  if (button) {
    button.disabled = failures.length === 0;
    button.textContent = failures.length === 0 ? 'Imputaciones aplicadas' : prev || 'Aplicar imputaciones';
  }
  if (okCount) {
    if (typeof toast !== 'undefined') toast.success(`Imputaciones aplicadas: ${okCount}`);
    if (typeof window.fetchProducts === 'function') {
      try { window.fetchProducts(); } catch (err) { /* ignore */ }
    }
  }
  if (failures.length) {
    if (typeof toast !== 'undefined') toast.error(`Fallos en ${failures.length} imputaciones`);
    lastWarnings = lastWarnings.concat(failures);
    showGptLog();
    const wrap = document.getElementById('gptWarnings');
    if (wrap) {
      const list = wrap.querySelector('ul');
      if (list) {
        failures.forEach(f => {
          const li = document.createElement('li');
          li.textContent = f;
          list.appendChild(li);
        });
      }
      wrap.classList.remove('hidden');
    }
  } else {
    pendingImputations = null;
  }
}

function renderDataBlocks(task, data, container) {
  container.innerHTML = '';
  pendingImputations = null;
  if (!data) return;
  const cleanData = Object.assign({}, data);
  delete cleanData.prompt_version;
  const weights = cleanData.weights;
  if (weights) {
    renderWeightsBlock(container, cleanData);
  }
  const imputed = cleanData.imputed || (task === 'imputacion' ? cleanData.results : null);
  if (imputed) {
    renderImputationBlock(container, imputed);
  }
  if (task === 'desire' && cleanData.results) {
    renderDesireBlock(container, cleanData.results);
  } else if (cleanData.results && !imputed) {
    renderGenericResults(container, cleanData.results);
  }
}

function updateMeta(panel, response) {
  const chip = panel?.querySelector('#gptChunkChip');
  const meta = response?.meta;
  if (chip) {
    if (meta?.chunks > 1) {
      chip.textContent = 'Procesado por lotes';
      chip.classList.remove('hidden');
    } else {
      chip.classList.add('hidden');
    }
  }
  const metaBox = panel?.querySelector('#gptMeta');
  if (!metaBox) return;
  const model = response?.model || meta?.model;
  const tokens = meta?.estimated_tokens;
  const calls = meta?.calls;
  const lines = [];
  if (model) lines.push(`<strong>Modelo:</strong> ${formatInline(model)}`);
  if (calls) lines.push(`<strong>Llamadas:</strong> ${calls}`);
  if (tokens) lines.push(`<strong>Tokens aprox.:</strong> ${tokens}`);
  if (!lines.length) {
    metaBox.classList.add('hidden');
  } else {
    metaBox.innerHTML = lines.join(' · ');
    metaBox.classList.remove('hidden');
  }
}

export function displayGptResponse(task, response) {
  const panel = ensurePanel();
  if (!panel) return;
  const textEl = panel.querySelector('#gptText');
  if (textEl) {
    if (response?.text) {
      textEl.classList.remove('muted');
      textEl.innerHTML = renderMarkdown(response.text);
    } else {
      textEl.classList.add('muted');
      textEl.innerHTML = '<p>Sin texto recibido.</p>';
    }
  }
  const highlightIds = extractHighlightIds(response?.data);
  highlightRows(highlightIds);
  updateWarnings(panel, response);
  updateRisks(panel, response?.data || null);
  const dataContainer = panel.querySelector('#gptDataBlocks');
  if (dataContainer) {
    renderDataBlocks(task, response?.data || null, dataContainer);
  }
  updateMeta(panel, response);
}

export function notifyOutcome(response) {
  const warnings = Array.isArray(response?.warnings) ? response.warnings : [];
  if (response?.ok === false) {
    if (typeof toast !== 'undefined') {
      toast.info('La IA devolvió avisos. Revisa el log.', {
        actionText: 'Ver log',
        onAction: () => showGptLog()
      });
    }
  } else if (warnings.length) {
    if (typeof toast !== 'undefined') {
      toast.info('La respuesta incluye advertencias.', {
        actionText: 'Ver log',
        onAction: () => showGptLog()
      });
    }
  }
}

export async function executeGptTask(task, {
  promptText,
  params = {},
  button = null,
  context = null,
  busyText = 'Procesando…'
} = {}) {
  const endpoint = ENDPOINTS[task];
  if (!endpoint) throw new Error(`Tarea GPT desconocida: ${task}`);
  const payload = {
    prompt_text: typeof promptText === 'string' && promptText.trim() ? promptText : (DEFAULT_PROMPTS[task] || ''),
    context: context && typeof context === 'object' ? context : collectContext(),
    params: params && typeof params === 'object' ? params : {}
  };
  let prevText = null;
  if (button) {
    prevText = button.textContent;
    button.disabled = true;
    if (busyText) button.textContent = busyText;
  }
  try {
    const response = await post(endpoint, payload, 60000);
    lastResponse = { task, response };
    displayGptResponse(task, response);
    notifyOutcome(response);
    return response;
  } finally {
    if (button) {
      button.disabled = false;
      if (prevText !== null) button.textContent = prevText;
    }
  }
}

export function showGptLog() {
  const panel = ensurePanel();
  if (!panel) return;
  const wrap = panel.querySelector('#gptWarnings');
  if (wrap) {
    wrap.classList.remove('hidden');
    wrap.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

export function getLastResponse() {
  return lastResponse;
}
