import { subscribe as subscribeSSE } from './sse.js';

const EC_BATCH_SIZE = 10;
const EC_MODEL = "gpt-4o-mini-2024-07-18";

function getAllFilteredRows() {
  if (typeof window.getAllFilteredRows === 'function') {
    try {
      return window.getAllFilteredRows();
    } catch {
      return [];
    }
  }
  return Array.isArray(window.products) ? window.products.slice() : [];
}

function isEditing(pid, field) {
  const active = document.activeElement;
  if (!active) return false;
  const tr = active.closest('tr');
  if (!tr) return false;
  const cb = tr.querySelector('input.rowCheck');
  if (!cb || cb.dataset.id !== String(pid)) return false;
  const td = active.closest('td[data-key]');
  if (!td) return false;
  return td.dataset.key === field;
}

function applyUpdates(product, updates, opts = {}) {
  const persist = opts.persist !== false;
  const applied = {};
  const row = document.querySelector(`input.rowCheck[data-id="${product.id}"]`)?.closest('tr');
  const map = {
    desire: 'td.ec-col-desire input',
    desire_magnitude: 'td.ec-col-desire-mag select',
    awareness_level: 'td.ec-col-awareness select',
    competition_level: 'td.ec-col-competition select'
  };
  Object.keys(map).forEach(k => {
    const nv = updates[k];
    if (nv === undefined || isEditing(product.id, k)) return;
    product[k] = nv;
    const el = row ? row.querySelector(map[k]) : null;
    if (el) {
      if (el.tagName === 'INPUT') el.value = nv || '';
      else el.value = nv || '';
    }
    applied[k] = nv;
  });
  if (persist && Object.keys(applied).length) {
    fetch(`/products/${product.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(applied)
    }).catch(() => {});
    if (window.ecAutoFitColumns && window.gridRoot) ecAutoFitColumns(gridRoot);
  }
  return applied;
}

function chunkArray(arr, size) {
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

async function processBatch(items) {
  const res = await fetch('/api/ia/batch-columns', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EC_MODEL, items })
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { const err = await res.json(); if (err.error) msg = err.error; } catch {}
    throw new Error(msg);
  }
  const data = await res.json();
  let ok = 0;
  let ko = 0;
  const okMap = data.ok || {};
  const koMap = data.ko || {};
  Object.keys(okMap).forEach(id => {
    const product = (window.products || []).find(p => String(p.id) === String(id));
    if (product) {
      applyUpdates(product, okMap[id]);
      ok++;
    } else {
      ko++;
    }
  });
  ko += Object.keys(koMap).length;
  return { ok, ko };
}

async function recalcDesireVisible(opts = {}) {
  const rows = getAllFilteredRows();
  if (!rows.length) {
    if (!opts.silent) toast.info('No hay productos');
    return;
  }
  const ids = rows.map(p => Number(p.id)).filter(id => Number.isInteger(id));
  try {
    const res = await fetch('/api/recalc-desire-all', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scope: 'filtered', ids })
    });
    if (!res.ok) {
      let msg = res.statusText || 'Error';
      try {
        const err = await res.json();
        if (err && err.error) msg = err.error;
      } catch {}
      throw new Error(msg);
    }
    const data = await res.json();
    if (!opts.silent) {
      const processed = Number(data.processed || 0);
      toast.info(`Desire recalculado: ${processed} items`);
    }
    updateMasterState();
  } catch (err) {
    if (!opts.silent) toast.error(`Recalc Desire: ${err.message}`);
  }
}

window.handleRecalcDesireVisible = recalcDesireVisible;

window.handleCompletarIA = async function(opts = {}) {
  const ids = opts.ids;
  let all;
  if (ids && Array.isArray(ids)) {
    all = (Array.isArray(window.products) ? window.products : []).filter(p => ids.includes(p.id));
  } else {
    all = getAllFilteredRows();
  }
  if (all.length === 0) {
    if (!opts.silent) toast.info('No hay productos');
    return;
  }
  let okTotal = 0;
  const chunks = chunkArray(all, EC_BATCH_SIZE);
  for (const ch of chunks) {
    const payload = ch.map(p => ({
      id: p.id,
      name: p.name,
      category: p.category,
      price: p.price,
      rating: p.rating,
      units_sold: p.units_sold,
      revenue: p.revenue,
      conversion_rate: p.conversion_rate,
      launch_date: p.launch_date,
      date_range: p.date_range,
      image_url: p.image_url || null
    }));
      try {
        const { ok, ko } = await processBatch(payload);
        okTotal += ok;
        if (!opts.silent) toast.info(`IA lote: +${ok} / ${payload.length} (fallos ${ko})`, { duration: 2000 });
      } catch (e) {
        if (!opts.silent) toast.error(`IA lote: ${e.message}`, { duration: 2000 });
      }
  }
  if (!opts.silent) toast.info(`IA: ${okTotal}/${all.length} completados`);
  updateMasterState();
};

function updateProductCaches(productId, updates) {
  const collections = [window.products, window.allProducts, window.__visibleProducts, window.__allProducts];
  collections.forEach(list => {
    if (!Array.isArray(list)) return;
    const found = list.find(item => String(item.id) === String(productId));
    if (!found) return;
    Object.keys(updates).forEach(key => {
      if (updates[key] !== undefined) {
        found[key] = updates[key];
      }
    });
  });
}

function applyRealtimeDom(productId, updates) {
  const tr = document.querySelector(`tr[data-product-id="${productId}"]`);
  if (!tr) return;
  if (Object.prototype.hasOwnProperty.call(updates, 'desire')) {
    const cell = tr.querySelector('td[data-col="desire"]');
    if (cell && !cell.querySelector('textarea.desire-editor')) {
      const text = updates.desire || '';
      cell.title = text;
      const wrap = cell.querySelector('.desire-text');
      if (wrap) {
        wrap.textContent = text;
      }
    }
  }
  const selectSelectors = {
    desire_magnitude: 'td[data-col="desire_magnitude"] select',
    awareness_level: 'td[data-col="awareness_level"] select',
    competition_level: 'td[data-col="competition_level"] select'
  };
  Object.entries(selectSelectors).forEach(([field, selector]) => {
    if (!Object.prototype.hasOwnProperty.call(updates, field)) return;
    const select = tr.querySelector(selector);
    if (!select) return;
    if (document.activeElement === select) return;
    select.value = updates[field] || '';
  });
}

function handleRealtimeRows(rows) {
  rows.forEach(row => {
    if (!row || typeof row !== 'object') return;
    const productId = row.id;
    if (productId === undefined || productId === null) return;
    const updates = {};
    ['desire', 'desire_magnitude', 'awareness_level', 'competition_level'].forEach(field => {
      if (Object.prototype.hasOwnProperty.call(row, field)) {
        updates[field] = row[field];
      }
    });
    if (!Object.keys(updates).length) return;
    updateProductCaches(productId, updates);
    applyRealtimeDom(productId, updates);
  });
  if (window.ecAutoFitColumns && window.gridRoot) {
    try { ecAutoFitColumns(window.gridRoot); } catch (err) { /* noop */ }
  }
}

subscribeSSE('ai-columns', ({ data }) => {
  if (!data || typeof data !== 'object') return;
  const rows = Array.isArray(data.rows) ? data.rows : [];
  handleRealtimeRows(rows);
});

subscribeSSE('products.patch', ({ data }) => {
  if (!data || typeof data !== 'object') return;
  const rows = Array.isArray(data.rows) ? data.rows : [];
  handleRealtimeRows(rows);
});
