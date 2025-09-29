import { AbortHub, fetchJson } from '/static/js/net.js';
import { makeProgressTracker } from '/static/js/loading.js';

const EC_BATCH_SIZE = 10;
const EC_MODEL = 'gpt-4o-mini-2024-07-18';

function aiFillStart(total, jobId) {
  document.dispatchEvent(new CustomEvent('ai-fill-start', { detail: { total, jobId } }));
}

function aiFillProgress(done, total, jobId) {
  const percent = Math.round((done / Math.max(total, 1)) * 100);
  document.dispatchEvent(new CustomEvent('ai-fill-progress', { detail: { done, total, percent, jobId } }));
}

function aiFillDone(total, jobId) {
  document.dispatchEvent(new CustomEvent('ai-fill-done', { detail: { total, jobId } }));
}

function aiFillError(message, jobId) {
  document.dispatchEvent(new CustomEvent('ai-fill-error', { detail: { message, jobId } }));
}

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

function applyUpdates(product, updates) {
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
  if (Object.keys(applied).length) {
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

async function processBatch(items, { signal } = {}) {
  const data = await fetchJson('/api/ia/batch-columns', {
    method: 'POST',
    body: JSON.stringify({ model: EC_MODEL, items }),
    signal,
    __skipLoadingHook: true
  });
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
    const data = await fetchJson('/api/recalc-desire-all', {
      method: 'POST',
      body: JSON.stringify({ scope: 'filtered', ids })
    });
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
  const total = all.length;
  const jobId = (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now());
  const chunks = chunkArray(all, EC_BATCH_SIZE);
  const totalBatches = chunks.length;
  const tracker = makeProgressTracker({ phase: 'gpt' });
  const etaMs = Math.max(30000, total * 2500);
  const stopEta = tracker.startEta(etaMs, { stage: 'IA… procesando' });

  let okTotal = 0;
  let processed = 0;
  let completedBatches = 0;

  aiFillStart(total, jobId);

  try {
    for (const chunk of chunks) {
      const payload = chunk.map(p => ({
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

      const ctrl = AbortHub.make();
      const release = AbortHub.track(ctrl);
      try {
        const { ok, ko } = await processBatch(payload, { signal: ctrl.signal });
        okTotal += ok;
        if (!opts.silent) {
          toast.info(`IA lote: +${ok} / ${payload.length} (fallos ${ko})`, { duration: 2000 });
        }
      } catch (err) {
        if (!opts.silent && err?.name !== 'AbortError') {
          toast.error(`IA lote: ${err.message || 'Error'}`, { duration: 2000 });
        }
        throw err;
      } finally {
        release();
        try { ctrl.abort(); } catch (_) {}
      }

      completedBatches += 1;
      processed += payload.length;
      tracker.step(completedBatches / Math.max(1, totalBatches), `IA… ${processed}/${total}`);
      aiFillProgress(Math.min(processed, total), total, jobId);
    }
  } catch (err) {
    stopEta();
    if (err?.name === 'AbortError' || String(err?.message).includes('user_cancelled')) {
      tracker.done({ cancelled: true });
      return;
    }
    tracker.done({ cancelled: true });
    aiFillError(err?.message || 'Error en IA', jobId);
    throw err;
  }

  stopEta();
  tracker.done({ cancelled: false });
  aiFillDone(total, jobId);

  if (!opts.silent) {
    toast.info(`IA: ${okTotal}/${all.length} completados`);
  }

  updateMasterState();
  try {
    await window.reloadTable?.({ skipProgress: true });
  } catch (_) {
    /* ignore */
  }
};
