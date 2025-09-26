import { fetchJson } from './net.js';

const net = (window.net && typeof window.net.json === 'function')
  ? window.net
  : {
      json: async (url, options = {}) => {
        const opts = { ...options };
        if (opts.body && typeof opts.body !== 'string') {
          opts.body = JSON.stringify(opts.body);
        }
        if (!opts.method) {
          opts.method = opts.body ? 'POST' : 'GET';
        }
        return fetchJson(url, opts);
      }
    };
window.net = Object.assign(window.net || {}, net);

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

function normalizeIds(values = []) {
  const unique = new Set();
  values.forEach(val => {
    const num = Number(val);
    if (Number.isInteger(num)) unique.add(num);
  });
  return Array.from(unique).sort((a, b) => a - b);
}

window.handleCompletarIA = async (opts = {}) => {
  const baseIds = Array.isArray(opts.ids) && opts.ids.length
    ? normalizeIds(opts.ids)
    : [];

  const collectedIds = [...baseIds];

  if (collectedIds.length === 0) {
    const { ids: allIds = [] } = await net.json('/api/products/ids', { method: 'GET' });
    collectedIds.push(...normalizeIds(allIds));
  }

  const ids = normalizeIds(collectedIds);

  if (ids.length === 0) {
    toast.info('No hay productos');
    return { counts: { ok: 0, ko: 0 }, summary: 'IA columns updated: 0/0' };
  }

  toast.info(`Generando columnas IA para ${ids.length} productos…`);
  try {
    const res = await net.json('/api/ia/run', { method: 'POST', body: { ids } });
    const ok = res?.counts?.ok ?? 0;
    const ko = res?.counts?.ko ?? 0;
    toast.success(`Columnas IA actualizadas: ${ok}/${ids.length} (fallidos: ${ko}).`);
    if (typeof window.reloadProducts === 'function') window.reloadProducts();
    if (typeof window.updateMasterState === 'function') window.updateMasterState();
    return res;
  } catch (err) {
    console.error('handleCompletarIA failed', err);
    throw err;
  }
};

