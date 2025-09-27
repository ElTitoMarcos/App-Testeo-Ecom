import { LoadingHelpers } from './loading.js';

const EC_BATCH_SIZE = 10;
const EC_MODEL = 'gpt-4o-mini-2024-07-18';
const IA_COLUMNS = ['desire', 'desire_magnitude', 'awareness_level', 'competition_level'];

const MAX_CONCURRENCY = 3;
const MAX_RETRIES = 5;
const BASE_DELAY = 400;
const ACTIVE_JOB_KEYS = new Set();

function getAllFilteredRows() {
  if (typeof window.getAllFilteredRows === 'function') {
    try {
      return window.getAllFilteredRows();
    } catch (err) {
      console.error('getAllFilteredRows failed', err);
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
      body: JSON.stringify(applied),
      __skipLoadingHook: true
    }).catch(() => {});
    if (window.ecAutoFitColumns && window.gridRoot) ecAutoFitColumns(gridRoot);
  }
  return applied;
}

function isColumnMissing(product, column) {
  if (!product) return false;
  const value = product[column];
  if (value === null || value === undefined) return true;
  if (typeof value === 'string' && value.trim() === '') return true;
  return false;
}

function formatProductPayload(product) {
  return {
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
    image_url: product.image_url || null
  };
}

function buildJobs(products) {
  const grouped = new Map();
  const plannedKeys = new Set();
  let activeSkipped = 0;

  for (const product of products) {
    if (!product || product.id === undefined || product.id === null) continue;
    const idStr = String(product.id);
    let entry = grouped.get(idStr);
    if (!entry) {
      entry = { product, columns: [] };
      grouped.set(idStr, entry);
    }
    for (const col of IA_COLUMNS) {
      if (!isColumnMissing(product, col)) continue;
      const key = `${idStr}:${col}`;
      if (ACTIVE_JOB_KEYS.has(key)) {
        activeSkipped++;
        continue;
      }
      if (plannedKeys.has(key)) continue;
      plannedKeys.add(key);
      entry.columns.push(col);
    }
    if (entry.columns.length === 0) {
      grouped.delete(idStr);
    }
  }

  const entries = Array.from(grouped.values());
  const jobs = [];
  for (let i = 0; i < entries.length; i += EC_BATCH_SIZE) {
    const slice = entries.slice(i, i + EC_BATCH_SIZE);
    const comboCount = slice.reduce((acc, item) => acc + item.columns.length, 0);
    if (!comboCount) continue;
    jobs.push({
      url: '/api/ia/batch-columns',
      payload: { model: EC_MODEL, items: slice.map(item => formatProductPayload(item.product)) },
      entries: slice.map(item => ({ id: item.product.id, product: item.product, columns: item.columns.slice() })),
      comboCount
    });
  }

  return { jobs, plannedKeys: Array.from(plannedKeys), totalUnits: plannedKeys.size, activeSkipped };
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function withBackoff(fn) {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const message = String((err && err.message) || err || '');
      if (/(429|rate limit)/i.test(message)) {
        const wait = Math.min(5000, BASE_DELAY * Math.pow(2, attempt)) + Math.random() * 150;
        await sleep(wait);
        continue;
      }
      if (attempt === MAX_RETRIES) throw err;
      await sleep(BASE_DELAY * (attempt + 1));
    }
  }
  throw new Error('unexpected_backoff_failure');
}

async function runJobs(jobs, onProgress) {
  if (!jobs.length) return [];
  let idx = 0;
  let inFlight = 0;
  const results = [];

  return new Promise(resolve => {
    const launchNext = () => {
      while (inFlight < MAX_CONCURRENCY && idx < jobs.length) {
        const job = jobs[idx++];
        inFlight++;
        withBackoff(() => fetch(job.url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(job.payload),
          __skipLoadingHook: true
        }))
          .then(async res => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json().catch(() => ({}));
          })
          .then(data => {
            results.push({ job, ok: true, data });
          })
          .catch(err => {
            console.error('IA job failed', err);
            results.push({ job, ok: false, error: String((err && err.message) || err) });
          })
          .finally(() => {
            inFlight--;
            if (typeof onProgress === 'function') {
              try { onProgress(job); } catch (progressErr) { console.error(progressErr); }
            }
            if (results.length === jobs.length) {
              resolve(results);
            } else {
              launchNext();
            }
          });
      }
    };
    launchNext();
  });
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
      } catch (e) {}
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
  if (!Array.isArray(all) || all.length === 0) {
    if (!opts.silent) toast.info('No hay productos');
    return;
  }

  const host = document.querySelector('header') || document.body;
  const tracker = LoadingHelpers.start('Importando y completando columnas con IA…', { host });

  const { jobs, plannedKeys, totalUnits, activeSkipped } = buildJobs(all);

  if (!jobs.length || totalUnits === 0) {
    if (!opts.silent) {
      if (activeSkipped > 0) {
        toast.info('IA en curso, espera a que finalice');
      } else {
        toast.info('Todo actualizado: no hay columnas pendientes');
      }
    }
    tracker.setStage(activeSkipped > 0 ? 'IA en curso…' : 'Sin trabajo pendiente');
    tracker.done();
    return;
  }

  plannedKeys.forEach(key => ACTIVE_JOB_KEYS.add(key));

  let completedUnits = 0;
  tracker.step(0, `IA: 0/${totalUnits}`);

  let results;
  try {
    results = await runJobs(jobs, job => {
      completedUnits += job.comboCount;
      const frac = Math.min(0.99, totalUnits ? completedUnits / totalUnits : 0);
      const doneUnits = Math.min(completedUnits, totalUnits);
      tracker.step(frac, `IA: ${doneUnits}/${totalUnits}`);
    });
  } catch (err) {
    tracker.setStage('Error en IA');
    tracker.done();
    plannedKeys.forEach(key => ACTIVE_JOB_KEYS.delete(key));
    if (!opts.silent) toast.error(`IA falló: ${err.message}`);
    return;
  }

  let successUnits = 0;
  const errors = [];

  for (const res of results) {
    if (!res || !res.job) continue;
    const job = res.job;
    if (!res.ok) {
      errors.push(res.error || 'Error desconocido');
      continue;
    }
    const data = res.data || {};
    const okMap = data.ok || {};
    const koMap = data.ko || {};
    for (const entry of job.entries) {
      const pidStr = String(entry.id);
      if (koMap && koMap[pidStr]) {
        errors.push(`Producto ${pidStr}: ${koMap[pidStr]}`);
        continue;
      }
      const updates = okMap ? okMap[pidStr] : null;
      if (!updates) {
        errors.push(`Producto ${pidStr}: sin datos`);
        continue;
      }
      const filtered = {};
      for (const col of entry.columns) {
        if (updates[col] !== undefined && updates[col] !== null && !(typeof updates[col] === 'string' && updates[col].trim() === '')) {
          filtered[col] = updates[col];
        }
      }
      const applied = applyUpdates(entry.product, filtered);
      const appliedCols = Object.keys(applied).length;
      successUnits += appliedCols;
      const missing = Math.max(0, entry.columns.length - appliedCols);
      if (missing > 0) {
        if (!updates || Object.keys(filtered).length === 0) {
          errors.push(`Producto ${pidStr}: columnas sin cambios`);
        }
      }
    }
  }

  const normalizedSuccess = Math.min(successUnits, totalUnits);
  const normalizedFailures = Math.max(0, totalUnits - normalizedSuccess);

  tracker.step(0.99, `IA: ${Math.min(completedUnits, totalUnits)}/${totalUnits}`);
  tracker.done();
  plannedKeys.forEach(key => ACTIVE_JOB_KEYS.delete(key));

  if (!opts.silent) {
    if (normalizedSuccess > 0) {
      toast.success(`IA completada: ${normalizedSuccess}/${totalUnits} columnas`);
    }
    if (normalizedFailures > 0) {
      toast.error(`IA sin completar: ${normalizedFailures} columnas`);
    } else if (errors.length) {
      toast.error('IA completada con avisos');
    }
  }

  if (errors.length) {
    console.warn('Errores IA', errors);
  }

  try {
    await reloadTable({ skipProgress: true });
  } catch (err) {
    console.error('reloadTable failed', err);
  }
};
