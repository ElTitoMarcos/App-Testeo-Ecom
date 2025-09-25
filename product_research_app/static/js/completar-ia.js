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

async function launchDesireBackfill(opts = {}) {
  const payload = {};
  if (opts.scope) payload.scope = opts.scope;
  if (Array.isArray(opts.ids) && opts.ids.length) payload.ids = opts.ids;
  if (opts.batch_size) payload.batch_size = opts.batch_size;
  if (opts.parallel) payload.parallel = opts.parallel;
  if (opts.max_retries !== undefined) payload.max_retries = opts.max_retries;

  let resp;
  try {
    resp = await fetch('/api/desire/backfill', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (err) {
    if (!opts.silent) toast.error(`Backfill Desire: ${err.message}`);
    return null;
  }
  if (!resp.ok) {
    let msg = resp.statusText || 'Error';
    try {
      const error = await resp.json();
      if (error && error.error) msg = error.error;
    } catch {}
    if (!opts.silent) toast.error(`Backfill Desire: ${msg}`);
    return null;
  }
  let data;
  try {
    data = await resp.json();
  } catch {
    data = {};
  }
  const taskId = data && data.task_id;
  if (!taskId) {
    if (!opts.silent) toast.error('Backfill Desire: sin task_id');
    return null;
  }
  if (!opts.silent) toast.info('Backfill Desire iniciado');

  const pollInterval = opts.pollInterval || 1500;
  let finalStatus = null;
  let lastPct = -1;
  const start = Date.now();

  while (true) {
    await new Promise(resolve => setTimeout(resolve, pollInterval));
    let statusResp;
    try {
      statusResp = await fetch(`/_desire_status?task_id=${encodeURIComponent(taskId)}`);
    } catch {
      continue;
    }
    if (!statusResp.ok) continue;
    let status;
    try { status = await statusResp.json(); } catch { continue; }
    if (!status) continue;
    finalStatus = status;
    if (status.status === 'error') {
      if (!opts.silent) {
        const msg = status.message || 'Error';
        toast.error(`Backfill Desire: ${msg}`);
      }
      return status;
    }
    if (status.status === 'done') {
      break;
    }
    if (!opts.silent && status.status === 'running') {
      const pct = Math.round((Number(status.progress) || 0) * 100);
      if (pct >= 0 && pct !== lastPct) {
        lastPct = pct;
        const elapsed = Math.round((Date.now() - start) / 1000);
        toast.info(`Backfill Desire ${pct}% (${elapsed}s)`, { duration: 1200 });
      }
    }
  }

  try {
    const resResp = await fetch(`/_desire_results?task_id=${encodeURIComponent(taskId)}`);
    if (resResp.ok) {
      const resJson = await resResp.json();
      if (resJson && typeof resJson === 'object') {
        finalStatus = { ...(finalStatus || {}), result: resJson };
      }
    }
  } catch {}

  if (!opts.silent) {
    const ok = Number((finalStatus && finalStatus.done) || 0);
    const fail = Number((finalStatus && finalStatus.failed) || 0);
    toast.info(`Backfill Desire completado (${ok} ok, ${fail} fallos)`);
  }
  updateMasterState();
  return finalStatus;
}

window.launchDesireBackfill = launchDesireBackfill;

async function launchAudit(opts = {}) {
  const payload = {};
  if (Array.isArray(opts.ids) && opts.ids.length) payload.ids = opts.ids;

  let resp;
  try {
    resp = await fetch('/api/audit/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (err) {
    if (!opts.silent) toast.error(`Auditoría: ${err.message}`);
    return null;
  }
  if (!resp.ok) {
    let msg = resp.statusText || 'Error';
    try {
      const error = await resp.json();
      if (error && error.error) msg = error.error;
    } catch {}
    if (!opts.silent) toast.error(`Auditoría: ${msg}`);
    return null;
  }
  let data;
  try { data = await resp.json(); } catch { data = {}; }
  const taskId = data && data.task_id;
  if (!taskId) {
    if (!opts.silent) toast.error('Auditoría: sin task_id');
    return null;
  }
  if (!opts.silent) toast.info('Auditoría iniciada');

  const pollInterval = opts.pollInterval || 1500;
  let finalStatus = null;

  while (true) {
    await new Promise(resolve => setTimeout(resolve, pollInterval));
    let statusResp;
    try {
      statusResp = await fetch(`/_audit_status?task_id=${encodeURIComponent(taskId)}`);
    } catch {
      continue;
    }
    if (!statusResp.ok) continue;
    let status;
    try { status = await statusResp.json(); } catch { continue; }
    if (!status) continue;
    finalStatus = status;
    if (status.status === 'error') {
      if (!opts.silent) {
        const msg = status.message || 'Error';
        toast.error(`Auditoría: ${msg}`);
      }
      break;
    }
    if (status.status === 'done') {
      break;
    }
  }

  if (finalStatus && finalStatus.status === 'done') {
    if (!opts.silent) toast.success('Auditoría completada');
    if (typeof updateMasterState === 'function') updateMasterState();
  }
  return finalStatus;
}

window.launchAudit = launchAudit;

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
