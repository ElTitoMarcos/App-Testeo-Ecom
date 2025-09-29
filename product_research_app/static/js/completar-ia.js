import { fetchJson } from '/static/js/net.js';

const GP = window.GlobalProgress ?? {
  beginAI() {},
  setAIFrac() {},
  finishOk() {},
  finishError() {},
  showCancel() {},
  isActive() { return false; }
};

async function startAIFill(ids) {
  const body = ids && ids.length ? { product_ids: ids } : {};
  return fetchJson('/ai_fill/start', {
    method: 'POST',
    body: JSON.stringify(body)
  });
}

async function pollAIFill(jobId, onProgress) {
  while (true) {
    let st;
    try {
      st = await fetchJson(`/ai_fill/progress?job_id=${encodeURIComponent(jobId)}&t=${Date.now()}`, {
        __skipLoadingHook: true,
        cache: 'no-store'
      });
    } catch (err) {
      if (err && err.name === 'AbortError') throw err;
      await new Promise(res => setTimeout(res, 1000));
      continue;
    }

    const pct = Math.max(0, Math.min(100, st.pct ?? ((st.processed ?? 0) / Math.max(1, st.total ?? 0) * 100)));
    if (typeof onProgress === 'function') {
      try { onProgress(pct / 100, st); } catch (_) {}
    }

    const statusVal = String(st.status || st.state || '').toLowerCase();
    const cancelled = statusVal === 'cancelled' || statusVal === 'canceled';
    const done = Boolean(st.done || cancelled || statusVal === 'done' || statusVal === 'completed' || statusVal === 'finished');
    if (done) {
      return { pct, cancelled, status: statusVal, payload: st };
    }
    await new Promise(res => setTimeout(res, 600));
  }
}

function computeEtaMs(total = 0, estimatedMs = 0) {
  if (estimatedMs && estimatedMs > 0) return estimatedMs;
  if (total && Number.isFinite(total)) {
    const perItem = 2500;
    return Math.max(4000, total * perItem);
  }
  return 6000;
}

export async function runAIFill(ids) {
  try {
    const start = await startAIFill(ids);
    const jobId = start.job_id;
    const total = Number(start.total || 0);
    const estMs = computeEtaMs(total, Number(start.estimated_ms || 0));

    GP.beginAI({
      estMs,
      onCancel: () => fetch('/ai_fill/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId })
      }).catch(() => {})
    });

    const result = await pollAIFill(jobId, (aiFrac) => {
      GP.setAIFrac(aiFrac);
    });

    GP.finishOk();
    try {
      await window.reloadTable?.({ skipProgress: true });
    } catch (_) {}

    if (result.cancelled) {
      toast.info('Proceso de IA cancelado');
    } else {
      toast.success('Columnas de IA generadas');
    }
    return result;
  } catch (err) {
    if (typeof GP.isActive === 'function' ? GP.isActive() : true) {
      GP.finishError();
    }
    toast.error(err?.message || 'Error en IA');
    throw err;
  }
}

function resolveIds(opts = {}) {
  if (opts && Array.isArray(opts.ids) && opts.ids.length) {
    return opts.ids;
  }
  if (typeof window.getSelectedProductIds === 'function') {
    const selected = window.getSelectedProductIds();
    if (Array.isArray(selected) && selected.length) return selected;
  }
  if (opts && opts.allVisible && Array.isArray(window.products)) {
    return window.products.map(p => p.id);
  }
  return [];
}

window.runAIFill = runAIFill;
window.handleCompletarIA = async function handleCompletarIA(opts = {}) {
  const ids = resolveIds(opts);
  if (!ids.length && (!Array.isArray(window.products) || window.products.length === 0)) {
    toast.info('No hay productos disponibles');
    return null;
  }
  return runAIFill(ids.length ? ids : undefined);
};
