// loading.js — barra en flujo con soporte multi-host (header y modal) y limpieza de legados

// ===== ETA por producto + Watchdog =====
let etaTimer = null;
let etaStart = 0;
let etaTotalMs = 0;
let etaLastPct = 0;
let etaFinished = false;

// Señales backend
let lastBackendSignalAt = 0;
let sawBackendActivity = false;
let watchdogTimer = null;

// Rango 3.4–3.7 s/ítem (por defecto 3.55)
const ETA_MIN = 3.4;
const ETA_MAX = 3.7;
const ETA_DEFAULT = 3.55;

// Watchdog: si hay silencio del backend y la ETA va alta, cerramos
const WATCHDOG_CHECK_MS = 1500;
const WATCHDOG_IDLE_MS = 10000;
const WATCHDOG_PCT_GATE = 80;

function startEtaProgress(totalItems, secondsPerItem = ETA_DEFAULT) {
  stopEtaProgress(false);
  etaFinished = false;

  if (!Number.isFinite(totalItems) || totalItems <= 0) totalItems = 1;
  const spi = Math.max(ETA_MIN, Math.min(ETA_MAX, secondsPerItem));

  etaTotalMs = Math.max(500, Math.round(totalItems * spi * 1000));
  etaStart = Date.now();
  etaLastPct = 0;

  // reset señales backend
  lastBackendSignalAt = 0;
  sawBackendActivity = false;

  setProgressUI(0);

  // Tick de ETA
  etaTimer = setInterval(() => {
    if (etaFinished) return;
    const elapsed = Date.now() - etaStart;
    const raw = (elapsed / etaTotalMs) * 100;
    const pct = Math.max(0, Math.min(99, Math.floor(raw)));
    if (pct > etaLastPct) {
      etaLastPct = pct;
      setProgressUI(pct);
    }
  }, 150);

  // Watchdog
  watchdogTimer = setInterval(() => {
    if (etaFinished) return clearInterval(watchdogTimer);
    const now = Date.now();
    const pctNow = etaLastPct;

    if (sawBackendActivity &&
        pctNow >= WATCHDOG_PCT_GATE &&
        lastBackendSignalAt > 0 &&
        (now - lastBackendSignalAt) >= WATCHDOG_IDLE_MS) {
      console.debug('[watchdog] cierre por silencio backend con ETA alta');
      markBackendDone();
    }
  }, WATCHDOG_CHECK_MS);
}

function stopEtaProgress(done) {
  if (etaTimer) { clearInterval(etaTimer); etaTimer = null; }
  if (watchdogTimer) { clearInterval(watchdogTimer); watchdogTimer = null; }
  if (done) {
    etaFinished = true;
    setProgressUI(100);
  }
}

function markBackendDone() {
  if (etaFinished) return;
  stopEtaProgress(true);
  refreshProductsTable();
}

async function refreshProductsTable() {
  try {
    if (window.table && table.ajax && typeof table.ajax.reload === 'function') {
      table.ajax.reload(null, false);
      return;
    }
    if (typeof window.reloadProductsTable === 'function') {
      await window.reloadProductsTable();
      return;
    }
    if (typeof window.fetchProductsAndRender === 'function') {
      await window.fetchProductsAndRender();
      return;
    }
    location.reload();
  } catch (e) {
    console.error('Error refrescando tabla:', e);
    location.reload();
  }
}

function setProgressUI(pct) {
  const bar = document.querySelector('.progress-fill, #import-progress .bar, [role="progressbar"] .bar');
  if (bar) bar.style.width = pct + '%';

  const label = document.querySelector('.progress-percent, #import-progress .percent');
  if (label) label.textContent = pct + '%';

  const topLabel = document.querySelector('#top-import-label, .import-status-text');
  if (topLabel) topLabel.textContent = pct + '%';
}

function wireJobDoneSignals({ jobId, sseUrl, pollUrl, totalItems }) {
  if (sseUrl) {
    try {
      const es = new EventSource(sseUrl);
      const close = () => { try { es.close(); } catch {} };

      const onMessage = (raw) => {
        backendHeartbeat();
        const msg = parseSseData(raw?.data);
        if (isDoneMsg(msg, totalItems)) {
          close();
          markBackendDone();
        }
      };

      es.addEventListener('message', (e) => onMessage(e));
      es.addEventListener('done', (e) => onMessage(e));
      es.addEventListener('complete', (e) => onMessage(e));
      es.addEventListener('finished', (e) => onMessage(e));

      es.addEventListener('error', () => {
        close();
        if (pollUrl) startPolling(pollUrl, totalItems);
      });

      return;
    } catch {
      // continuamos con polling
    }
  }

  if (pollUrl) startPolling(pollUrl, totalItems);
}

function startPolling(url, totalItems) {
  const iv = setInterval(async () => {
    if (etaFinished) return clearInterval(iv);
    try {
      const r = await fetch(url, { cache: 'no-store' });
      backendHeartbeat();
      let j = null;
      try { j = await r.json(); } catch {}
      if (isDoneMsg(j, totalItems)) {
        clearInterval(iv);
        markBackendDone();
      }
    } catch {
      // ignoramos y seguimos
    }
  }, 1200);
}

function backendHeartbeat() {
  lastBackendSignalAt = Date.now();
  sawBackendActivity = true;
}

function parseSseData(data) {
  if (!data) return null;
  if (typeof data === 'string') {
    try { return JSON.parse(data); }
    catch { return { _text: data }; }
  }
  if (typeof data === 'object') return data;
  return null;
}

function isDoneMsg(m, totalItems) {
  if (!m) return false;

  const text = (m._text || m.message || '').toString();
  if (text.includes('ai.run done')) return true;
  if (text.toLowerCase().includes('completed') || text.toLowerCase().includes('finished')) return true;

  const norm = (v) => (v == null ? '' : String(v)).toLowerCase();
  const tnum = (v) => { const n = Number(v); return Number.isFinite(n) ? n : NaN; };

  const phase = norm(m.phase || m.stage);
  const status = norm(m.status || m.state);
  const pending = tnum(m.pending ?? m.remaining);
  const remaining = tnum(m.remaining ?? m.pending);
  const processed = tnum(m.processed ?? m.done ?? m.parsed);
  const total = tnum(m.total ?? m.items ?? totalItems);

  const notAiPhases = ['upload','import','parse','parsed','queued','triage','reading'];
  if (phase && notAiPhases.includes(phase)) return false;

  if (status === 'done' || status === 'ok' || status === 'completed' || status === 'finished') {
    if (Number.isFinite(remaining) && remaining === 0) return true;
    if (Number.isFinite(pending) && pending === 0) return true;
    if (Number.isFinite(processed) && Number.isFinite(total) && total > 0 && processed >= total) return true;
  }

  if (Number.isFinite(remaining) && remaining === 0) return true;
  if (Number.isFinite(pending) && pending === 0) return true;
  if (Number.isFinite(processed) && Number.isFinite(total) && total > 0 && processed >= total) return true;

  return false;
}

// ===== Legacy killer: por si algún módulo viejo intenta crear su barra/overlay =====
(function killLegacy() {
  const zap = () => {
    document.querySelectorAll('#top-progress, .loading-overlay').forEach(n => n.remove());
  };
  zap();
  const mo = new MutationObserver(zap);
  mo.observe(document.documentElement, { childList: true, subtree: true });
})();

// ===== ProgressRail: una barra por "host" (slot). host = elemento contenedor (header, modal, etc.)
function createRailInHost(host) {
  if (!host) return null;
  let rail = host.querySelector(':scope > .progress-rail');
  if (rail) return rail;

  host.classList.add('active');
  rail = document.createElement('div');
  rail.className = 'progress-rail';
  rail.innerHTML = `
    <div class="progress-fill"></div>
    <span class="progress-meta"><span class="progress-title">Proceso</span><span class="progress-stage">Iniciando…</span></span>
    <span class="progress-percent">0%</span>
  `;
  host.appendChild(rail);
  return rail;
}

function ensureSlot(el) {
  // Si el host es un modal o un hijo suyo, usa/crea .modal-progress-slot. Si no, usa #progress-slot-global.
  let host = el && (el.closest('.modal')?.querySelector('.modal-progress-slot'));
  if (!host) host = document.querySelector('#progress-slot-global');

  // Si no existe el slot de modal, créalo en caliente bajo el title del diálogo
  if (!host && el && el.closest('.modal')) {
    const modal = el.closest('.modal');
    const header = modal.querySelector('.modal-header') || modal.querySelector('[data-role="modal-header"]') || modal;
    host = document.createElement('div');
    host.className = 'modal-progress-slot progress-slot active';
    header.appendChild(host);
  }
  return host;
}

const Rails = new WeakMap(); // host -> { rail, fill, pctEl, titleEl, stageEl, tasks: Map }

function getRailState(host) {
  if (!host) return null;
  let state = Rails.get(host);
  if (state) return state;
  const rail = createRailInHost(host);
  if (!rail) return null;
  const fill = rail.querySelector('.progress-fill');
  const pctEl = rail.querySelector('.progress-percent');
  const titleEl = rail.querySelector('.progress-title');
  const stageEl = rail.querySelector('.progress-stage');
  state = { rail, fill, pctEl, titleEl, stageEl, tasks: new Map(), hideTimer: null };
  Rails.set(host, state);
  return state;
}

function refreshHost(host) {
  const s = getRailState(host); if (!s) return;
  const tasks = s.tasks;
  const hasTasks = tasks.size > 0;

  // 1) Activar/colapsar el slot (barra)
  host.classList.toggle('active', hasTasks);

  const isGlobalHost = host && host.id === 'progress-slot-global';

  // 2) Mostrar/ocultar el wrapper completo
  if (isGlobalHost) {
    const wrapper = document.getElementById('global-progress-wrapper');
    if (wrapper) wrapper.classList.toggle('show', hasTasks);
  }

  // 3) Mostrar/ocultar el botón Cancelar
  if (isGlobalHost) {
    const cancelBtn = document.getElementById('progress-cancel-btn');
    if (cancelBtn) cancelBtn.style.display = hasTasks ? 'inline-flex' : 'none';
  }

  // 4) Si no hay tareas, limpiar el slot (evita textos/0%)
  if (!hasTasks) {
    clearTimeout(s.hideTimer);
    s.hideTimer = null;
    tasks.clear();
    host.innerHTML = '';
    Rails.delete(host);
    return;
  }
  // promedio simple de progresos
  let sum = 0, last;
  for (const t of tasks.values()) { sum += (t.progress || 0); last = t; }
  const avg = Math.min(0.99, sum / tasks.size);
  const pct = Math.round(avg * 100);
  s.fill.style.width = pct + '%';
  s.pctEl.textContent = pct + '%';
  if (last) {
    if (last.title) s.titleEl.textContent = last.title;
    if (last.stage) s.stageEl.textContent = last.stage;
  }
}

function startTaskInHost({ title = 'Procesando…', hostEl = null } = {}) {
  const host = ensureSlot(hostEl);
  const s = getRailState(host);
  if (!s) return { step(){}, setStage(){}, done(){} };

  const id = `${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
  s.tasks.set(id, { progress: 0, title, stage: 'Iniciando…' });
  refreshHost(host);

  return {
    step(frac, stage) {
      const t = s.tasks.get(id); if (!t) return;
      t.progress = Math.max(0, Math.min(1, frac));
      if (stage) t.stage = stage;
      refreshHost(host);
    },
    setStage(stage) {
      const t = s.tasks.get(id); if (!t) return;
      t.stage = stage; refreshHost(host);
    },
    done() {
      s.tasks.delete(id);
      refreshHost(host);
    }
  };
}

// Exponer helper público
export const LoadingHelpers = {
  start(title, opts = {}) {
    return startTaskInHost({ title, hostEl: opts.host || null });
  },
  startEtaProgress,
  stopEtaProgress,
  markBackendDone,
  wireJobDoneSignals,
  setProgressUI
};

export {
  startEtaProgress,
  stopEtaProgress,
  markBackendDone,
  wireJobDoneSignals,
  setProgressUI
};

if (typeof window !== 'undefined') {
  Object.assign(window, {
    startEtaProgress,
    stopEtaProgress,
    markBackendDone,
    wireJobDoneSignals,
    setProgressUI
  });
}

// ===== Hooks de red: si se pasa init.__hostEl, el progreso aparece en ese host; si no, en el global =====
(() => {
  const _fetch = window.fetch;
  window.fetch = async function(input, init = {}) {
    if (init && init.__skipLoadingHook) {
      return _fetch.call(this, input, init);
    }
    const host = init.__hostEl || null;
    const t = startTaskInHost({ title: 'Cargando datos', hostEl: host });
    try { return await _fetch(input, init); }
    finally { t.done(); }
  };

  const _open = XMLHttpRequest.prototype.open;
  const _send = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function(method, url, async, user, password) {
    this.__method = method; this.__url = url;
    return _open.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function(body) {
    if (this.__skipLoadingHook) {
      return _send.apply(this, arguments);
    }
    const host = this.__hostEl || null;
    const t = startTaskInHost({ title: 'Comunicando…', hostEl: host });
    const end = () => t.done();
    this.addEventListener('loadend', end);
    this.addEventListener('error', end);
    this.addEventListener('abort', end);
    try { return _send.apply(this, arguments); }
    catch (e) { end(); throw e; }
  };
})();

document.addEventListener('DOMContentLoaded', () => {
  const wrapper = document.getElementById('global-progress-wrapper');
  const host = document.getElementById('progress-slot-global');
  const btn = document.getElementById('progress-cancel-btn');
  if (wrapper) wrapper.classList.remove('show');
  if (host) {
    host.classList.remove('active');
    host.innerHTML = '';
  }
  if (btn) btn.style.display = 'none';
});

// Ajusta el offset del thead de #productTable para que quede justo bajo el topbar sticky
(() => {
  const root   = document.documentElement;
  const topbar = document.getElementById('topBar') || document.querySelector('header');

  function setStickyOffset(){
    const h = topbar ? Math.round(topbar.getBoundingClientRect().height) : 0;
    root.style.setProperty('--topbar-sticky-offset', `${h}px`);
  }

  // Inicial + en resize
  setStickyOffset();
  window.addEventListener('resize', setStickyOffset);

  // Recalcular si cambia la altura del topbar (p.ej., aparece/desaparece la barra de progreso)
  if (window.ResizeObserver && topbar){
    const ro = new ResizeObserver(setStickyOffset);
    ro.observe(topbar);
  }
})();
