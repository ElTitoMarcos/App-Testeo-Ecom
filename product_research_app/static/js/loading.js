import { AbortHub } from '/static/js/net.js';

// loading.js — barra en flujo con soporte multi-host (header y modal) y limpieza de legados

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
  if (tasks.size === 0) {
    // completar al 100% brevemente y colapsar el slot
    s.fill.style.width = '100%';
    s.pctEl.textContent = '100%';
    clearTimeout(s.hideTimer);
    s.hideTimer = setTimeout(() => {
      s.fill.style.width = '0%';
      s.pctEl.textContent = '0%';
      host.classList.remove('active'); // colapsa el slot (height:0)
    }, 300);
    return;
  }
  // promedio simple de progresos
  let sum = 0, last;
  for (const t of tasks.values()) { sum += (t.progress || 0); last = t; }
  const avg = Math.min(0.99, sum / tasks.size);
  const pct = Math.round(avg * 100);
  s.fill.style.width = pct + '%';
  s.pctEl.textContent = pct + '%';
  host.classList.add('active');
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
  }
};

// --- UI de progreso global + Cancel ---
export const ProgressUI = (() => {
  let btn;
  let mounted = false;
  let active = 0;

  function host() {
    return document.querySelector('#app-header .right-controls')
      || document.querySelector('#global-progress-wrapper')
      || document.body;
  }

  function ensure() {
    if (mounted) return;
    btn = document.createElement('button');
    btn.id = 'cancelProgressBtn';
    btn.type = 'button';
    btn.className = 'cancel-progress-btn';
    btn.textContent = 'Cancelar';
    btn.style.display = 'none';
    btn.addEventListener('click', () => {
      AbortHub.cancelAll('user_cancelled');
      btn.disabled = true;
      btn.textContent = 'Cancelando…';
    });
    host()?.appendChild(btn);
    mounted = true;
  }

  function show() {
    ensure();
    if (!btn) return;
    btn.style.display = 'inline-flex';
    btn.disabled = false;
    btn.textContent = 'Cancelar';
  }

  function hide() {
    if (!btn) return;
    btn.style.display = 'none';
  }

  function beginTask() {
    active += 1;
    show();
  }

  function endTask() {
    active = Math.max(0, active - 1);
    if (active === 0) hide();
  }

  return { beginTask, endTask, show, hide };
})();

// --- Tracker utilitario con reparto 20/80 y ETA ---
export function makeProgressTracker({ slot = document.querySelector('#progress-slot-global'), phase = 'import', title } = {}) {
  const BASE = phase === 'import' ? 0 : 0.20;
  const MAX = phase === 'import' ? 0.20 : 0.99;
  const span = Math.max(0.0001, MAX - BASE);
  const defaultTitle = title || (phase === 'import' ? 'Importando catálogo' : 'Completando con IA');

  const task = LoadingHelpers.start(defaultTitle, { host: slot || null });
  let frac = BASE;
  let stageText = phase === 'import' ? 'Preparando…' : 'IA: preparando…';
  let raf = 0;
  let closed = false;

  ProgressUI.beginTask();

  function render() {
    task.step(Math.min(1, Math.max(0, frac)), stageText);
  }

  function toGlobal(x) {
    const clamped = Math.max(0, Math.min(1, Number.isFinite(x) ? x : 0));
    return BASE + span * clamped;
  }

  function setStage(stage) {
    if (stage) {
      stageText = stage;
      render();
    }
  }

  function step(x, stage) {
    const target = Math.min(MAX, Math.max(frac, toGlobal(x)));
    frac = target;
    if (stage) stageText = stage;
    cancelAnimationFrame(raf);
    render();
  }

  function bumpToward(targetFrac, seconds = 1.0, stage) {
    const end = Math.min(MAX, Math.max(frac, targetFrac));
    if (end <= frac) {
      if (stage) stageText = stage;
      render();
      return;
    }
    if (stage) stageText = stage;
    const start = frac;
    const dur = Math.max(50, seconds * 1000);
    const t0 = performance.now();
    cancelAnimationFrame(raf);
    const tick = now => {
      const k = Math.min(1, (now - t0) / dur);
      frac = start + (end - start) * k;
      render();
      if (k < 1) {
        raf = requestAnimationFrame(tick);
      }
    };
    raf = requestAnimationFrame(tick);
  }

  function done({ cancelled = false, finalFrac, stage } = {}) {
    if (closed) return;
    closed = true;
    cancelAnimationFrame(raf);
    if (!cancelled) {
      if (typeof finalFrac === 'number') {
        frac = Math.max(frac, Math.min(1, finalFrac));
      } else {
        frac = 1;
      }
      if (stage) stageText = stage;
      render();
    }
    try { task.done(); }
    catch (_) { /* noop */ }
    ProgressUI.endTask();
  }

  function startEta(ms, { stage } = {}) {
    const targetMs = Math.max(1000, Number(ms) || 0);
    const start = performance.now();
    const label = stage || (phase === 'gpt' ? 'IA… procesando' : 'Procesando…');
    let cleared = false;
    const timer = setInterval(() => {
      const elapsed = performance.now() - start;
      const ratio = Math.min(1, elapsed / targetMs);
      step(Math.min(0.98, ratio), label);
      if (ratio >= 1) {
        bumpToward(0.99, 1.2, label);
        clearInterval(timer);
        cleared = true;
      }
    }, 500);
    return () => {
      if (!cleared) clearInterval(timer);
    };
  }

  render();

  return { step, bumpToward, done, startEta, setStage, base: BASE, max: MAX };
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
