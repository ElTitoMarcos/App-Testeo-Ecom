const APPROACH = 0.30; // suavizado de animación (0.2–0.4 va bien)
function clamp01(x){ return Math.max(0, Math.min(1, x || 0)); }

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

const Rails = new WeakMap(); // host -> { host, rail, fill, pctEl, titleEl, stageEl, tasks: Map, shown, reported, unlock100, _raf }

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
  state = {
    host,
    rail,
    fill,
    pctEl,
    titleEl,
    stageEl,
    tasks: new Map(),
    hideTimer: null,
    shown: 0,
    reported: 0,
    unlock100: false,
    _raf: null
  };
  Rails.set(host, state);
  return state;
}

function finishAndHide(state) {
  const host = state.host;
  if (!host) return;
  if (state._raf) {
    cancelAnimationFrame(state._raf);
    state._raf = null;
  }
  state.shown = 0;
  state.reported = 0;
  state.unlock100 = false;
  if (state.fill) state.fill.style.width = '0%';
  if (state.pctEl) state.pctEl.textContent = '0%';
  host.classList.remove('active');

  const isGlobalHost = host && host.id === 'progress-slot-global';
  if (isGlobalHost) {
    const wrapper = document.getElementById('global-progress-wrapper');
    if (wrapper) wrapper.classList.remove('show');
    const cancelBtn = document.getElementById('progress-cancel-btn');
    if (cancelBtn) cancelBtn.style.display = 'none';
  }

  state.tasks.clear();
  host.innerHTML = '';
  Rails.delete(host);
}

function rafLoop(state) {
  state._raf = null;
  const target = state.unlock100 ? 1 : state.reported;
  state.shown = state.shown + (target - state.shown) * APPROACH;

  const shownPct = Math.round(state.shown * 100);
  const realPct  = Math.round(state.reported * 100);

  state.fill.style.width = (state.shown * 100) + '%';

  const labelPct = state.unlock100 ? shownPct : realPct;
  if (state.pctEl) state.pctEl.textContent = labelPct + '%';

  if (state.unlock100 && state.shown >= 0.999) {
    state.unlock100 = false;
    finishAndHide(state);
    return;
  }
  state._raf = requestAnimationFrame(() => rafLoop(state));
}

function ensureLoop(state) {
  if (!state._raf) state._raf = requestAnimationFrame(() => rafLoop(state));
}

function refreshHost(host) {
  const s = getRailState(host); if (!s) return;
  const tasks = s.tasks;
  const hasTasks = tasks.size > 0;
  const isActive = hasTasks || s.unlock100;

  host.classList.toggle('active', isActive);

  const isGlobalHost = host && host.id === 'progress-slot-global';

  if (isGlobalHost) {
    const wrapper = document.getElementById('global-progress-wrapper');
    if (wrapper) wrapper.classList.toggle('show', isActive);
    const cancelBtn = document.getElementById('progress-cancel-btn');
    if (cancelBtn) cancelBtn.style.display = isActive ? 'inline-flex' : 'none';
  }

  if (!isActive) {
    finishAndHide(s);
    return;
  }

  ensureLoop(s);
}

function startTaskInHost({ title = 'Procesando…', hostEl = null } = {}) {
  const host = ensureSlot(hostEl);
  const s = getRailState(host);
  if (!s) return { step(){}, setStage(){}, done(){} };

  const id = `${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
  const task = { progress: 0, title, stage: 'Iniciando…' };
  s.tasks.set(id, task);
  s.unlock100 = false;
  s.reported = 0;
  s.shown = 0;
  if (s.titleEl) s.titleEl.textContent = title;
  if (s.stageEl) s.stageEl.textContent = task.stage;
  if (s.pctEl) s.pctEl.textContent = '0%';
  if (s.fill) s.fill.style.width = '0%';
  refreshHost(host);

  return {
    step(frac, stage) {
      const t = s.tasks.get(id); if (!t) return;
      const clamped = clamp01(frac);
      t.progress = clamped;
      s.reported = clamped;
      if (stage) {
        t.stage = stage;
        if (s.stageEl) s.stageEl.textContent = stage;
      }
      refreshHost(host);
    },
    setStage(stage) {
      const t = s.tasks.get(id); if (!t) return;
      t.stage = stage;
      if (s.stageEl) s.stageEl.textContent = stage;
      refreshHost(host);
    },
    done() {
      if (!s.tasks.has(id)) return;
      s.tasks.delete(id);
      if (s.tasks.size === 0) {
        s.unlock100 = true;
        ensureLoop(s);
      } else {
        let last;
        for (const t of s.tasks.values()) last = t;
        if (last) {
          if (last.title && s.titleEl) s.titleEl.textContent = last.title;
          if (last.stage && s.stageEl) s.stageEl.textContent = last.stage;
          if (typeof last.progress === 'number') s.reported = clamp01(last.progress);
        }
      }
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
