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

// Config
const FINAL_CAP = 0.97;

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
  state = {
    rail,
    fill,
    pctEl,
    titleEl,
    stageEl,
    tasks: new Map(),
    hideTimer: null,
    shown: 0,
    unlock100: false,
    lastBumpAt: Date.now(),
    lastRaw: 0,
    raf: null,
  };
  paintProgress(state, 0);
  Rails.set(host, state);
  return state;
}

function paintProgress(s, frac) {
  const pct = Math.max(0, Math.min(1, frac));
  s.fill.style.transform = `translateX(${(-1 + pct) * 100}%)`;
  if (s.pctEl) s.pctEl.textContent = `${Math.round(pct * 100)}%`;
}

function refreshHost(host) {
  const s = getRailState(host); if (!s) return;
  const tasks = s.tasks;
  const hasTasks = tasks.size > 0;

  let raw = 0;
  if (hasTasks) {
    let sum = 0;
    for (const t of tasks.values()) sum += (t.progress || 0);
    raw = sum / tasks.size;
  }

  if (raw > s.lastRaw + 0.0005) s.lastBumpAt = Date.now();
  s.lastRaw = raw;

  if (!hasTasks && !s.unlock100) s.unlock100 = true;

  const shouldShow = hasTasks || s.unlock100;

  host.classList.toggle('active', shouldShow);

  const isGlobalHost = host && host.id === 'progress-slot-global';
  if (isGlobalHost) {
    const wrapper = document.getElementById('global-progress-wrapper');
    if (wrapper) wrapper.classList.toggle('show', shouldShow);
    const cancelBtn = document.getElementById('progress-cancel-btn');
    if (cancelBtn) cancelBtn.style.display = hasTasks ? 'inline-flex' : 'none';
  }

  const target = s.unlock100 ? 1 : Math.min(raw, FINAL_CAP);

  s.shown = s.shown + (target - s.shown) * 0.25;
  paintProgress(s, s.shown);

  if (hasTasks) {
    // Actualizar título/etapa desde la tarea más reciente
    let last;
    for (const t of tasks.values()) last = t;
    if (last) {
      if (last.title && s.titleEl) s.titleEl.textContent = last.title;
      if (last.stage && s.stageEl) s.stageEl.textContent = last.stage;
    }
  }

  if (s.stageEl && !s.unlock100 && Date.now() - s.lastBumpAt > 20000) {
    s.stageEl.textContent = 'Últimos detalles…';
  }

  if (s.unlock100 && s.shown >= 0.999) {
    clearTimeout(s.hideTimer);
    s.hideTimer = setTimeout(() => {
      if (s.raf) {
        cancelAnimationFrame(s.raf);
        s.raf = null;
      }
      s.shown = 0;
      s.unlock100 = false;
      paintProgress(s, 0);
      host.classList.remove('active');
      if (isGlobalHost) {
        const wrapper = document.getElementById('global-progress-wrapper');
        if (wrapper) wrapper.classList.remove('show');
        const cancelBtn = document.getElementById('progress-cancel-btn');
        if (cancelBtn) cancelBtn.style.display = 'none';
      }
      s.lastRaw = 0;
    }, 800);
  } else if (!s.unlock100) {
    clearTimeout(s.hideTimer);
    s.hideTimer = null;
  }

  const delta = Math.abs(target - s.shown);
  if (delta > 0.001 || (s.unlock100 && s.shown < 0.999)) {
    if (!s.raf) {
      s.raf = requestAnimationFrame(() => {
        s.raf = null;
        refreshHost(host);
      });
    }
  } else if (s.raf) {
    cancelAnimationFrame(s.raf);
    s.raf = null;
  }
}

function startTaskInHost({ title = 'Procesando…', hostEl = null } = {}) {
  const host = ensureSlot(hostEl);
  const s = getRailState(host);
  if (!s) return { step(){}, setStage(){}, done(){} };

  const id = `${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
  clearTimeout(s.hideTimer);
  s.hideTimer = null;
  s.unlock100 = false;
  s.lastBumpAt = Date.now();
  s.lastRaw = 0;
  if (s.raf) {
    cancelAnimationFrame(s.raf);
    s.raf = null;
  }
  s.tasks.set(id, { progress: 0, title, stage: 'Iniciando…' });
  if (s.titleEl) s.titleEl.textContent = title;
  if (s.stageEl) s.stageEl.textContent = 'Iniciando…';
  refreshHost(host);

  return {
    step(frac, stage) {
      const t = s.tasks.get(id); if (!t) return;
      t.progress = Math.max(0, Math.min(1, frac));
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
      s.tasks.delete(id);
      if (s.tasks.size === 0) s.unlock100 = true;
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
