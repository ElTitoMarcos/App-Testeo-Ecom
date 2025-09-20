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

function ensureGlobalSlot() {
  const bar = document.querySelector('#global-progress .progress-bar');
  if (!bar) return null;
  let slot = bar.querySelector('#progress-slot-global') || bar.querySelector('.progress-slot');
  if (!slot) {
    slot = document.createElement('div');
    slot.id = 'progress-slot-global';
    slot.className = 'progress-slot';
    bar.insertBefore(slot, bar.firstChild || null);
  } else if (!slot.id) {
    slot.id = 'progress-slot-global';
  }
  return slot;
}

function ensureSlot(el) {
  // Si el host es un modal o un hijo suyo, usa/crea .modal-progress-slot. Si no, usa el slot global.
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
  if (!host) host = ensureGlobalSlot();
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
  clearTimeout(s.hideTimer);
  const progressBar = host?.closest('.progress-bar') || null;
  const progressHost = progressBar?.closest('.progress-host') || null;
  const label = progressHost?.querySelector('.progress-label') || progressBar?.querySelector('.progress-label') || null;
  const tasks = s.tasks;
  if (tasks.size === 0) {
    // completar al 100% brevemente y colapsar el slot
    s.fill.style.width = '100%';
    s.pctEl.textContent = '100%';
    s.hideTimer = setTimeout(() => {
      s.fill.style.width = '0%';
      s.pctEl.textContent = '0%';
      host.classList.remove('active'); // colapsa el slot (height:0)
      progressBar?.classList.remove('is-cancelled');
      if (progressHost) progressHost.hidden = true;
      if (label) label.textContent = 'Listo';
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
  if (progressHost) progressHost.hidden = false;
  host.classList.add('active');
  progressBar?.classList.remove('is-cancelled');
  if (last) {
    if (last.title) s.titleEl.textContent = last.title;
    if (last.stage) {
      s.stageEl.textContent = last.stage;
      if (label) label.textContent = last.stage;
    } else if (label) {
      label.textContent = last.title;
    }
  } else if (label) {
    label.textContent = 'Procesando…';
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
