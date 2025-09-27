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

// ==== E2E progress (solo frontend) ====
const E2E_CFG = {
  importWeight: 0.80,        // hasta ~80% con import
  iaWeight: 0.20,            // 20% final con IA
  idleMs: 1600,              // considera terminado IA si no hay POSTs en este intervalo
};

const E2E = {
  // import
  importPct: 0,              // 0..1 (alimentado por la barra existente de import)
  // ia
  iaTotal: 0,
  iaDone: 0,
  inFlightPost: 0,
  lastPostTs: 0,
  enabled: true,             // se puede desactivar si no hay import
  title: 'Proceso',
  stage: 'Procesando…',
  // animación suave
  uiPct: 0,                  // 0..1
  targetPct: 0,              // 0..1
  raf: 0,
};

function clamp01(v) {
  const num = Number(v);
  if (!Number.isFinite(num)) return 0;
  return Math.max(0, Math.min(1, num));
}

function resetE2EState({ enabled = false } = {}) {
  if (E2E.raf) {
    cancelAnimationFrame(E2E.raf);
    E2E.raf = 0;
  }
  E2E.importPct = 0;
  E2E.iaTotal = 0;
  E2E.iaDone = 0;
  E2E.inFlightPost = 0;
  E2E.lastPostTs = 0;
  E2E.uiPct = 0;
  E2E.targetPct = 0;
  E2E.title = 'Proceso';
  E2E.stage = 'Procesando…';
  E2E.enabled = enabled;
}

function prepareE2EForImport() {
  resetE2EState({ enabled: true });
  E2E.lastPostTs = Date.now();
}

function shouldSkipByUrlAndMethod(url, method) {
  const m = (method || 'GET').toUpperCase();
  if (m === 'GET') return true; // no contamos GETs en el progreso
  const u = String(url || '');
  if (/\b\/_import_status\b/i.test(u)) return true; // no sumes el poll de estado
  return false;
}

function setImportFractionSmooth(frac) {
  if (!E2E.enabled) return;
  const f = clamp01(frac || 0);
  E2E.importPct = f;
  if (f >= 1) {
    E2E.lastPostTs = Date.now();
  }
  bumpTargetPct();
}

function computeTarget() {
  if (!E2E.enabled) {
    return { pct: E2E.uiPct || 0, done: false };
  }

  const a = (E2E.importPct || 0) * E2E_CFG.importWeight;

  let iaFrac = 0;
  if (E2E.iaTotal > 0) {
    iaFrac = E2E.iaDone / Math.max(E2E.iaTotal, 1);
  } else {
    iaFrac = 0;
  }
  const b = iaFrac * E2E_CFG.iaWeight;

  const idle = (Date.now() - (E2E.lastPostTs || 0)) > E2E_CFG.idleMs && E2E.inFlightPost === 0;
  const done = (E2E.importPct >= 1 && (E2E.iaTotal === 0 ? idle : (E2E.iaDone >= E2E.iaTotal && idle)));

  return { pct: done ? 1 : Math.min(0.995, a + b), done };
}

function bumpTargetPct() {
  if (!E2E.enabled) return;
  const { pct, done } = computeTarget();
  E2E.targetPct = Math.max(E2E.targetPct, pct);
  animateE2E(done);
}

function animateE2E(doneFlag) {
  if (!E2E.enabled) return;
  if (E2E.raf) cancelAnimationFrame(E2E.raf);
  let finalizing = !!doneFlag;
  const step = () => {
    const { pct, done } = computeTarget();
    if (pct > E2E.targetPct) {
      E2E.targetPct = pct;
    }
    if (done) {
      finalizing = true;
    }

    const delta = E2E.targetPct - E2E.uiPct;
    if (Math.abs(delta) < 0.0025) {
      E2E.uiPct = E2E.targetPct;
    } else {
      const inc = Math.max(0.003, Math.min(0.012, Math.abs(delta) * 0.12));
      E2E.uiPct += Math.sign(delta) * inc;
    }

    try {
      if (finalizing) {
        E2E.stage = 'Completado';
      }
      LoadingHelpers.peek(E2E.uiPct, finalizing ? 'Completado' : undefined);
    } catch (_e) {
      // ignore
    }

    if (finalizing && Math.abs(1 - E2E.uiPct) < 0.0025) {
      try { LoadingHelpers.finish?.(); } catch (_e) {}
      E2E.enabled = false;
      E2E.raf = 0;
      return;
    }
    E2E.raf = requestAnimationFrame(step);
  };
  E2E.raf = requestAnimationFrame(step);
}

resetE2EState({ enabled: false });

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
    e2e: { active: false, progress: 0, title: 'Proceso', stage: 'Procesando…' }
  };
  Rails.set(host, state);
  return state;
}

function refreshHost(host) {
  const s = getRailState(host); if (!s) return;
  const tasks = s.tasks;
  const e2eActive = !!(s.e2e && s.e2e.active && E2E.enabled);
  if (e2eActive) {
    const frac = clamp01(s.e2e.progress || 0);
    const pct = Math.round(frac * 100);
    s.fill.style.width = pct + '%';
    s.pctEl.textContent = pct + '%';
    host.classList.add('active');
    if (s.e2e.title) s.titleEl.textContent = s.e2e.title;
    if (s.e2e.stage) s.stageEl.textContent = s.e2e.stage;
    clearTimeout(s.hideTimer);
    s.hideTimer = null;
    return;
  }
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
  const isImport = typeof title === 'string' && /importando\s+cat[aá]logo/i.test(title);
  if (isImport) {
    prepareE2EForImport();
    E2E.title = title || 'Proceso';
    E2E.stage = 'Iniciando…';
  }
  s.tasks.set(id, { progress: 0, title, stage: 'Iniciando…', isImport });
  refreshHost(host);

  return {
    step(frac, stage) {
      const t = s.tasks.get(id); if (!t) return;
      t.progress = Math.max(0, Math.min(1, frac));
      if (stage) t.stage = stage;
      refreshHost(host);
      if (t.isImport) {
        if (stage) {
          E2E.stage = stage;
        } else {
          E2E.stage = t.stage;
        }
        setImportFractionSmooth(frac);
      }
    },
    setStage(stage) {
      const t = s.tasks.get(id); if (!t) return;
      t.stage = stage; refreshHost(host);
      if (t.isImport && stage) {
        E2E.stage = stage;
      }
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
  peek(fraction, stage) {
    if (!E2E.enabled) return;
    const host = ensureSlot(null);
    const s = getRailState(host);
    if (!s) return;
    const frac = clamp01(fraction || 0);
    s.e2e.active = true;
    s.e2e.progress = frac;
    const stageText = stage || E2E.stage || s.e2e.stage;
    if (stageText) s.e2e.stage = stageText;
    s.e2e.title = E2E.title || s.e2e.title || 'Proceso';
    refreshHost(host);
  },
  finish(stage) {
    const host = ensureSlot(null);
    const s = getRailState(host);
    if (!s) return;
    const stageText = stage || E2E.stage;
    if (stageText) {
      s.stageEl.textContent = stageText;
      E2E.stage = stageText;
    }
    if (s.e2e) {
      s.e2e.active = false;
      s.e2e.progress = 1;
    }
    refreshHost(host);
  }
};

// ===== Hooks de red: si se pasa init.__hostEl, el progreso aparece en ese host; si no, en el global =====
(() => {
  const origFetch = window.fetch.bind(window);
  window.fetch = async function(input, init = {}) {
    const initObj = init || {};
    if (initObj && initObj.__skipLoadingHook) {
      return origFetch(input, initObj);
    }

    let method = initObj.method;
    let url = input;
    if (!method && typeof input === 'object' && input) {
      method = input.method;
    }
    if (typeof input === 'object' && input && input.url) {
      url = input.url;
    }
    method = (method || 'GET').toString().toUpperCase();
    const skipCounting = shouldSkipByUrlAndMethod(url, method);

    let counted = false;
    if (!skipCounting && /^(POST|PUT|PATCH)$/.test(method)) {
      counted = true;
      E2E.inFlightPost++;
      E2E.iaTotal++;
      E2E.lastPostTs = Date.now();
    }

    const host = initObj.__hostEl || null;
    const t = startTaskInHost({ title: 'Cargando datos', hostEl: host });
    try {
      return await origFetch(input, initObj);
    } finally {
      t.done();
      if (counted) {
        E2E.inFlightPost = Math.max(0, E2E.inFlightPost - 1);
        E2E.iaDone = Math.min(E2E.iaTotal, E2E.iaDone + 1);
        E2E.lastPostTs = Date.now();
        bumpTargetPct();
      }
    }
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
    const method = (this.__method || 'GET').toString();
    const url = this.__url;
    const skipCounting = shouldSkipByUrlAndMethod(url, method);
    let counted = false;
    if (!skipCounting && /^(POST|PUT|PATCH)$/i.test(method)) {
      counted = true;
      E2E.inFlightPost++;
      E2E.iaTotal++;
      E2E.lastPostTs = Date.now();
    }
    const host = this.__hostEl || null;
    const t = startTaskInHost({ title: 'Comunicando…', hostEl: host });
    let settled = false;
    const end = () => {
      if (settled) return;
      settled = true;
      t.done();
      if (counted) {
        E2E.inFlightPost = Math.max(0, E2E.inFlightPost - 1);
        E2E.iaDone = Math.min(E2E.iaTotal, E2E.iaDone + 1);
        E2E.lastPostTs = Date.now();
        bumpTargetPct();
      }
    };
    this.addEventListener('loadend', end);
    this.addEventListener('error', end);
    this.addEventListener('abort', end);
    try { return _send.apply(this, arguments); }
    catch (e) { end(); throw e; }
  };
})();
