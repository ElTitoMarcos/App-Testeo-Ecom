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

// ==== E2E progress (frontend-only) =========================================
const E2E_CFG = {
  importWeight: 0.80,     // tramo visual hasta ~80% para import
  iaWeight: 0.20,         // 20% final para IA
  iaPollMs: 4500,         // sondeo suave para estimar IA
  iaPollTimeoutMs: 12 * 60 * 1000, // tope de espera
  iaStableHits: 2,        // confirmación de "todo completo" 2 veces seguidas
};

const E2E = {
  enabled: true,
  // import
  importFrac: 0,          // 0..1 (alimentado por la barra ya existente)
  // IA observada
  iaFracObserved: 0,      // 0..1 (estimado por /products)
  iaWatcher: null,
  iaWatchStart: 0,
  iaStableCount: 0,
  // animación suave
  uiFrac: 0,              // 0..1
  targetFrac: 0,          // 0..1
  raf: 0,
  // metadata UI
  title: 'Proceso',
  stage: 'Procesando…',
};

function clamp01(v) {
  const num = Number(v);
  if (!Number.isFinite(num)) return 0;
  return Math.max(0, Math.min(1, num));
}

function __resetE2EState({ enabled = false } = {}) {
  if (E2E.raf) {
    cancelAnimationFrame(E2E.raf);
    E2E.raf = 0;
  }
  if (E2E.iaWatcher) {
    clearInterval(E2E.iaWatcher);
    E2E.iaWatcher = null;
  }
  E2E.importFrac = 0;
  E2E.iaFracObserved = 0;
  E2E.iaWatchStart = 0;
  E2E.iaStableCount = 0;
  E2E.uiFrac = 0;
  E2E.targetFrac = 0;
  E2E.enabled = enabled;
  E2E.title = 'Proceso';
  E2E.stage = 'Procesando…';
}

function __prepareE2EForImport(title) {
  __resetE2EState({ enabled: true });
  if (title) {
    E2E.title = title;
  }
  E2E.stage = 'Iniciando…';
  __animateTowards(0, false);
}

// ==== skip helpers =========================================================
function shouldSkipByUrlAndMethod(url, method) {
  const m = (method || 'GET').toUpperCase();
  if (m === 'GET') return true; // GET no suma progreso
  const u = String(url || '');
  if (/\b\/_import_status\b/i.test(u)) return true; // tampoco el poll de estado
  return false;
}

function __setImportFracForE2E(frac) {
  const f = Math.max(0, Math.min(1, Number(frac) || 0));
  E2E.importFrac = f;
  __bumpTarget();
  __startIaWatcherIfNeeded();
}

__resetE2EState({ enabled: false });

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
    __prepareE2EForImport(title || 'Proceso');
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
        __setImportFracForE2E(frac);
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
      if (isImport) {
        __startIaWatcherIfNeeded();
      }
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
    const host = initObj.__hostEl || null;
    const t = skipCounting ? null : startTaskInHost({ title: 'Cargando datos', hostEl: host });
    try {
      return await origFetch(input, initObj);
    } finally {
      t?.done();
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
    const host = this.__hostEl || null;
    const t = skipCounting ? null : startTaskInHost({ title: 'Comunicando…', hostEl: host });
    let settled = false;
    const end = () => {
      if (settled) return;
      settled = true;
      t?.done();
    };
    this.addEventListener('loadend', end);
    this.addEventListener('error', end);
    this.addEventListener('abort', end);
    try { return _send.apply(this, arguments); }
    catch (e) { end(); throw e; }
  };
})();

async function __fetchProductsLight() {
  // GET con __skipLoadingHook para no generar tareas ni mover la barra
  const res = await fetch('/products', { method:'GET', __skipLoadingHook: true });
  if (!res.ok) throw new Error('products fetch failed');
  return res.json();
}

// Define qué significa "IA completa" (ajusta campos si tu app usa otros)
const __IA_MAG = new Set(['Low','Medium','High']);
const __IA_AWARE = new Set(['Unaware','Problem-Aware','Solution-Aware','Product-Aware','Most Aware']);
const __IA_COMP = new Set(['Low','Medium','High']);
function __aiFieldsMissing(p) {
  const desire = (p && (p.desire ?? '')).toString().trim();
  const mag    = p && p.desire_magnitude;
  const aware  = p && p.awareness_level;
  const comp   = p && p.competition_level;
  const okDes  = desire.length > 0;
  const okMag  = __IA_MAG.has(mag);
  const okAw   = __IA_AWARE.has(aware);
  const okComp = __IA_COMP.has(comp);
  return !(okDes && okMag && okAw && okComp);
}

function __mapToGlobalFrac(importFrac, iaFracObserved) {
  // Import ocupa importWeight, IA el resto
  const a = (importFrac || 0) * E2E_CFG.importWeight;
  const b = (iaFracObserved || 0) * E2E_CFG.iaWeight;
  // nunca pasamos de 99.5% sin confirmación final
  return Math.min(0.995, a + b);
}

function __animateTowards(target, doneFlag) {
  if (E2E.raf) cancelAnimationFrame(E2E.raf);
  const step = () => {
    const delta = target - E2E.uiFrac;
    if (Math.abs(delta) < 0.0025) {
      E2E.uiFrac = target;
    } else {
      const inc = Math.max(0.003, Math.min(0.012, Math.abs(delta) * 0.12));
      E2E.uiFrac += Math.sign(delta) * inc;
    }
    try { LoadingHelpers.peek?.(E2E.uiFrac, doneFlag ? 'Completado' : undefined); } catch (_) {}
    if (doneFlag && Math.abs(1 - E2E.uiFrac) < 0.003) {
      try { LoadingHelpers.finish?.(); } catch (_) {}
      E2E.enabled = false;
      E2E.raf = 0;
      return;
    }
    E2E.raf = requestAnimationFrame(step);
  };
  E2E.raf = requestAnimationFrame(step);
}

function __bumpTarget() {
  // Calcula el objetivo en base a import + IA observada
  const target = __mapToGlobalFrac(E2E.importFrac, E2E.iaFracObserved);
  E2E.targetFrac = Math.max(E2E.targetFrac, target);
  __animateTowards(E2E.targetFrac, false);
}

async function __runIaWatcherOnce() {
  try {
    const data = await __fetchProductsLight();
    const total = Array.isArray(data) ? data.length : 0;
    if (!total) return; // sin datos, no movemos nada
    let missing = 0;
    for (const p of data) { if (__aiFieldsMissing(p)) missing++; }
    const done = total - missing;
    const fracLocal = done / total;         // 0..1 (solo IA)
    // Suaviza subidas y “acumula” el mejor valor observado
    E2E.iaFracObserved = Math.max(E2E.iaFracObserved, fracLocal);
    __bumpTarget();

    if (missing === 0) {
      E2E.iaStableCount++;
    } else {
      E2E.iaStableCount = 0;
    }
  } catch (_e) {
    // ignora errores de red puntuales
  }
}

function __startIaWatcherIfNeeded() {
  // Arranca cuando import llega a 100% (o casi)
  if (E2E.iaWatcher || !E2E.enabled) return;
  if (E2E.importFrac < 0.999) return;
  E2E.iaWatchStart = Date.now();
  E2E.iaWatcher = setInterval(async () => {
    const elapsed = Date.now() - E2E.iaWatchStart;
    if (elapsed > E2E_CFG.iaPollTimeoutMs) {
      // damos por terminado visualmente (pero sin “victoria falsa” antes de tiempo)
      __animateTowards(0.999, false);
      clearInterval(E2E.iaWatcher); E2E.iaWatcher = null;
      return;
    }
    await __runIaWatcherOnce();
    // Confirmación: dos ticks seguidos con 0 pendientes
    if (E2E.iaStableCount >= E2E_CFG.iaStableHits) {
      clearInterval(E2E.iaWatcher); E2E.iaWatcher = null;
      // Empuje a 100% y cierre
      __animateTowards(1, true);
    }
  }, E2E_CFG.iaPollMs);
}
