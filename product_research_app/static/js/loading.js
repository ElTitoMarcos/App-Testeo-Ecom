// loading.js — barra en flujo con soporte multi-host (header y modal) y limpieza de legados

// ==== [PROGRESS ORCHESTRATOR] - SOLO FRONT, SIN TOCAR BACKEND ====

// Segmentos de la barra
const PROGRESS_SEG = {
  UPLOAD: [0.00, 0.25],
  IMPORT: [0.25, 0.75],
  AI:     [0.75, 0.995],
  DONE:   1.0,
};

function clamp01(x){ return Math.max(0, Math.min(1, x)); }
function mapToSegment(f, seg){ return seg[0] + (seg[1]-seg[0]) * clamp01(f); }

class ProgressOrchestrator {
  constructor({barEl, textEl, containerEl}) {
    this.barEl = barEl;
    this.textEl = textEl;
    this.containerEl = containerEl || (barEl?.closest('[data-progress-shell]') || null);

    this.target = 0;         // progreso lógico
    this.display = 0;        // progreso mostrado (animado)
    this.maxTick = 0.02;     // velocidad máx por frame (~2%)
    this.ai = { total: 0, done: 0, seen: new Set() };

    this._raf = null;
    this._animate = this._animate.bind(this);
    this._ensureLoop();
  }

  _ensureLoop(){
    if (this._raf == null) this._raf = requestAnimationFrame(this._animate);
  }

  _animate(){
    // easing suave hacia this.target
    if (this.display < this.target) {
      const gap = this.target - this.display;
      const step = Math.min(this.maxTick, Math.max(0.002, gap * 0.12));
      this.display = Math.min(this.target, this.display + step);
      this._paint();
    }
    this._raf = requestAnimationFrame(this._animate);
  }

  _activate(){
    document.documentElement.classList.remove('progress-done');
    if (this.containerEl) this.containerEl.classList.add('is-visible');
  }

  _resetAi(){
    this.ai.total = 0;
    this.ai.done = 0;
    this.ai.seen.clear();
  }

  reset(){
    this.target = 0;
    this.display = 0;
    this._resetAi();
    if (this.barEl) this.barEl.style.width = '0%';
    if (this.textEl) this.textEl.textContent = '0%';
    if (this.containerEl) this.containerEl.classList.remove('is-visible');
    document.documentElement.classList.remove('progress-done');
  }

  _paint(){
    const pct = Math.round(this.display * 100);
    if (this.barEl)  this.barEl.style.width = pct + '%';
    if (this.textEl) this.textEl.textContent = pct + '%';
    if (this.display < 1) document.documentElement.classList.remove('progress-done');
  }

  // --- Fases ---
  setUploadProgress(f0to1){
    if (f0to1 <= 0) {
      this.reset();
    }
    this._activate();
    const p = mapToSegment(f0to1, PROGRESS_SEG.UPLOAD);
    this.target = Math.max(this.target, p);
  }
  setImportPercent(pct0to100){
    this._activate();
    const p = mapToSegment(pct0to100/100, PROGRESS_SEG.IMPORT);
    this.target = Math.max(this.target, p);
  }

  // IA: total y avances incrementales
  setAiTotal(total){
    if (!Number.isFinite(total) || total <= 0) return;
    this._activate();
    this.ai.total = total;
    // si ya llevamos 'done', re-pinta destino AI
    this._bumpAiTarget();
  }
  // Marca una fila/producto visto para no contar doble
  noteAiRowCompleted(rowId){
    if (rowId == null) return;
    this._activate();
    if (!this.ai.seen.has(rowId)) {
      this.ai.seen.add(rowId);
      this.ai.done = Math.min(this.ai.done + 1, this.ai.total || Infinity);
      this._bumpAiTarget();
    }
  }
  // Si backend emite completed/total, podemos fijar valores exactos
  setAiProgress(completed, total){
    if (Number.isFinite(total) && total > 0) {
      this._activate();
      this.ai.total = total;
    }
    if (Number.isFinite(completed)) {
      this._activate();
      this.ai.done = Math.max(this.ai.done, completed);
    }
    this._bumpAiTarget();
  }
  _bumpAiTarget(){
    const f = (this.ai.total > 0) ? this.ai.done / this.ai.total : 0;
    const p = mapToSegment(f, PROGRESS_SEG.AI);
    this.target = Math.max(this.target, p);
    if (this.ai.total > 0 && this.ai.done >= this.ai.total) this.finish();
  }

  // Cierre solo cuando IA termina
  finish(){
    this.target = PROGRESS_SEG.DONE;
    this._activate();
    // deja que la animación llegue visualmente al 100% y oculta
    setTimeout(() => {
      document.documentElement.classList.add('progress-done'); // usa esto en CSS si quieres ocultar
      if (this.containerEl) this.containerEl.classList.remove('is-visible');
    }, 500);
  }
}

// Instancia única ligada al DOM actual
window._progress = (() => {
  let barEl = document.querySelector('[data-progress-bar]') || document.getElementById('progress-bar');
  let textEl = document.querySelector('[data-progress-text]') || document.getElementById('progress-text');
  let containerEl = barEl?.closest('[data-progress-shell]') || textEl?.closest('[data-progress-shell]') || null;

  if (!barEl) {
    const mount = document.querySelector('#global-progress-wrapper') || document.body;
    let shell = mount.querySelector('[data-progress-shell]');
    if (!shell) {
      shell = document.createElement('div');
      shell.setAttribute('data-progress-shell', '');
      shell.className = 'progress-orchestrator-shell';
      shell.innerHTML = `
        <div class="progress-orchestrator-bar">
          <div class="progress-orchestrator-fill" data-progress-bar></div>
        </div>
        <span class="progress-orchestrator-text" data-progress-text>0%</span>
      `;
      mount.appendChild(shell);
    }
    containerEl = shell;
    barEl = shell.querySelector('[data-progress-bar]');
    textEl = shell.querySelector('[data-progress-text]');
  }

  return new ProgressOrchestrator({ barEl, textEl, containerEl });
})();


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
