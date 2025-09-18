// loading.js — hook global de progreso para TODA la app (fetch + XHR + helpers de pasos)
const TopProgress = (() => {
  let root, bar, active = 0, prog = 0, timer;
  const ensure = () => {
    if (root) return;
    root = document.createElement('div');
    root.id = 'top-progress';
    bar = document.createElement('div');
    bar.className = 'bar';
    root.appendChild(bar);
    document.body.appendChild(root);
  };
  const set = (p) => { ensure(); prog = Math.max(0, Math.min(1, p)); bar.style.width = (prog * 100) + '%'; };
  const showIndeterminate = () => { ensure(); root.classList.add('indeterminate'); };
  const clearIndeterminate = () => { ensure(); root.classList.remove('indeterminate'); };
  const start = () => {
    ensure();
    active++;
    // arranque suave hacia 0.2 mientras no haya progreso real
    if (!timer) {
      timer = setInterval(() => {
        if (prog < 0.2) set(prog + 0.02);
        if (active === 0) { clearInterval(timer); timer = null; }
      }, 120);
    }
  };
  const done = () => {
    active = Math.max(0, active - 1);
    if (active === 0) {
      set(1);
      setTimeout(() => { set(0); bar.style.width = '0%'; }, 180);
      clearIndeterminate();
    }
  };
  return { set, start, done, showIndeterminate, clearIndeterminate };
})();

const Overlay = (() => {
  let el, fill, percent, title, desc;
  const ensure = () => {
    if (el) return;
    el = document.createElement('div');
    el.className = 'loading-overlay';
    el.innerHTML = `
      <div class="loading-card">
        <div class="loading-title" id="ov-title">Procesando…</div>
        <div class="loading-desc" id="ov-desc"></div>
        <div class="loading-bar"><div class="loading-fill" id="ov-fill"></div></div>
        <div class="loading-percent" id="ov-perc">0%</div>
      </div>`;
    document.body.appendChild(el);
    fill = el.querySelector('#ov-fill');
    percent = el.querySelector('#ov-perc');
    title = el.querySelector('#ov-title');
    desc = el.querySelector('#ov-desc');
  };
  const open = (t, d) => {
    ensure();
    title.textContent = t || 'Procesando…';
    desc.textContent = d || '';
    fill.style.width = '0%';
    percent.textContent = '0%';
    el.style.display = 'flex';
  };
  const update = (p, d) => {
    ensure();
    if (typeof d === 'string') desc.textContent = d;
    const v = Math.max(0, Math.min(100, Math.round(p)));
    fill.style.width = v + '%';
    percent.textContent = v + '%';
  };
  const close = () => { ensure(); el.style.display = 'none'; };
  return { open, update, close };
})();

// Estado por-botón
const withButtonLoading = (btn, text = 'Cargando…') => {
  if (!btn) return () => {};
  const prev = btn.getAttribute('data-loading-text');
  btn.classList.add('btn-loading');
  btn.setAttribute('data-loading-text', text);
  btn.disabled = true;
  return () => {
    btn.disabled = false;
    if (prev === null) btn.removeAttribute('data-loading-text');
    else btn.setAttribute('data-loading-text', prev);
    btn.classList.remove('btn-loading');
  };
};

// Gestor global
const Loading = (() => {
  // tareas en curso (para agregación sencilla)
  const tasks = new Map(); // id -> {weight, progress}
  const updateTop = () => {
    if (tasks.size === 0) return;
    let sumW = 0, sumP = 0;
    for (const {weight, progress} of tasks.values()) {
      sumW += weight || 1;
      sumP += (progress ?? 0) * (weight || 1);
    }
    const p = sumW ? (sumP / sumW) : 0;
    TopProgress.set(Math.min(0.95, p)); // reserva el 5% final para cierre
  };
  const startTask = (label, {weight = 1, determinate = false, showOverlay = false, btn} = {}) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2,7)}`;
    tasks.set(id, {label, weight, progress: 0});
    TopProgress.start();
    if (!determinate) TopProgress.showIndeterminate();
    let endBtn;
    if (btn) endBtn = withButtonLoading(btn, 'Cargando…');
    if (showOverlay) Overlay.open(label, '');

    updateTop();
    return {
      id,
      step(p, msg) {
        const t = tasks.get(id); if (!t) return;
        t.progress = Math.max(0, Math.min(1, p));
        if (showOverlay) Overlay.update(t.progress * 100, msg);
        TopProgress.clearIndeterminate();
        updateTop();
      },
      done() {
        tasks.delete(id);
        if (showOverlay) Overlay.update(100, 'Listo');
        TopProgress.done();
        if (showOverlay) setTimeout(() => Overlay.close(), 200);
        if (endBtn) endBtn();
      }
    };
  };
  return { startTask };
})();
window.AppLoading = Loading; // disponible globalmente

// --- Hook global: fetch + XHR ---
(() => {
  // FETCH
  const _fetch = window.fetch;
  window.fetch = async function(input, init = {}) {
    // botón origen (si viene)
    const btn = init && init.__buttonEl;
    const task = Loading.startTask('Cargando datos', { determinate: false, btn });
    try {
      const res = await _fetch(input, init);
      return res;
    } finally {
      task.done();
    }
  };

  // XHR (para uploads o libs que no usan fetch)
  const _open = XMLHttpRequest.prototype.open;
  const _send = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function(method, url, async, user, password) {
    this.__url = url;
    return _open.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function(body) {
    const task = Loading.startTask('Comunicando…', { determinate: false });
    const clean = () => task.done();
    this.addEventListener('loadend', clean);
    this.addEventListener('error', clean);
    this.addEventListener('abort', clean);
    try { return _send.apply(this, arguments); }
    catch (e) { clean(); throw e; }
  };
})();

// --- Helper para procesos por pasos locales (render, cálculos, etc.) ---
// Uso: const t = AppLoading.startTask('Actualizar tendencias', { determinate:true, showOverlay:true, btn:evt.currentTarget }); t.step(0.33); … t.done();
export const LoadingHelpers = {
  start(label, opts) { return window.AppLoading.startTask(label, { determinate:true, showOverlay:true, ...(opts||{}) }); }
};
