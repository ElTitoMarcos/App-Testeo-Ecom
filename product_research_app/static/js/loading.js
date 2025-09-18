// loading.js — progreso embebido en cabecera, sin overlay ni barra superior global
const HeaderProgress = (() => {
  let host, fill, label;
  let active = 0;
  const tasks = new Map(); // id -> progress [0..1]

  const ensure = () => {
    if (host) return;
    host = document.querySelector('#header-progress');
    if (!host) return;
    fill = host.querySelector('.hp-fill');
    label = host.querySelector('.hp-label');
  };

  const setPct = (pct) => {
    ensure(); if (!host) return;
    const p = Math.max(0, Math.min(100, Math.round(pct)));
    fill.style.width = p + '%';
    host.setAttribute('aria-valuenow', String(p));
    label.textContent = p > 0 && p < 100 ? p + '%' : '';
    host.style.opacity = p === 0 ? 0 : 1;
  };

  const refresh = (finishing=false) => {
    if (tasks.size === 0) {
      if (finishing) setPct(100);
      setTimeout(() => setPct(0), 220);
      return;
    }
    let sum = 0;
    for (const v of tasks.values()) sum += (v ?? 0);
    const avg = sum / tasks.size;
    setPct(Math.min(99, avg * 100));
  };

  const startTask = (labelTxt = 'Cargando…') => {
    ensure();
    const id = `${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
    tasks.set(id, 0);
    active++;
    if (active === 1) setPct(2); // arranque visual
    return {
      id,
      step(frac) { tasks.set(id, Math.max(0, Math.min(1, frac))); refresh(false); },
      done() { tasks.delete(id); active = Math.max(0, active - 1); refresh(true); }
    };
  };

  return { startTask };
})();

window.AppLoading = HeaderProgress;

// --- Hooks globales de red (ligeros) ---
// No estimamos porcentaje aquí; solo marcamos tarea en curso.
(() => {
  const SKIP_KEY = '__skipLoadingHook';
  const _fetch = window.fetch;
  window.fetch = async function(input, init) {
    if (init && init[SKIP_KEY]) {
      const cloned = { ...init };
      delete cloned[SKIP_KEY];
      return _fetch.call(this, input, cloned);
    }
    const t = window.AppLoading.startTask('Red');
    try { return await _fetch.call(this, input, init); }
    finally { t.done(); }
  };

  const _open = XMLHttpRequest.prototype.open;
  const _send = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function(method, url, async, user, password) {
    this.__url = url; return _open.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function(body) {
    if (this && this[SKIP_KEY]) {
      return _send.apply(this, arguments);
    }
    const t = window.AppLoading.startTask('Red');
    const end = () => t.done();
    this.addEventListener('loadend', end);
    this.addEventListener('error', end);
    this.addEventListener('abort', end);
    try { return _send.apply(this, arguments); }
    catch (e) { end(); throw e; }
  };
})();

// Helper de pasos para procesos largos (con % real)
export const LoadingHelpers = {
  start(label) { return window.AppLoading.startTask(label); }
};
