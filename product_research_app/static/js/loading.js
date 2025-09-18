// loading.js — barra gruesa en cabecera con % y mensaje de etapa
const HeaderProgress = (() => {
  let host, track, fill, pctEl, meta, titleEl, stageEl;
  const tasks = new Map(); // id -> {progress, title, stage}

  function ensure(){
    if (host) return;
    host = document.querySelector('#header-progress');
    if (!host) return;
    track = host.querySelector('.hp-track');
    fill  = host.querySelector('.hp-fill');
    pctEl = host.querySelector('.hp-percent');
    meta  = host.querySelector('.hp-meta');
    titleEl = meta.querySelector('.hp-title');
    stageEl = meta.querySelector('.hp-stage');
  }

  function setPct(p){
    ensure(); if (!host) return;
    const pct = Math.max(0, Math.min(100, Math.round(p)));
    fill.style.width = pct + '%';
    pctEl.textContent = pct + '%';
    host.setAttribute('aria-valuenow', String(pct));
    if (pct > 0 && pct < 100) host.classList.add('is-active');
    else if (pct === 100) {
      // pequeña pausa al 100% antes de ocultar
      setTimeout(() => { host.classList.remove('is-active'); fill.style.width = '0%'; pctEl.textContent = '0%'; }, 350);
    } else {
      host.classList.remove('is-active');
    }
  }

  function refresh(){
    if (tasks.size === 0) { setPct(0); return; }
    // promedio simple de progresos
    let sum = 0; let last;
    for (const t of tasks.values()){ sum += (t.progress || 0); last = t; }
    const avg = Math.min(0.99, sum / tasks.size);
    setPct(Math.round(avg * 100));
    // muestra el título/etapa del último iniciado
    if (last){
      if (last.title) titleEl.textContent = last.title;
      if (last.stage) stageEl.textContent = last.stage;
    }
  }

  function startTask(title = 'Procesando…'){
    ensure();
    const id = `${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
    tasks.set(id, { progress: 0, title, stage: 'Iniciando' });
    refresh();

    return {
      id,
      step(frac, stage){
        const t = tasks.get(id); if (!t) return;
        t.progress = Math.max(0, Math.min(1, frac));
        if (stage) { t.stage = stage; }
        refresh();
      },
      setStage(stage){
        const t = tasks.get(id); if (!t) return;
        t.stage = stage; refresh();
      },
      done(){
        tasks.delete(id);
        if (tasks.size === 0) setPct(100); else refresh();
      }
    };
  }

  return { startTask };
})();

window.AppLoading = HeaderProgress;

// Hooks de red (opcionales: marcan actividad, sin %)
(() => {
  const _fetch = window.fetch;
  window.fetch = async function(input, init){
    const t = window.AppLoading.startTask('Cargando datos');
    try { return await _fetch(input, init); }
    finally { t.done(); }
  };

  const _open = XMLHttpRequest.prototype.open;
  const _send = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function(method, url, async, user, password){
    this.__url = url; return _open.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function(body){
    const t = window.AppLoading.startTask('Comunicando…');
    const end = () => t.done();
    this.addEventListener('loadend', end);
    this.addEventListener('error', end);
    this.addEventListener('abort', end);
    try { return _send.apply(this, arguments); }
    catch(e){ end(); throw e; }
  };
})();

// Helper para tareas con % real
export const LoadingHelpers = {
  start(title){ return window.AppLoading.startTask(title); }
};
