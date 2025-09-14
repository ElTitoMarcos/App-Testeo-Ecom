// Common overlay utilities
(function(){
  const OFFSET = 12; // distance from anchor
  const SAFE_MARGIN = 16; // distance from viewport edges
  window.POPOVER_OFFSET = OFFSET;
  window.POPOVER_MARGIN = SAFE_MARGIN;

  function ensureOverlayRoot(){
    let root = document.getElementById('overlay');
    if(!root){
      root = document.createElement('div');
      root.id = 'overlay';
    }
    if(document.body.lastElementChild !== root){
      document.body.appendChild(root);
    }
    return root;
  }

  window.ensureOverlayRoot = ensureOverlayRoot;
  if(document.readyState === 'loading'){
    window.addEventListener('DOMContentLoaded', ensureOverlayRoot);
  } else {
    ensureOverlayRoot();
  }
})();

// Simple modal manager with stacking
(function(){
  const stack = [];
  const BASE_Z = 2000;

  function trapFocus(container){
    const focusable = container.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if(!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length-1];
    container.addEventListener('keydown', e => {
      if(e.key !== 'Tab') return;
      if(e.shiftKey){
        if(document.activeElement === first){ e.preventDefault(); last.focus(); }
      }else{
        if(document.activeElement === last){ e.preventDefault(); first.focus(); }
      }
    });
  }

  function open(content, opts={}){
    const {onClose, returnFocus, closeOnBackdrop=true} = opts;
    const root = window.ensureOverlayRoot ? window.ensureOverlayRoot() : document.body;
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay hidden';
    overlay.style.zIndex = BASE_Z + stack.length * 10;
    overlay.appendChild(content);
    root.appendChild(overlay);

    if(stack.length){
      const prev = stack[stack.length-1];
      prev.content.setAttribute('aria-hidden', 'true');
      prev.content.inert = true;
    }

    function close(){
      document.removeEventListener('keydown', escHandler);
      overlay.removeEventListener('click', backdropHandler);
      overlay.classList.remove('open');
      setTimeout(()=>overlay.remove(), 120);
      stack.pop();
      if(stack.length){
        const prev = stack[stack.length-1];
        prev.content.removeAttribute('aria-hidden');
        prev.content.inert = false;
      } else {
        document.body.style.overflow = '';
      }
      if(returnFocus) returnFocus.focus();
      if(onClose) onClose();
    }

    function escHandler(e){ if(e.key === 'Escape') close(); }
    function backdropHandler(e){ if(e.target === overlay) close(); }

    if(closeOnBackdrop){
      overlay.addEventListener('click', backdropHandler);
    }
    document.addEventListener('keydown', escHandler);

    document.body.style.overflow = 'hidden';
    overlay.classList.remove('hidden');
    requestAnimationFrame(() => overlay.classList.add('open'));
    trapFocus(content);

    const handle = {overlay, content, close};
    stack.push(handle);
    return handle;
  }

  window.modalManager = { open };
})();

(function(){
  function confirmDialog(msg, opts={}){
    const {okText='Aceptar', cancelText='Cancelar'} = opts;
    return new Promise(res => {
      const root = window.ensureOverlayRoot ? window.ensureOverlayRoot() : document.body;
      const wrap = document.createElement('div');
      wrap.className = 'confirm-overlay';
      wrap.innerHTML = `<div class="confirm-box"><p>${msg}</p><div class="confirm-actions"><button class="btn-cancel">${cancelText}</button><button class="btn-ok">${okText}</button></div></div>`;
      root.appendChild(wrap);
      wrap.querySelector('.btn-cancel').onclick = () => { wrap.remove(); res(false); };
      wrap.querySelector('.btn-ok').onclick = () => { wrap.remove(); res(true); };
      wrap.querySelector('.btn-ok').focus();
    });
  }

  function promptDialog(title, placeholder=''){
    return new Promise(res => {
      const root = window.ensureOverlayRoot ? window.ensureOverlayRoot() : document.body;
      const wrap = document.createElement('div');
      wrap.className = 'confirm-overlay';
      wrap.innerHTML = `<div class="confirm-box"><h3>${title}</h3><input class="prompt-input" placeholder="${placeholder}"><div class="confirm-actions"><button class="btn-cancel">Cancelar</button><button class="btn-ok">Aceptar</button></div></div>`;
      root.appendChild(wrap);
      const input = wrap.querySelector('input');
      input.focus();
      wrap.querySelector('.btn-cancel').onclick = () => { wrap.remove(); res(null); };
      wrap.querySelector('.btn-ok').onclick = () => { const v = input.value.trim(); wrap.remove(); res(v || null); };
    });
  }

  window.confirmDialog = confirmDialog;
  window.promptDialog = promptDialog;
})();

// ---- helpers fecha (reusar si ya existen) ----
function toISOFromDDMMYYYY(v){ const s=(v||'').trim(); const m=s.match(/^(\d{2})\/(\d{2})\/(\d{4})$/); if(!m) return null; const[,dd,mm,yyyy]=m; return `${yyyy}-${mm}-${dd}`; }
function formatDDMMYYYY(d){ const dd=String(d.getDate()).padStart(2,'0'); const mm=String(d.getMonth()+1).padStart(2,'0'); const yyyy=d.getFullYear(); return `${dd}/${mm}/${yyyy}`; }

// ---- init fechas si están vacías ----
function initTrendDatesIfEmpty(){
  const $desde = document.querySelector('#fecha-desde');
  const $hasta = document.querySelector('#fecha-hasta');
  const today = new Date();
  const from = new Date(today); from.setDate(today.getDate() - 29);
  if ($desde && !$desde.value) $desde.value = formatDDMMYYYY(from);
  if ($hasta && !$hasta.value) $hasta.value = formatDDMMYYYY(today);
}

// ---- mostrar/ocultar vistas ----
function openTrends(){
  const trends = document.querySelector('#section-trends');
  const products = document.querySelector('#section-products');
  if (trends) trends.hidden = false;
  if (products) products.hidden = true;
  document.querySelector('#btn-ver-tendencias')?.classList.add('active');
  initTrendDatesIfEmpty();
  if (typeof fetchTrends === 'function') {
    fetchTrends();
  } else {
    // Fallback: petición mínima al endpoint para no quedar “muerto”
    (async ()=>{
      const url = new URL('/api/trends/summary', window.location.origin);
      const res = await fetch(url.toString(), { credentials:'same-origin' });
      const json = res.ok ? await res.json() : null;
      if (json && typeof renderTrends === 'function') renderTrends(json);
    })();
  }
}
function closeTrends(){
  const trends = document.querySelector('#section-trends');
  const products = document.querySelector('#section-products');
  if (trends) trends.hidden = true;
  if (products) products.hidden = false;
  document.querySelector('#btn-ver-tendencias')?.classList.remove('active');
}

// ---- delegación global: funciona aunque el botón se renderice dinámicamente ----
(function wireToggleTrends(){
  if (window.__wiredToggleTrends) return;
  window.__wiredToggleTrends = true;

  document.addEventListener('click', (e)=>{
    const el = e.target.closest('[data-action="toggle-trends"], #btn-ver-tendencias');
    if (!el) return;
    e.preventDefault(); // evita submit/navegación
    const trendsHidden = document.querySelector('#section-trends')?.hidden !== false;
    (trendsHidden ? openTrends : closeTrends)();
  });

  // Tecla Escape para salir de Tendencias
  document.addEventListener('keydown', (e)=>{
    if (e.key === 'Escape' && document.querySelector('#section-trends')?.hidden === false) {
      closeTrends();
    }
  });
})();

// Botón "Aplicar" (por si sigue sin disparar)
(function wireTrendsApply(){
  const btn = document.querySelector('#btn-aplicar-tendencias');
  if (!btn) return;
  btn.addEventListener('click', (e)=>{
    e.preventDefault();
    if (typeof fetchTrends === 'function') fetchTrends();
  });
})();
