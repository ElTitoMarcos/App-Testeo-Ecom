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
  const BASE_Z = 1000;

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
