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
