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
