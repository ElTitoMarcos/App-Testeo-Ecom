(function updateStickyOffsets(){
  const root = document.documentElement;
  const px = n => (n || 0) + 'px';
  function recalc(){
    const topbar = document.querySelector('.topbar');
    const toolbar = document.querySelector('.toolbar, .filters-row, .controls-row');
    const progress = document.getElementById('global-progress');
    const topH = topbar ? topbar.offsetHeight : 0;
    const toolH = toolbar ? toolbar.offsetHeight : 0;
    const progH = (progress && !progress.hidden) ? progress.offsetHeight : 0;
    root.style.setProperty('--topbar-h', px(topH));
    root.style.setProperty('--toolbar-h', px(toolH));
    root.style.setProperty('--progress-h', px(progH));
  }
  window.addEventListener('resize', recalc);
  new MutationObserver(recalc).observe(document.body, {subtree:true, attributes:true, attributeFilter:['style','class','hidden']});
  requestAnimationFrame(recalc);
})();
