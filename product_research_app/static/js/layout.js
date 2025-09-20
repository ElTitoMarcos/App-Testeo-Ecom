(function stickyOffsets(){
  const root = document.documentElement;
  const px = n => (n||0)+'px';
  function update(){
    const header = document.querySelector('.app-header');
    const progress = document.querySelector('#global-progress');
    const progressBar = progress?.querySelector('.progress-slot');
    const cancelBtn = progress?.querySelector('.btn-cancel');
    const isActive = Boolean(progress && progress.offsetParent !== null && (
      (progressBar && progressBar.classList.contains('active')) || (cancelBtn && !cancelBtn.hidden)
    ));
    const progressHeight = isActive ? progress.offsetHeight : 0;
    root.style.setProperty('--progress-h', px(progressHeight));
    const headerHeight = (header?.offsetHeight || 0) - progressHeight;
    root.style.setProperty('--topbar-h', px(headerHeight > 0 ? headerHeight : 0));
  }
  window.addEventListener('resize', update);
  new MutationObserver(update).observe(document.body, {subtree:true, attributes:true, attributeFilter:['class','style','hidden']});
  requestAnimationFrame(update);
})();
