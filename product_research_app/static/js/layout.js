(function calcStickyOffset() {
  const root = document.documentElement;
  const px = (n) => `${n || 0}px`;
  function recalc() {
    const topbar = document.querySelector('.topbar');
    const toolbar = document.querySelector('.toolbar');
    root.style.setProperty('--topbar-h', px(topbar?.offsetHeight || 0));
    root.style.setProperty('--toolbar-h', px(toolbar?.offsetHeight || 0));
  }
  window.addEventListener('resize', recalc, { passive: true });
  new MutationObserver(recalc).observe(document.body, {
    subtree: true,
    attributes: true,
    attributeFilter: ['class', 'style', 'hidden']
  });
  requestAnimationFrame(recalc);
})();
