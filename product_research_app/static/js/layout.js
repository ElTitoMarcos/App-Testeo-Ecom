(function calcStickyOffset(){
  const root = document.documentElement;
  const px = n => (n || 0) + 'px';
  function recalc(){
    const topbar = document.querySelector('.topbar');
    const toolbar = document.querySelector('.toolbar');
    root.style.setProperty('--topbar-h', px(topbar?.offsetHeight || 0));
    root.style.setProperty('--toolbar-h', px(toolbar?.offsetHeight || 0));
  }
  window.addEventListener('resize', recalc);
  new MutationObserver(recalc).observe(document.body, { subtree:true, attributes:true, attributeFilter:['style','class','hidden'] });
  requestAnimationFrame(recalc);
})();

const pHost = document.getElementById('inline-progress');
const pFill = pHost?.querySelector('.fill');
const pBtn = document.getElementById('btn-cancel-import');

window.currentProgressPct = 0;

function progressShow(v){
  if(pHost) pHost.hidden = !v;
}

function progressSet(pct){
  const num = Number(pct);
  const clamped = Math.max(0, Math.min(100, Number.isFinite(num) ? num : 0));
  window.currentProgressPct = clamped;
  if(pFill) pFill.style.width = clamped + '%';
}

window.progressShow = progressShow;
window.progressSet = progressSet;

window.onImportStart = taskId => {
  window.currentTaskId = taskId;
  if(pBtn) pBtn.disabled = false;
  progressSet(0);
  progressShow(true);
};

window.onImportTick = pct => {
  progressSet(pct ?? 0);
};

window.onImportEnd = () => {
  window.currentTaskId = null;
  if(pBtn) pBtn.disabled = false;
  progressShow(false);
};

pBtn?.addEventListener('click', async () => {
  if(!window.currentTaskId) return;
  pBtn.disabled = true;
  try{
    await fetch('/_import_cancel', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ task_id: window.currentTaskId })
    });
  }catch(e){}
  stopImportPolling?.();
  window.currentTaskId = null;
  progressSet(100);
  setTimeout(() => progressShow(false), 500);
  pBtn.disabled = false;
});
