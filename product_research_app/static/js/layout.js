// ------- PROGRESO -------
const piHost  = document.getElementById('progress-info');
const piFill  = piHost?.querySelector('.pi-fill');
const piPct   = piHost?.querySelector('.pi-pct');
const piPhase = piHost?.querySelector('.pi-phase');
const piSub   = piHost?.querySelector('.pi-sub');
const piBtn   = document.getElementById('btn-cancel-import');

function piShow(v){ if(piHost) piHost.hidden = !v; }
function piSet(pct, phase, sub){
  if(typeof pct === 'number'){
    const p = Math.max(0, Math.min(100, pct));
    if(piFill) piFill.style.width = p + '%';
    if(piPct)  piPct.textContent = p + '%';
  }
  if(phase && piPhase) piPhase.textContent = phase;
  if(sub   && piSub)   piSub.textContent   = sub;
}

// Hooks del importador (conecta con tu flujo existente)
window.onImportStart = (taskId)=>{
  window.currentTaskId = taskId;
  piSet(0,'Importando catálogo','Preparando…');
  piShow(true);
};
window.onImportTick  = (pct, phase, sub)=>{ piSet(pct ?? 0, phase, sub); };
window.onImportEnd   = ()=>{ piShow(false); };

piBtn?.addEventListener('click', async ()=>{
  if(!window.currentTaskId) return;
  piBtn.disabled = true;
  try{
    await fetch('/_import_cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:window.currentTaskId})});
  }catch(e){}
  window.stopImportPolling?.();
  piSet(100,'Cancelado','');
  setTimeout(()=>piShow(false), 600);
  piBtn.disabled = false;
});

// ------- OFFSET STICKY -------
(function stickyOffset(){
  const root = document.documentElement, px = n => (n||0)+'px';
  function recalc(){
    const topbar  = document.querySelector('.topbar');
    const toolbar = document.querySelector('.toolbar'); // fila de filtros/botones
    root.style.setProperty('--topbar-h',  px(topbar?.offsetHeight || 0));
    root.style.setProperty('--toolbar-h', px(toolbar?.offsetHeight || 0));
  }
  window.addEventListener('resize', recalc);
  new MutationObserver(recalc).observe(document.body,{subtree:true,attributes:true,attributeFilter:['class','style','hidden']});
  requestAnimationFrame(recalc);
})();
