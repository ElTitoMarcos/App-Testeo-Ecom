const selection = new Set();
let currentPageIds = [];
const master = document.getElementById('selectAll');
function updateMasterState(){
  const selectedOnPage = currentPageIds.filter(id => selection.has(id)).length;
  master.indeterminate = selectedOnPage>0 && selectedOnPage<currentPageIds.length;
  master.checked = selectedOnPage===currentPageIds.length && currentPageIds.length>0;
  document.getElementById('btnDelete').disabled = selection.size===0;
  document.getElementById('btnExport').disabled = selection.size===0;
}
master.addEventListener('change', ()=>{
  if(master.checked){ currentPageIds.forEach(id=>selection.add(id)); }
  else { currentPageIds.forEach(id=>selection.delete(id)); }
  renderTable();
  updateMasterState();
});

function firesFor(score0to5){
  const n = Math.max(0, Math.min(5, Math.round(score0to5 || 0)));
  return '🔥'.repeat(n);
}

const legendBtn = document.getElementById('legendBtn');
const legendPop = document.getElementById('legendPop');
if(legendBtn && legendPop){
  legendBtn.addEventListener('click', ()=>legendPop.classList.toggle('hidden'));
  document.addEventListener('click',(e)=>{ if(!legendPop.contains(e.target) && e.target!==legendBtn) legendPop.classList.add('hidden'); });
}
