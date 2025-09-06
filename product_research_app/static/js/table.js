const selection = new Set();
let currentPageIds = [];
const master = document.getElementById('selectAll');

import('./format.js').then(m => {
  window.abbr = m.abbr;
  window.winnerScoreClass = m.winnerScoreClass;
});

function updateMasterState(){
  const selectedOnPage = currentPageIds.filter(id => selection.has(id)).length;
  master.indeterminate = selectedOnPage>0 && selectedOnPage<currentPageIds.length;
  master.checked = selectedOnPage===currentPageIds.length && currentPageIds.length>0;
  const disable = selection.size===0;
  document.getElementById('btnDelete').disabled = disable;
  document.getElementById('btnExport').disabled = disable;
  document.getElementById('btnAddToGroup').disabled = disable;
  const selCount = document.getElementById('selCount');
  if(selCount){ selCount.textContent = selection.size ? `${selection.size} seleccionados` : ''; }
}
master.addEventListener('change', ()=>{
  if(master.checked){ currentPageIds.forEach(id=>selection.add(String(id))); }
  else { currentPageIds.forEach(id=>selection.delete(String(id))); }
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
