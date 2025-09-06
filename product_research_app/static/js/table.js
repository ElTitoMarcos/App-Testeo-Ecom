const selection = new Set();
let currentPageIds = [];
const master = document.getElementById('selectAll');
const selCountEl = document.getElementById('selCount');
const tbody = document.querySelector('#productTable tbody');

import('./format.js').then(m => {
  window.abbr = m.abbr;
  window.winnerScoreClass = m.winnerScoreClass;
});

function updateMasterState(){
  const selectedOnPage = currentPageIds.filter(id => selection.has(id)).length;
  master.indeterminate = selectedOnPage>0 && selectedOnPage<currentPageIds.length;
  master.checked = selectedOnPage===currentPageIds.length && currentPageIds.length>0;
  const disable = selection.size===0;
  ['btnDelete','btnExport','btnAddToGroup'].forEach(id=>{
    const btn = document.getElementById(id);
    if(btn) btn.disabled = disable;
  });
  master.setAttribute('aria-checked', master.indeterminate ? 'mixed' : master.checked ? 'true' : 'false');
  if(selCountEl){ selCountEl.textContent = selection.size ? `${selection.size} seleccionados` : ''; }
}
master.addEventListener('change', ()=>{
  if(master.checked){ currentPageIds.forEach(id=>selection.add(String(id))); }
  else { currentPageIds.forEach(id=>selection.delete(String(id))); }
  tbody.querySelectorAll('.rowCheck').forEach(cb=>{
    cb.checked = master.checked;
    cb.closest('tr').classList.toggle('selected', master.checked);
  });
  updateMasterState();
});

tbody.addEventListener('change', e => {
  if(e.target.classList.contains('rowCheck')){
    const cb = e.target;
    const id = cb.dataset.id;
    if(cb.checked) selection.add(id); else selection.delete(id);
    cb.closest('tr').classList.toggle('selected', cb.checked);
    updateMasterState();
  }
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

updateMasterState();
