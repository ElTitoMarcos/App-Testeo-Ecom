const selection = new Set();
let currentPageIds = [];
const master = document.getElementById('selectAll');
let bottomBar = null;

import('./format.js').then(m => {
  window.abbr = m.abbr;
  window.scoreClass = m.scoreClass;
});

function updateMasterState(){
  const selectedOnPage = currentPageIds.filter(id => selection.has(id)).length;
  master.indeterminate = selectedOnPage>0 && selectedOnPage<currentPageIds.length;
  master.checked = selectedOnPage===currentPageIds.length && currentPageIds.length>0;
  document.getElementById('btnDelete').disabled = selection.size===0;
  document.getElementById('btnExport').disabled = selection.size===0;
  if(bottomBar){
    if(selection.size>0){
      bottomBar.classList.remove('hidden');
      document.getElementById('selCount').textContent = `${selection.size} seleccionados`;
    }else{
      bottomBar.classList.add('hidden');
    }
  }
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

const table = document.getElementById('productTable');
if(table){
  bottomBar = document.createElement('div');
  bottomBar.id = 'bottomBar';
  bottomBar.className = 'bottombar hidden';
  bottomBar.innerHTML = '<div id="selCount"></div><div><button id="bbDelete">Eliminar</button><button id="bbExport">Exportar</button><button id="bbAddGroup">Añadir a grupo</button></div>';
  table.parentElement.appendChild(bottomBar);
  document.getElementById('bbDelete').addEventListener('click', ()=>document.getElementById('btnDelete').click());
  document.getElementById('bbExport').addEventListener('click', ()=>document.getElementById('btnExport').click());
  document.getElementById('bbAddGroup').addEventListener('click', ()=>document.getElementById('btnAddToGroup').click());
}
