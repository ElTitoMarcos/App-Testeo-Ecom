const selection = new Set();
let currentPageIds = [];
let master = null;
let bottomBar = null;

import('./format.js').then(m => {
  window.abbr = m.abbr;
  window.winnerScoreClass = m.winnerScoreClass;
});

function ensureMaster(){
  if(!master){
    master = document.getElementById('selectAll');
    if(master){
      master.addEventListener('change', ()=>{
        if(master.checked){ currentPageIds.forEach(id=>selection.add(String(id))); }
        else { currentPageIds.forEach(id=>selection.delete(String(id))); }
        renderTable();
        updateMasterState();
      });
    }
  }
}

function updateMasterState(){
  ensureMaster();
  if(!master) return;
  const selectedOnPage = currentPageIds.filter(id => selection.has(id)).length;
  master.indeterminate = selectedOnPage>0 && selectedOnPage<currentPageIds.length;
  master.checked = selectedOnPage===currentPageIds.length && currentPageIds.length>0;
  document.getElementById('btnDelete').disabled = selection.size===0;
  document.getElementById('btnExport').disabled = selection.size===0;
  if(bottomBar){
    document.getElementById('selCount').textContent = `${selection.size} seleccionados`;
    if(selection.size>0){
      bottomBar.classList.remove('hidden');
    }else{
      bottomBar.classList.add('hidden');
    }
  }
}

function firesFor(score0to5){
  const n = Math.max(0, Math.min(5, Math.round(score0to5 || 0)));
  return '🔥'.repeat(n);
}
const table = document.getElementById('productTable');
if(table){
  bottomBar = document.createElement('div');
  bottomBar.id = 'bottomBar';
  bottomBar.className = 'bottombar hidden';
    bottomBar.innerHTML = '<div style="display:flex; align-items:center; gap:8px;"><button id="legendBtn" class="legend-btn">ℹ️</button><span id="selCount"></span></div><div><button id="bbDelete">Eliminar</button><button id="bbExport">Exportar</button><button id="bbAddGroup">Añadir a grupo</button></div>';
  table.parentElement.appendChild(bottomBar);
    const legendBtn = document.getElementById('legendBtn');
    const legendPop = document.getElementById('legendPop');
    if(legendBtn && legendPop){
      legendBtn.addEventListener('click', ()=>legendPop.classList.toggle('hidden'));
      document.addEventListener('click',(e)=>{ if(!legendPop.contains(e.target) && e.target!==legendBtn) legendPop.classList.add('hidden'); });
    }
  document.getElementById('bbDelete').addEventListener('click', ()=>document.getElementById('btnDelete').click());
  document.getElementById('bbExport').addEventListener('click', ()=>document.getElementById('btnExport').click());
  document.getElementById('bbAddGroup').addEventListener('click', ()=>document.getElementById('btnAddToGroup').click());
}
