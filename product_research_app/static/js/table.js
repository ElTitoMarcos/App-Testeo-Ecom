const selection = new Set();
const EC_BA_MAX_BATCH = 100;
let batchWarned = false;
let currentPageIds = [];
let master = null;
const bottomBar = document.getElementById('bottomBar');

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
  const disable = selection.size===0;
  const btnDel = document.getElementById('btnDelete');
  const btnExp = document.getElementById('btnExport');
  const btnAdd = document.getElementById('btnAddToGroup');
  const btnBa = document.getElementById('btn-ba-gpt');
  const btnBaBatch = document.getElementById('btn-ba-batch');
  if(btnDel) btnDel.disabled = disable;
  if(btnExp) btnExp.disabled = disable;
  if(btnAdd) btnAdd.disabled = disable;
  if(btnBa) btnBa.disabled = disable;
  if(btnBaBatch) btnBaBatch.disabled = disable;
  if(selection.size > EC_BA_MAX_BATCH){
    if(!batchWarned){ toast.info(`Máximo ${EC_BA_MAX_BATCH} productos por lote`); batchWarned=true; }
  } else { batchWarned=false; }
  if(bottomBar){
    const selEl = document.getElementById('selCount');
    if(selEl) selEl.textContent = `${selection.size} seleccionados`;
    bottomBar.classList.toggle('hidden', disable);
    if(!disable){
      document.body.style.paddingBottom = bottomBar.offsetHeight + 'px';
    } else {
      document.body.style.paddingBottom = '';
    }
  }
}

function firesFor(score0to5){
  const n = Math.max(0, Math.min(5, Math.round(score0to5 || 0)));
  return '🔥'.repeat(n);
}
const table = document.getElementById('productTable');
if(bottomBar){
  const legendBtn = document.getElementById('legendBtn');
  const legendPop = document.getElementById('legendPop');
  if(legendBtn && legendPop){
    const overlay = window.ensureOverlayRoot ? window.ensureOverlayRoot() : document.body;
    if(legendPop.parentNode !== overlay){ overlay.appendChild(legendPop); }
    legendBtn.addEventListener('click', ()=>legendPop.classList.toggle('hidden'));
    document.addEventListener('click', (e)=>{
      if(!legendPop.contains(e.target) && e.target!==legendBtn) legendPop.classList.add('hidden');
    });
  }
}

const tbody = table ? table.querySelector('tbody') : null;
let lastClickedCheck = null;

if (tbody) {
  tbody.addEventListener('click', (e) => {
    const checkbox = e.target.closest('input.rowCheck');
    if (checkbox) {
      if (e.shiftKey && lastClickedCheck) {
        e.preventDefault();
        const boxes = Array.from(tbody.querySelectorAll('input.rowCheck'));
        const start = boxes.indexOf(checkbox);
        const end = boxes.indexOf(lastClickedCheck);
        const [min, max] = start < end ? [start, end] : [end, start];
        const state = lastClickedCheck.checked;
        for (let i = min; i <= max; i++) {
          boxes[i].checked = state;
          boxes[i].dispatchEvent(new Event('change'));
        }
      }
      lastClickedCheck = checkbox;
      return;
    }

    if (e.target.closest('button, a, input, select, textarea, label, img')) return;

    const row = e.target.closest('tr');
    if (!row) return;
    const cb = row.querySelector('input.rowCheck');
    if (!cb) return;

    if (e.shiftKey && lastClickedCheck) {
      const boxes = Array.from(tbody.querySelectorAll('input.rowCheck'));
      const start = boxes.indexOf(cb);
      const end = boxes.indexOf(lastClickedCheck);
      const [min, max] = start < end ? [start, end] : [end, start];
      const state = lastClickedCheck.checked;
      for (let i = min; i <= max; i++) {
        boxes[i].checked = state;
        boxes[i].dispatchEvent(new Event('change'));
      }
    } else {
      cb.checked = !cb.checked;
      cb.dispatchEvent(new Event('change'));
    }
    lastClickedCheck = cb;
  });
}
