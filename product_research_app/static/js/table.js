const selection = new Set();
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
  const noneSelected = selection.size===0;
  const btnDel = document.getElementById('btnDelete');
  const btnExp = document.getElementById('btnExport');
  const btnAdd = document.getElementById('btnAddToGroup');
  const btnGen = document.getElementById('btnGenWinner');
  if(btnDel) btnDel.disabled = noneSelected;
  if(btnExp) btnExp.disabled = noneSelected;
  if(btnAdd) btnAdd.disabled = noneSelected;
  if(btnGen){
    const ap = window.allProducts || [];
    const needs = Array.from(selection).some(id => {
      const prod = ap.find(p => String(p.id)===String(id));
      const val = prod ? Number(prod.winner_score_v2_pct) : 0;
      return !val;
    });
    btnGen.disabled = noneSelected || !needs;
  }
  if(bottomBar){
    const selEl = document.getElementById('selCount');
    if(selEl) selEl.textContent = `${selection.size} seleccionados`;
    bottomBar.classList.toggle('hidden', noneSelected);
    if(!noneSelected){
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
