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
  const disable = selection.size===0;
  const btnDel = document.getElementById('btnDelete');
  const btnExp = document.getElementById('btnExport');
  const btnAdd = document.getElementById('btnAddToGroup');
  if(btnDel) btnDel.disabled = disable;
  if(btnExp) btnExp.disabled = disable;
  if(btnAdd) btnAdd.disabled = disable;
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
function getSelectedIds() {
  const cbs = document.querySelectorAll('tbody input[type="checkbox"]:checked');
  const ids = [];
  cbs.forEach(cb => {
    const tr = cb.closest('tr');
    const idFromData = cb.dataset.id || tr?.dataset.id;
    const idFromCell = tr?.querySelector('td:nth-child(1)')?.innerText?.trim();
    const id = String(idFromData || idFromCell || "");
    if (id) ids.push(id);
  });
  return ids;
}

function showSpinner(flag) {
  const sp = document.getElementById('aiSpinner');
  if (sp) sp.style.display = flag ? 'inline-block' : 'none';
}

function showToastMsg(msg) {
  if (window.toast?.info) window.toast.info(msg);
  else alert(msg);
}

function findRowById(id) {
  const rows = document.querySelectorAll('#productTable tbody tr');
  for (const tr of rows) {
    const idFromData = tr.dataset.id || tr.querySelector('input[type="checkbox"]')?.dataset.id;
    const idFromCell = tr.querySelector('td:nth-child(1)')?.innerText?.trim();
    if (String(idFromData || idFromCell) === String(id)) return tr;
  }
  return null;
}

function getCell(tr, headerName) {
  const ths = document.querySelectorAll('#productTable thead th');
  let index = -1;
  ths.forEach((th, i) => {
    if (th.textContent.trim() === headerName) index = i;
  });
  if (index >= 0) return tr.children[index];
  return null;
}

function setRowValue(tr, headerName, value) {
  const cell = getCell(tr, headerName);
  if (!cell) return;
  const input = cell.querySelector('input, textarea');
  if (input) input.value = value;
  else cell.textContent = value;
}

function setRowSelect(tr, headerName, value) {
  const cell = getCell(tr, headerName);
  if (!cell) return;
  const select = cell.querySelector('select');
  if (select) {
    const target = String(value || '').toLowerCase();
    Array.from(select.options).forEach(opt => {
      if (opt.value.toLowerCase() === target || opt.text.toLowerCase() === target) {
        select.value = opt.value;
      }
    });
  } else {
    setRowValue(tr, headerName, value);
  }
}

async function onGenerateWithAI() {
  const btn = document.getElementById('btn-ai-generate');
  const ids = getSelectedIds();
  if (!ids.length) { showToastMsg('Selecciona al menos un producto'); return; }
  try {
    btn.disabled = true; showSpinner(true);
    const res = await fetch('/api/desire', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ productIds: ids })
    });
    const data = await res.json();
    if (!res.ok || !data?.ok) throw new Error('Error en /api/desire');
    let updatedInPlace = false;
    for (const r of data.results || []) {
      if (r.ok && r.analysis) {
        const tr = findRowById(r.id);
        if (tr) {
          setRowValue(tr, 'Desire', r.analysis.desire);
          setRowSelect(tr, 'Desire Magnitude', r.analysis.desireMagnitude);
          setRowSelect(tr, 'Awareness Level', r.analysis.awarenessLevel);
          setRowSelect(tr, 'Competition Level', r.analysis.competitionLevel);
          updatedInPlace = true;
        }
      } else if (r.error) {
        showToastMsg(`ID ${r.id}: ${r.error}`);
      }
    }
    if (!updatedInPlace) location.reload();
    else showToastMsg('Generación completada ✅');
  } catch (e) {
    showToastMsg('Error llamando a /api/desire');
    console.error(e);
  } finally {
    showSpinner(false); btn.disabled = false;
  }
}

document.getElementById('btn-ai-generate')?.addEventListener('click', onGenerateWithAI);
