import * as groupsService from './groups-service.js';

let handle;
let searchFilter = '';

function buildModal(){
  const modal = document.createElement('div');
  modal.className = 'modal';
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.innerHTML = `
    <header class="modal-header">
      <h2 id="mgTitle">Gestionar grupos</h2>
      <span class="modal-subtitle" id="mgCount"></span>
      <button class="modal-close" id="mgClose" aria-label="Cerrar">✕</button>
    </header>
    <div class="modal-body">
      <input type="text" id="mgSearch" placeholder="Buscar grupo…" />
      <div id="mgList" class="group-list" style="overflow:auto; max-height:320px;"></div>
    </div>
    <footer class="modal-footer">
      <button id="mgCreate">Crear grupo…</button>
    </footer>`;
  return modal;
}

function renderList(filter=''){
  const lists = window.listCache || [];
  const listEl = handle.content.querySelector('#mgList');
  const countEl = handle.content.querySelector('#mgCount');
  if(countEl) countEl.textContent = `${lists.length} grupos`;
  const rows = lists.filter(l => l.name.toLowerCase().includes(filter));
  const scroll = listEl.scrollTop;
  if(rows.length === 0){
    listEl.innerHTML = '<p style="padding:8px;">Sin grupos</p>';
    listEl.scrollTop = scroll;
    return;
  }
  let html = '';
  rows.forEach(l => {
    html += `<div class="group-row" data-id="${l.id}">
        <span class="group-name">${l.name}</span>
        <span class="group-count">${l.count}</span>
        <div class="group-actions"><button class="mg-del" aria-label="Eliminar">Eliminar</button></div>
      </div>`;
  });
  listEl.innerHTML = html;
  listEl.scrollTop = scroll;
}

async function handleDelete(id, name){
  const choice = await confirmDelete(name, id);
  if(!choice) return;
  const btn = handle.content.querySelector(`.group-row[data-id="${id}"] .mg-del`);
  if(btn){
    btn.disabled = true;
    btn.classList.add('loading');
  }
  try{
    await groupsService.deleteGroup(id, choice);
    await groupsService.listGroups();
    renderList(searchFilter);
    toast.success(`Grupo "${name}" eliminado`);
  }catch(err){
    console.error(err);
    toast.error(`Error al eliminar grupo: ${err.message}`);
    if(btn){ btn.disabled=false; btn.classList.remove('loading'); }
  }
}

function confirmDelete(name, id){
  return new Promise(res => {
    const wrap = document.createElement('div');
    wrap.className = 'confirm-overlay';
    const lists = (window.listCache || []).filter(g => g.id !== id);
    let opts = '<option value="" disabled selected>Mover a…</option>';
    lists.forEach(l => { opts += `<option value="${l.id}">${l.name}</option>`; });
    wrap.innerHTML = `<div class="confirm-box"><p>¿Eliminar grupo \"${name}\"?</p><select id="mgMoveSel">${opts}</select><div class="confirm-actions"><button class="btn-cancel">Cancelar</button><button class="btn-remove">Quitar del grupo</button><button class="btn-move" disabled>Mover y borrar</button></div></div>`;
    handle.overlay.appendChild(wrap);
    const sel = wrap.querySelector('#mgMoveSel');
    const btnMove = wrap.querySelector('.btn-move');
    sel.addEventListener('change', () => { btnMove.disabled = !sel.value; });
    wrap.querySelector('.btn-cancel').onclick = () => { wrap.remove(); res(null); };
    wrap.querySelector('.btn-remove').onclick = () => { wrap.remove(); res({mode:'remove'}); };
    btnMove.onclick = () => { const targetId = parseInt(sel.value); wrap.remove(); res({mode:'move', targetGroupId:targetId}); };
    wrap.querySelector('.btn-remove').focus();
  });
}

function openCreate(btn){
  const inner = document.createElement('div');
  inner.className = 'modal';
  inner.innerHTML = `
    <header class="modal-header"><h3>Crear grupo</h3><button class="modal-close" aria-label="Cerrar">✕</button></header>
    <div class="modal-body"><input type="text" id="cgName" placeholder="Nombre"/></div>
    <footer class="modal-footer"><button id="cgSave" disabled>Crear</button></footer>`;
  const child = modalManager.open(inner, {returnFocus: btn});
  const input = inner.querySelector('#cgName');
  const save = inner.querySelector('#cgSave');
  input.addEventListener('input', () => { save.disabled = !input.value.trim(); });
  inner.querySelector('.modal-close').addEventListener('click', () => child.close());
  save.addEventListener('click', async () => {
    const name = input.value.trim();
    if(!name) return;
    save.disabled = true;
    try{
      await groupsService.createGroup(name);
      await groupsService.listGroups();
      renderList(searchFilter);
      toast.success(`Grupo "${name}" creado`);
      child.close();
    }catch(err){
      console.error(err);
      toast.error('Error al crear grupo');
      save.disabled = false;
    }
  });
  input.focus();
}

export async function open(){
  await groupsService.listGroups();
  const modal = buildModal();
  handle = modalManager.open(modal, {returnFocus: triggerBtn});
  const search = modal.querySelector('#mgSearch');
  search.value = searchFilter;
  search.addEventListener('input', e => { searchFilter = e.target.value.toLowerCase(); renderList(searchFilter); });
  modal.querySelector('#mgClose').addEventListener('click', () => handle.close());
  modal.querySelector('#mgCreate').addEventListener('click', e => openCreate(e.currentTarget));
  modal.querySelector('#mgList').addEventListener('click', e => {
    const row = e.target.closest('.group-row');
    if(e.target.classList.contains('mg-del') && row){
      const id = parseInt(row.dataset.id);
      const name = row.querySelector('.group-name').textContent;
      handleDelete(id, name);
    }
  });
  renderList(searchFilter);
  modal.querySelector('#mgSearch').focus();
}

const triggerBtn = document.getElementById('btnManageGroups');
if(triggerBtn) triggerBtn.addEventListener('click', open);

function registerAction(name, fn) {
  if (typeof window === 'undefined' || !name || typeof fn !== 'function') return;
  if (typeof window.__registerAction === 'function') {
    window.__registerAction(name, fn);
    return;
  }
  const pending = window.__pendingActions || (window.__pendingActions = []);
  const exists = pending.some((entry) => {
    if (!entry) return false;
    if (typeof entry.name === 'string') return entry.name === name;
    if (Array.isArray(entry)) return entry[0] === name;
    return false;
  });
  if (!exists) pending.push({ name, fn });
}

if (typeof window !== 'undefined') {
  if (!window.openManageGroups) window.openManageGroups = open;
  registerAction('open-manage-groups', open);
}
