(function(){
  let overlay;

  function ensureModal(){
    if(overlay) return overlay;
    const root = window.ensureOverlayRoot ? window.ensureOverlayRoot() : document.body;
    overlay = document.createElement('div');
    overlay.id = 'manageGroupsModal';
    overlay.className = 'modal-overlay hidden';
    overlay.innerHTML = `
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="mgTitle">
        <header class="modal-header">
          <h2 id="mgTitle">Gestionar grupos</h2>
          <span class="modal-subtitle" id="mgCount"></span>
          <button class="modal-close" id="mgClose" aria-label="Cerrar">✕</button>
        </header>
        <div class="modal-body">
          <input type="text" id="mgSearch" placeholder="Buscar grupo…" />
          <div id="mgList" class="group-list"></div>
        </div>
        <footer class="modal-footer">
          <button id="mgCreate">Crear grupo…</button>
        </footer>
      </div>`;
    root.appendChild(overlay);

    overlay.addEventListener('click', e => { if(e.target === overlay) close(); });
    overlay.querySelector('#mgClose').addEventListener('click', close);
    overlay.querySelector('#mgSearch').addEventListener('input', e => renderList(e.target.value.toLowerCase()));
    overlay.querySelector('#mgCreate').addEventListener('click', async () => {
      const name = await promptDialog('Crear grupo', 'Nombre del grupo');
      if(!name) return;
      try {
        await fetchJson('/create_list', {method:'POST', body: JSON.stringify({name})});
        await loadLists();
        renderList('');
        toast.success(`Grupo "${name}" creado`);
        document.dispatchEvent(new CustomEvent('groups-updated'));
      } catch(err){ console.error(err); toast.error('Error al crear grupo'); }
    });
    return overlay;
  }

  function trapFocus(container){
    const focusable = container.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if(!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length-1];
    container.addEventListener('keydown', e => {
      if(e.key !== 'Tab') return;
      if(e.shiftKey){
        if(document.activeElement === first){ e.preventDefault(); last.focus(); }
      }else{
        if(document.activeElement === last){ e.preventDefault(); first.focus(); }
      }
    });
  }

  function confirmDelete(name, id){
    return new Promise(res => {
      const wrap = document.createElement('div');
      wrap.className = 'confirm-overlay';
      const lists = (window.listCache || []).filter(g => g.id !== id);
      let opts = '<option value="" disabled selected>Mover a…</option>';
      lists.forEach(l => { opts += `<option value="${l.id}">${l.name}</option>`; });
      wrap.innerHTML = `<div class="confirm-box"><p>¿Eliminar grupo \"${name}\"?</p><select id="mgMoveSel">${opts}</select><div class="confirm-actions"><button class="btn-cancel">Cancelar</button><button class="btn-remove">Quitar del grupo</button><button class="btn-move" disabled>Mover y borrar</button></div></div>`;
      overlay.appendChild(wrap);
      const sel = wrap.querySelector('#mgMoveSel');
      const btnMove = wrap.querySelector('.btn-move');
      sel.addEventListener('change', () => { btnMove.disabled = !sel.value; });
      wrap.querySelector('.btn-cancel').onclick = () => { wrap.remove(); res(null); };
      wrap.querySelector('.btn-remove').onclick = () => { wrap.remove(); res({mode:'remove'}); };
      btnMove.onclick = () => { const targetId = parseInt(sel.value); wrap.remove(); res({mode:'move', targetId}); };
      wrap.querySelector('.btn-remove').focus();
    });
  }

  function renderList(filter=''){
    const lists = (window.listCache || []);
    const listEl = overlay.querySelector('#mgList');
    const countEl = overlay.querySelector('#mgCount');
    if(countEl) countEl.textContent = `${lists.length} grupos`;
    const rows = lists.filter(l => l.name.toLowerCase().includes(filter));
    if(rows.length === 0){
      listEl.innerHTML = '<p style="padding:8px;">Sin grupos</p>';
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
    listEl.querySelectorAll('.mg-del').forEach(btn => btn.addEventListener('click', handleDelete));
  }

  async function handleDelete(e){
    const btn = e.target;
    const row = btn.closest('.group-row');
    const id = parseInt(row.dataset.id);
    const name = row.querySelector('.group-name').textContent;
    const choice = await confirmDelete(name, id);
    if(!choice) return;
    btn.disabled = true;
    btn.classList.add('loading');
    try {
      await deleteGroup(id, choice);
      close();
      toast.success(`Grupo "${name}" eliminado`);
    } catch(err){ console.error(err); toast.error(`Error al eliminar grupo: ${err.message}`); btn.disabled = false; btn.classList.remove('loading'); }
  }

  function open(){
    const modal = ensureModal();
    renderList('');
    modal.classList.remove('hidden');
    requestAnimationFrame(() => modal.classList.add('open'));
    document.body.style.overflow = 'hidden';
    const search = modal.querySelector('#mgSearch');
    if(search) search.focus();
    trapFocus(modal);
    document.addEventListener('keydown', escHandler);
  }

  function close(){
    if(!overlay) return;
    overlay.classList.remove('open');
    setTimeout(() => overlay.classList.add('hidden'), 120);
    document.body.style.overflow = '';
    document.removeEventListener('keydown', escHandler);
  }

  function escHandler(e){ if(e.key === 'Escape') close(); }

  window.openManageGroups = open;
  const btn = document.getElementById('btnManageGroups');
  if(btn) btn.addEventListener('click', open);
})();

