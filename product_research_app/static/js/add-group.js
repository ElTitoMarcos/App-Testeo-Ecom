// Popover for assigning selected items to a group
import * as groupsService from './groups-service.js';

(function(){
  const btn = document.getElementById('btnAddToGroup');
  if(!btn) return;

  const OFFSET = window.POPOVER_OFFSET || 12;
  const MARGIN = window.POPOVER_MARGIN || 16;
  const overlay = window.ensureOverlayRoot ? window.ensureOverlayRoot() : document.body;
  const pop = document.createElement('div');
  pop.id = 'addGroupPop';
  pop.className = 'popover hidden';
  pop.style.maxWidth = '280px';
  overlay.appendChild(pop);

  function buildList(filter=""){
    const lists = (window.listCache || []).filter(l => l.name.toLowerCase().includes(filter));
    let html = '<input type="text" id="grpSearch" placeholder="Buscar grupo" style="width:100%; margin-bottom:8px;">';
    html += '<div id="grpList" style="max-height:240px; overflow:auto;">';
    lists.forEach(l => { html += `<div class="grp-item" data-id="${l.id}" style="padding:4px 8px; cursor:pointer;">${l.name}</div>`; });
    html += '</div>';
    html += '<div id="grpCreate" style="padding:4px 8px; margin-top:8px; cursor:pointer; border-top:1px solid #ccc;">Crear grupo...</div>';
    pop.innerHTML = html;

    pop.querySelectorAll('.grp-item').forEach(el => {
      el.addEventListener('click', async () => {
        const id = parseInt(el.dataset.id);
        const groupName = el.textContent;
        const ids = Array.from(selection || [], Number);
        if(!ids.length){ toast.info('Selecciona productos para añadir'); return; }
        try{
          await fetchJson('/add_to_list', {method:'POST', body: JSON.stringify({id, ids})});
          toast.success(`${ids.length} añadidos a ${groupName}`);
          hide();
          loadLists();
        }catch(err){ console.error(err); toast.error('Error al añadir al grupo'); }
      });
    });
    const search = pop.querySelector('#grpSearch');
    search.addEventListener('input', e => buildList(e.target.value.toLowerCase()));
    const create = pop.querySelector('#grpCreate');
    create.addEventListener('click', async () => {
      const name = await promptDialog('Crear grupo', 'Nombre del grupo');
      if(!name) return;
      try{
        await groupsService.createGroup(name);
        buildList('');
        toast.success(`Grupo "${name}" creado`);
        loadLists();
      }catch(err){ console.error(err); toast.error('Error al crear grupo'); }
    });
    search.focus();
  }

  function show(){
    buildList('');
    pop.classList.remove('hidden');
    pop.style.visibility = 'hidden';
    pop.scrollTop = 0;
    const rect = btn.getBoundingClientRect();
    const w = pop.offsetWidth;
    const h = pop.offsetHeight;
    const vw = window.innerWidth;
    let left = rect.left;
    let top = rect.top - h - OFFSET;
    if(left + w > vw - MARGIN) left = vw - w - MARGIN;
    if(left < MARGIN) left = MARGIN;
    if(top < MARGIN) top = MARGIN;
    pop.style.left = `${left}px`;
    pop.style.top = `${top}px`;
    pop.style.maxHeight = `${rect.top - OFFSET - MARGIN}px`;
    pop.style.visibility = '';
  }
  function hide(){ pop.classList.add('hidden'); }

  btn.addEventListener('click', () => {
    if(pop.classList.contains('hidden')) show();
    else hide();
  });
  document.addEventListener('click', e => {
    if(!pop.contains(e.target) && e.target !== btn) hide();
  });
  document.addEventListener('keydown', e => {
    if(e.key === 'Escape') hide();
  });
  pop.addEventListener('wheel', e => e.stopPropagation());
  document.addEventListener('groups-updated', () => {
    if(!pop.classList.contains('hidden')) buildList(pop.querySelector('#grpSearch')?.value.toLowerCase()||'');
  });
})();
