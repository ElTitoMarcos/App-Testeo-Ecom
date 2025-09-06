(function(){
  function buildDialog(){
    const overlay = window.ensureOverlayRoot ? window.ensureOverlayRoot() : document.body;
    let dlg = document.getElementById('manageGroups');
    if(!dlg){
      dlg = document.createElement('div');
      dlg.id = 'manageGroups';
      dlg.className = 'popover hidden';
      dlg.style.maxWidth = '320px';
      overlay.appendChild(dlg);
    }
    const lists = (window.listCache || []);
    let html = '<h3>Grupos</h3>';
    html += '<div style="max-height:240px;overflow:auto;">';
    lists.forEach(l => {
      html += `<div class="mg-row" data-id="${l.id}" data-count="${l.count}" style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">`+
              `<span>${l.name}</span>`+
              `<button class="mg-del" title="Eliminar" aria-label="Eliminar" style="color:#c00;border:none;background:none;cursor:pointer;">🗑</button>`+
              `</div>`;
    });
    html += '</div>';
    html += '<div style="text-align:right;margin-top:8px;"><button id="mgClose">Cerrar</button></div>';
    dlg.innerHTML = html;
    dlg.querySelector('#mgClose').addEventListener('click', ()=> dlg.classList.add('hidden'));
    dlg.querySelectorAll('.mg-del').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const row = e.target.closest('.mg-row');
        const id = parseInt(row.dataset.id);
        const count = parseInt(row.dataset.count);
        btn.disabled = true;
        try{
          let mode = 'remove';
          let target = null;
          if(count > 0){
            const move = confirm('Mover productos a otro grupo? Cancelar para quitar');
            if(move){
              const others = (window.listCache||[]).filter(g=>g.id!==id);
              if(!others.length){ toast.info('No hay grupo destino'); btn.disabled=false; return; }
              const opt = prompt('ID del grupo destino:\n'+ others.map(g=>`${g.id}: ${g.name}`).join('\n'));
              if(!opt){ btn.disabled=false; return; }
              target = parseInt(opt);
              mode = 'move';
            }
          }else if(!confirm('Eliminar grupo vacío?')){ btn.disabled=false; return; }
          await deleteList(id, mode, target);
          buildDialog();
        }catch(err){ console.error(err); }
        btn.disabled = false;
      });
    });
    return dlg;
  }
  window.openManageGroups = function(){
    const dlg = buildDialog();
    dlg.classList.remove('hidden');
    dlg.style.visibility = 'hidden';
    dlg.scrollTop = 0;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const w = dlg.offsetWidth;
    const h = dlg.offsetHeight;
    dlg.style.left = `${(vw - w)/2}px`;
    dlg.style.top = `${(vh - h)/2}px`;
    dlg.style.visibility = '';
  }
})();
