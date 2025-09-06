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
        e.stopPropagation();
        e.preventDefault();
        const row = e.target.closest('.mg-row');
        const id = parseInt(row.dataset.id);
        const count = parseInt(row.dataset.count);
        const name = row.querySelector('span').textContent;
        const prevHtml = btn.innerHTML;
        row.style.pointerEvents = 'none';
        btn.disabled = true;
        btn.innerHTML = '⏳';
        try{
          let mode = 'remove';
          let target = null;
          if(count > 0){
            const move = confirm('Mover productos a otro grupo? Cancelar para quitar');
            if(move){
              const others = (window.listCache||[]).filter(g=>g.id!==id);
              if(!others.length){ toast.info('No hay grupo destino'); throw new Error('no target'); }
              const opt = prompt('ID del grupo destino:\n'+ others.map(g=>`${g.id}: ${g.name}`).join('\n'));
              if(!opt) throw new Error('cancel');
              target = parseInt(opt);
              mode = 'move';
            }
          }else if(!confirm('Eliminar grupo vacío?')){ throw new Error('cancel'); }
          await deleteGroup(id, {mode, target});
          dlg.classList.add('hidden');
          toast.success(`Grupo "${name}" eliminado`);
        }catch(err){
          if(err.message!=='cancel') console.error(err);
          row.style.pointerEvents = '';
          btn.disabled = false;
          btn.innerHTML = prevHtml;
        }
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
  const btn = document.getElementById('btnManageGroups');
  if(btn) btn.addEventListener('click', openManageGroups);
  document.addEventListener('keydown', e => {
    if(e.altKey && e.key.toLowerCase() === 'g' && !['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)){
      e.preventDefault();
      openManageGroups();
    }
  });
})();
