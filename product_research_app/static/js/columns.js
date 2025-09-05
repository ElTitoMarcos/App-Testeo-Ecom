// Column visibility manager
(function(){
  const btn = document.getElementById('btnColumns');
  const panel = document.getElementById('columnsPanel');
  if(!btn || !panel) return;

  function loadState(){
    try{ return JSON.parse(localStorage.columnsVisibility || '{}'); }catch(e){ return {}; }
  }
  function saveState(state){
    localStorage.columnsVisibility = JSON.stringify(state);
  }

  function apply(){
    const state = loadState();
    document.querySelectorAll('th[data-key], td[data-key]').forEach(el => {
      const key = el.dataset.key;
      const visible = state[key] !== false;
      el.classList.toggle('col-hidden', !visible);
    });
    panel.querySelectorAll('input[type="checkbox"][data-key]').forEach(cb => {
      const key = cb.dataset.key;
      cb.checked = state[key] !== false;
    });
  }

  function build(){
    panel.innerHTML = '';
    const state = loadState();
    document.querySelectorAll('th[data-key]').forEach(th => {
      const key = th.dataset.key;
      const id = 'col-vis-' + key;
      const div = document.createElement('div');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.id = id;
      cb.dataset.key = key;
      cb.checked = state[key] !== false;
      const label = document.createElement('label');
      label.htmlFor = id;
      label.textContent = th.textContent || key;
      cb.addEventListener('change', () => {
        const s = loadState();
        s[key] = cb.checked;
        saveState(s);
        document.querySelectorAll(`[data-key="${key}"]`).forEach(cell => {
          cell.classList.toggle('col-hidden', !cb.checked);
        });
      });
      div.appendChild(cb);
      div.appendChild(label);
      panel.appendChild(div);
    });
    apply();
  }

  btn.addEventListener('click', () => {
    if(panel.classList.contains('hidden')){
      const rect = btn.getBoundingClientRect();
      panel.style.top = `${rect.bottom + window.scrollY}px`;
      panel.style.left = `${rect.left + window.scrollX}px`;
      panel.classList.remove('hidden');
    } else {
      panel.classList.add('hidden');
    }
  });

  document.addEventListener('click', e => {
    if(!panel.contains(e.target) && e.target !== btn){
      panel.classList.add('hidden');
    }
  });

  window.refreshColumns = build;
  window.applyColumnVisibility = apply;

  // initial attempt (will populate once table is rendered)
  build();
})();
