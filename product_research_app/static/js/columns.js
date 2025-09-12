// Column visibility manager
(function(){
  const VIEW_SCHEMA_VERSION = 4;
  const VIEW_KEY = 'productsViewState';
  const VIEW_VER_KEY = 'productsViewStateVersion';
  function ensureViewVersion(){
    const ver = Number(localStorage.getItem(VIEW_VER_KEY) || '0');
    if(ver !== VIEW_SCHEMA_VERSION){
      localStorage.removeItem(VIEW_KEY);
      localStorage.removeItem('columnsVisibility');
      localStorage.setItem(VIEW_VER_KEY, String(VIEW_SCHEMA_VERSION));
    }
  }
  ensureViewVersion();

  const btn = document.getElementById('btnColumns');
  const panel = document.getElementById('columnsPanel');
  if(!btn || !panel) return;

  const OFFSET = window.POPOVER_OFFSET || 12;
  const MARGIN = window.POPOVER_MARGIN || 16;
  const overlay = window.ensureOverlayRoot ? window.ensureOverlayRoot() : document.body;

  // ensure panel lives in overlay portal to avoid clipping by ancestors
  if(panel.parentNode !== overlay){
    overlay.appendChild(panel);
  }

  function loadState(){
    try{ return JSON.parse(localStorage.columnsVisibility || '{}'); }catch(e){ return {}; }
  }
  function saveState(state){
    localStorage.columnsVisibility = JSON.stringify(state);
  }

  function ensureColumnVisible(key){
    const state = loadState();
    if(!(key in state)){
      state[key] = true;
      saveState(state);
    }
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
      panel.classList.remove('hidden');
      panel.style.visibility = 'hidden';
      panel.scrollTop = 0;
      // measure once rendered
      const w = panel.offsetWidth;
      const h = panel.offsetHeight;
      const vw = window.innerWidth;
      let left = rect.left;
      let top = rect.top - h - OFFSET;
      // clamp within viewport respecting safe margin
      if(left + w > vw - MARGIN){ left = vw - w - MARGIN; }
      if(left < MARGIN){ left = MARGIN; }
      if(top < MARGIN){ top = MARGIN; }
      panel.style.left = `${left}px`;
      panel.style.top = `${top}px`;
      panel.style.maxHeight = `${rect.top - OFFSET - MARGIN}px`;
      panel.style.visibility = '';
    } else {
      panel.classList.add('hidden');
    }
  });

  // close on Esc
  document.addEventListener('keydown', e => {
    if(e.key === 'Escape') panel.classList.add('hidden');
  });

  document.addEventListener('click', e => {
    if(!panel.contains(e.target) && e.target !== btn){
      panel.classList.add('hidden');
    }
  });

  // prevent panel scroll from bubbling to page/table
  panel.addEventListener('wheel', e => e.stopPropagation());

  window.refreshColumns = build;
  window.applyColumnVisibility = apply;
  window.ensureColumnVisible = ensureColumnVisible;

  // initial attempt (will populate once table is rendered)
  ensureColumnVisible('price');
  ensureColumnVisible('desire');
  build();
})();
