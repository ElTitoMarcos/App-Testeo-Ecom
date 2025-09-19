(function(){
  const root = typeof window !== 'undefined' ? window : globalThis;
  if (!root) return;

  const Actions = {};
  const registry = (root.__actionsRegistry = root.__actionsRegistry || new Set());

  const register = (name, fn) => {
    if (!name || typeof name !== 'string' || typeof fn !== 'function') return;
    Actions[name] = fn;
    registry.add(name);
  };

  const pending = Array.isArray(root.__pendingActions) ? root.__pendingActions.slice() : [];
  pending.forEach((entry) => {
    if (!entry) return;
    const name = typeof entry.name === 'string' ? entry.name : entry[0];
    const fn = typeof entry.fn === 'function' ? entry.fn : entry[1];
    register(name, fn);
  });
  root.__pendingActions = [];

  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-action], button[data-action], a[data-action]');
    if (!el) return;
    const act = el.dataset.action;
    if (!act) return;
    const fn = Actions[act] || root[act];
    if (typeof fn === 'function') {
      e.preventDefault();
      console.debug('Acción ejecutada:', act, el);
      try {
        fn(el);
      } catch (err) {
        console.error('Action error', act, err);
      }
    } else {
      console.warn('Acción sin handler:', act, el);
    }
  }, { capture: false, passive: false });

  root.__registerAction = (name, fn) => {
    if (!name || typeof name !== 'string' || typeof fn !== 'function') return;
    register(name, fn);
  };
  root.__actionsHubReady = true;
})();
