(function(){
  const toast = (msg) => { try { window.showToast && showToast(msg, 'error'); } catch(_) { console.error(msg); } };
  window.addEventListener('error', e => toast('JS: ' + (e.error?.message || e.message)));
  window.addEventListener('unhandledrejection', e => toast('Promise: ' + (e.reason?.message || e.reason)));
  document.addEventListener('click', (e) => {
    const t = e.target.closest('button, a, [data-action]');
    if (t) console.debug('CLICK', t.tagName, t.id||'', t.dataset.action||'', (t.textContent||'').trim().slice(0,40));
  }, {capture:true});
})();
