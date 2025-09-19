// /static/js/legacy-progress-shim.js
(function(){
  const noop = (...a)=>{ try{ console.debug('[legacy-progress noop]', ...a);}catch(_){} };
  const toSSE = (payload)=>{ try{ window.SSEBus && console.debug('[legacy->SSE]', payload); }catch(_){} };

  window.showLoadingBar   = window.showLoadingBar   || function(msg){ toSSE({operation:'legacy', message: msg||'loading'}); };
  window.updateLoadingBar = window.updateLoadingBar || function(p){ toSSE({operation:'legacy', percent: Number(p)||0}); };
  window.hideLoadingBar   = window.hideLoadingBar   || function(){ toSSE({operation:'legacy', percent:100}); };

  // Si detectas otros nombres, añádelos:
  window.setProgress = window.setProgress || noop;
  window.startProgress = window.startProgress || noop;
  window.stopProgress  = window.stopProgress  || noop;

  // Logger mínimo por si aún hay errores
  window.addEventListener('error', e => console.error('[JS Error]', e.error||e.message));
  window.addEventListener('unhandledrejection', e => console.error('[Promise Error]', e.reason));
})();
