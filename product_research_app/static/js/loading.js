(function(){
  const bar = document.querySelector('#loadingBar');
  const label = document.querySelector('#loadingText');
  const wrap = document.querySelector('#loadingWrap');
  if(!bar || !label || !wrap) return;

  let last = 0;

  function setProgress(p){
    const clamped = Math.max(0, Math.min(1, p || 0));
    last = clamped;
    const pct = Math.round(clamped * 100);
    bar.style.width = pct + '%';
    label.textContent = pct + '%';
  }

  function completeAndHide(){
    setProgress(1);
    wrap.classList.add('is-complete');
    setTimeout(() => {
      wrap.style.display = 'none';
      window.dispatchEvent(new CustomEvent('products:reload'));
    }, 400);
  }

  function boot(){
    const es = new EventSource('/events/ai');
    es.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data);
        if (payload.event === 'progress') {
          const d = payload.data || {};
          if (typeof d.percent === 'number') {
            setProgress(d.percent);
          } else if (d.total) {
            setProgress(d.processed / d.total);
          }
        } else if (payload.event === 'done') {
          completeAndHide();
          es.close();
        }
      } catch(_) {}
    };
    es.onerror = () => {
      // si se corta SSE y ya vamos altos, no dejar colgado
      if (last >= 0.85) completeAndHide();
    };
  }

  setTimeout(() => { if (last === 0) setProgress(0.01); }, 300);
  boot();
})();
