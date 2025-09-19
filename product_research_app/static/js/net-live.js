(function(){
  const listeners = new Set();
  const bus = window.SSEBus || {};
  bus.on = function(handler){
    if (typeof handler !== 'function') return () => {};
    listeners.add(handler);
    return () => listeners.delete(handler);
  };
  bus.emit = function(payload){
    listeners.forEach((fn) => {
      try {
        fn(payload);
      } catch (err) {
        console.error('[SSE handler]', err);
      }
    });
  };
  window.SSEBus = bus;

  let sinceProducts = new Date().toISOString();
  let sinceEnrich = new Date().toISOString();
  let fallbackStarted = false;
  let source = null;

  const handleEvent = (data) => {
    if (!data || typeof data !== 'object') return;
    try {
      if (data.type === 'import.batch') {
        window.LiveTable?.onImportBatch(data.rows || []);
        return;
      }
      if (data.type === 'enrich.batch') {
        window.LiveTable?.onEnrichBatch(data.updates || []);
        return;
      }
      if (data.type === 'import.done' || data.type === 'enrich.done') {
        console.log('[LIVE]', data.type, data);
      }
    } catch (err) {
      console.error('[LIVE handler]', err);
    }
  };

  const poll = async () => {
    try {
      const prodUrl = `/products/delta?since=${encodeURIComponent(sinceProducts)}&limit=1000`;
      const enrichUrl = `/enrich/delta?since=${encodeURIComponent(sinceEnrich)}&limit=2000`;
      const [p, e] = await Promise.all([
        fetch(prodUrl, { cache: 'no-store' }).then((r) => r.ok ? r.json() : {}),
        fetch(enrichUrl, { cache: 'no-store' }).then((r) => r.ok ? r.json() : {}),
      ]);
      if (p && typeof p.next_since === 'string') {
        sinceProducts = p.next_since;
      } else if (!p || typeof p !== 'object') {
        sinceProducts = new Date().toISOString();
      }
      if (e && typeof e.next_since === 'string') {
        sinceEnrich = e.next_since;
      } else if (!e || typeof e !== 'object') {
        sinceEnrich = new Date().toISOString();
      }
      if (p && Array.isArray(p.rows) && p.rows.length) {
        window.LiveTable?.onImportBatch(p.rows);
      }
      if (e && Array.isArray(e.rows) && e.rows.length) {
        window.LiveTable?.onEnrichBatch(e.rows);
      }
    } catch (err) {
      console.warn('[poll error]', err);
    } finally {
      setTimeout(poll, 2500);
    }
  };

  const startPolling = () => {
    if (fallbackStarted) return;
    fallbackStarted = true;
    if (source && typeof source.close === 'function') {
      try { source.close(); } catch (err) { /* noop */ }
    }
    poll();
  };

  const startSSE = () => {
    if (!('EventSource' in window)) {
      return false;
    }
    try {
      source = new EventSource('/events');
    } catch (err) {
      console.warn('[SSE] init failed', err);
      return false;
    }
    let opened = false;
    const watchdog = setTimeout(() => {
      if (!opened) {
        console.warn('[SSE] timeout waiting for open, falling back to polling');
        startPolling();
      }
    }, 3500);

    source.onopen = () => {
      opened = true;
      clearTimeout(watchdog);
    };

    source.onmessage = (event) => {
      opened = true;
      clearTimeout(watchdog);
      if (!event.data) return;
      try {
        const payload = JSON.parse(event.data);
        bus.emit(payload);
      } catch (err) {
        console.warn('[SSE parse]', err);
      }
    };

    source.onerror = (err) => {
      console.warn('[SSE] error', err);
      if (source.readyState === EventSource.CLOSED) {
        startPolling();
      }
    };
    return true;
  };

  const unsubscribe = bus.on(handleEvent);
  if (!startSSE()) {
    startPolling();
  }

  window.addEventListener('beforeunload', () => {
    if (unsubscribe) {
      try { unsubscribe(); } catch (err) { /* noop */ }
    }
    if (source && typeof source.close === 'function') {
      try { source.close(); } catch (err) { /* noop */ }
    }
  });
})();
