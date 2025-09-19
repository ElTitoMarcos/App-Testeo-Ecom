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
  const base = 2500;
  const max = 20000;
  let delay = base;
  let emptyHits = 0;
  let running = false;
  let stopRequested = false;
  let timerId = null;

  const advanceCursor = (current, candidate) => {
    if (typeof candidate !== 'string' || !candidate) return current;
    if (!current || candidate > current) return candidate;
    return current;
  };

  const resolveCursor = (payload, fallbackIso) => {
    if (payload && typeof payload.next_since === 'string' && payload.next_since) {
      return payload.next_since;
    }
    if (payload && typeof payload.server_now === 'string' && payload.server_now) {
      return payload.server_now;
    }
    return fallbackIso;
  };

  function nextDelay(empty) {
    if (empty) {
      emptyHits += 1;
      const exponent = Math.min(emptyHits, 3);
      delay = Math.min(max, base * Math.pow(2, exponent));
    } else {
      emptyHits = 0;
      delay = base;
    }
  }

  async function pollOnce() {
    if (stopRequested) return false;
    let manualSchedule = false;
    try {
      const fetchOpts = { cache: 'no-store', __skipLoadingHook: true };
      const [prod, enrich] = await Promise.all([
        fetch(`/products/delta?since=${encodeURIComponent(sinceProducts)}&limit=1000`, fetchOpts)
          .then((r) => (r && r.ok ? r.json() : null))
          .catch(() => null),
        fetch(`/enrich/delta?since=${encodeURIComponent(sinceEnrich)}&limit=2000`, fetchOpts)
          .then((r) => (r && r.ok ? r.json() : null))
          .catch(() => null),
      ]);

      let got = 0;
      if (Array.isArray(prod?.rows) && prod.rows.length) {
        window.LiveTable?.onImportBatch(prod.rows);
        got += prod.rows.length;
      }
      if (Array.isArray(enrich?.rows) && enrich.rows.length) {
        window.LiveTable?.onEnrichBatch(enrich.rows);
        got += enrich.rows.length;
      }

      const fallbackNow = new Date().toISOString();
      sinceProducts = advanceCursor(sinceProducts, resolveCursor(prod, fallbackNow));
      sinceEnrich = advanceCursor(sinceEnrich, resolveCursor(enrich, fallbackNow));

      const empty = got === 0;
      nextDelay(empty);

      const active = Boolean((prod && prod.active_jobs) || (enrich && enrich.active_jobs));
      if (!active && empty && emptyHits >= 3) {
        manualSchedule = true;
        schedule();
      }
    } catch (err) {
      console.warn('[poll error]', err);
      nextDelay(true);
    }
    return !manualSchedule;
  }

  function schedule() {
    if (stopRequested) return;
    if (timerId) {
      clearTimeout(timerId);
    }
    const wait = delay;
    timerId = setTimeout(async function loop(){
      timerId = null;
      if (stopRequested) return;
      if (running) {
        schedule();
        return;
      }
      running = true;
      const shouldAuto = await pollOnce();
      running = false;
      if (stopRequested) return;
      if (shouldAuto !== false && timerId === null) {
        schedule();
      }
    }, wait);
  }

  window.LiveStream = {
    start(){
      stopRequested = false;
      emptyHits = 0;
      delay = base;
      schedule();
    },
    stop(){
      stopRequested = true;
      if (timerId) {
        clearTimeout(timerId);
        timerId = null;
      }
      running = false;
    },
    bump(){
      emptyHits = 0;
      delay = base;
      if (!stopRequested) {
        schedule();
      }
    }
  };

  const handleEvent = (data) => {
    if (!data || typeof data !== 'object') return;
    try {
      if (data.type === 'import.batch') {
        const rows = Array.isArray(data.rows) ? data.rows : [];
        if (rows.length) {
          window.LiveTable?.onImportBatch(rows);
        }
        return;
      }
      if (data.type === 'enrich.batch') {
        const rows = Array.isArray(data.updates) ? data.updates : [];
        if (rows.length) {
          window.LiveTable?.onEnrichBatch(rows);
        }
        return;
      }
      if (data.type === 'job.started') {
        window.LiveStream?.bump?.();
        window.LiveStream?.start?.();
        return;
      }
      if (data.type === 'job.finished') {
        window.LiveStream?.bump?.();
        return;
      }
      if (data.type === 'import.done' || data.type === 'enrich.done') {
        console.log('[LIVE]', data.type, data);
      }
    } catch (err) {
      console.error('[LIVE handler]', err);
    }
  };

  const unsubscribe = bus.on(handleEvent);

  let fallbackStarted = false;
  let source = null;

  const startPolling = () => {
    if (fallbackStarted) return;
    fallbackStarted = true;
    if (source && typeof source.close === 'function') {
      try { source.close(); } catch (err) { /* noop */ }
    }
    window.LiveStream.start();
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
    window.LiveStream.stop();
  });
})();
