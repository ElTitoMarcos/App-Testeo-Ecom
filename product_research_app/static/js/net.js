const API_BASE_URL = window.API_BASE_URL || '';

export async function fetchJson(url, opts, timeoutMs = 25000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const options = Object.assign({ signal: controller.signal }, opts);
    options.headers = Object.assign({}, options.headers);
    if (!(options.body instanceof FormData) && !options.headers['Content-Type']) {
      options.headers['Content-Type'] = 'application/json';
    }
    const r = await fetch(API_BASE_URL + url, options);
    const j = await r.json().catch(() => ({}));
    if (!r.ok || j.ok === false) {
      const lp = j.log_path || '';
      const msg = j.message || j.error || r.statusText || 'Error';
      toast.error(`Error ${r.status}: ${msg}`, {
        actionText: lp ? 'Copiar ruta' : '',
        onAction: () => navigator.clipboard.writeText(lp)
      });
      throw new Error(msg);
    }
    return j;
  } catch (err) {
    if (err.name === 'AbortError') {
      toast.error('Error: timeout');
    } else {
      toast.error('Error de red');
    }
    throw err;
  } finally {
    clearTimeout(id);
  }
}

export async function updateProductField(id, data, timeoutMs = 25000) {
  return fetchJson(`/api/products/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  }, timeoutMs);
}

export async function patch(url, data, timeoutMs = 25000) {
  return fetchJson(url, {
    method: 'PATCH',
    body: JSON.stringify(data),
  }, timeoutMs);
}

export async function post(url, data, timeoutMs = 25000) {
  return fetchJson(url, {
    method: 'POST',
    body: JSON.stringify(data)
  }, timeoutMs);
}

(function(){
  if (typeof window === 'undefined') return;

  let es = null;
  let reconnectTimer = null;
  let pollingTimer = null;
  let pollingActive = false;
  const listeners = new Set();

  function dispatch(payload) {
    if (payload == null) return;
    listeners.forEach((fn) => {
      try {
        fn(payload);
      } catch (err) {
        console.error('SSE listener error', err);
      }
    });
  }

  function stopPolling() {
    pollingActive = false;
    if (pollingTimer) {
      clearTimeout(pollingTimer);
      pollingTimer = null;
    }
  }

  function startPolling() {
    if (pollingActive) return;
    pollingActive = true;
    console.log('[SSE] fallback polling...');

    const tick = async () => {
      if (!pollingActive) return;
      try {
        const response = await fetch('/events/poll', { cache: 'no-store' });
        if (response.ok) {
          const data = await response.json().catch(() => null);
          const events = Array.isArray(data?.events) ? data.events : [];
          events.forEach((eventPayload) => dispatch(eventPayload));
        }
      } catch (err) {
        // ignore polling errors
      }
      if (pollingActive) {
        pollingTimer = setTimeout(tick, 3000);
      }
    };

    tick();
  }

  function startSSE() {
    if (es) return es;
    if (typeof window.EventSource !== 'function') {
      console.warn('[SSE] EventSource no soportado en este navegador');
      startPolling();
      return null;
    }

    console.log('[SSE] opening...');
    es = new EventSource('/events');
    es.onopen = () => {
      console.log('[SSE] open');
      stopPolling();
    };
    es.onmessage = (ev) => {
      if (!ev || !ev.data) return;
      try {
        const parsed = JSON.parse(ev.data);
        dispatch(parsed);
      } catch (err) {
        // keepalives or invalid JSON
      }
    };
    es.onerror = (event) => {
      console.warn('[SSE] error', event);
      if (es) {
        try {
          console.log('[SSE] closing');
          es.close();
        } catch (_) {
          // ignore
        }
      }
      es = null;
      startPolling();
      if (!reconnectTimer) {
        reconnectTimer = setTimeout(() => {
          reconnectTimer = null;
          startSSE();
        }, 5000);
      }
    };

    return es;
  }

  window.SSEBus = {
    connect() {
      const source = startSSE();
      if (!source) {
        startPolling();
      }
      return source;
    },
    on(fn) {
      if (typeof fn !== 'function') return () => {};
      listeners.add(fn);
      return () => listeners.delete(fn);
    }
  };
})();
