// --- AbortHub global para cancelar en cascada ---
export const AbortHub = (() => {
  const controllers = new Set();

  function track(ctrl) {
    controllers.add(ctrl);
    return () => controllers.delete(ctrl);
  }

  function make() {
    const c = new AbortController();
    track(c);
    return c;
  }

  function cancelAll(reason = 'user_cancelled') {
    controllers.forEach(c => {
      try {
        c.abort(reason);
      } catch (_) {
        /* noop */
      }
    });
    controllers.clear();
  }

  async function trackedFetch(input, init = {}) {
    const c = make();
    const opts = { ...init, signal: init.signal ?? c.signal };
    try {
      return await fetch(input, opts);
    } finally {
      controllers.delete(c);
    }
  }

  function trackXhr(xhr) {
    const c = make();
    const remove = () => controllers.delete(c);
    const onAbort = () => {
      try {
        xhr.abort();
      } catch (_) {
        /* noop */
      }
    };
    xhr.abortController = c;
    c.signal.addEventListener('abort', onAbort);
    xhr.addEventListener('loadend', remove, { once: true });
    xhr.addEventListener('error', remove, { once: true });
    xhr.addEventListener('abort', remove, { once: true });
    return () => {
      c.signal.removeEventListener('abort', onAbort);
      remove();
    };
  }

  return { make, track, cancelAll, trackedFetch, trackXhr };
})();

const API_BASE_URL = window.API_BASE_URL || '';

export async function fetchJson(url, init = {}, timeoutMs = 25000) {
  const controller = AbortHub.make();
  const untrack = AbortHub.track(controller);
  const timeoutId = timeoutMs ? setTimeout(() => {
    try {
      controller.abort('timeout');
    } catch (_) {
      /* noop */
    }
  }, timeoutMs) : null;

  const headers = { ...(init.headers || {}) };
  if (!(init.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  const opts = {
    ...init,
    headers,
    signal: init.signal ?? controller.signal,
  };

  try {
    const response = await AbortHub.trackedFetch(API_BASE_URL + url, opts);
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) {
      const lp = data.log_path || '';
      const msg = data.message || data.error || response.statusText || 'Error';
      toast.error(`Error ${response.status}: ${msg}`, {
        actionText: lp ? 'Copiar ruta' : '',
        onAction: () => navigator.clipboard.writeText(lp)
      });
      throw new Error(msg);
    }
    return data;
  } catch (err) {
    const signal = opts.signal;
    const reason = signal?.reason;
    if (err.name === 'AbortError') {
      if (reason === 'user_cancelled') {
        throw err;
      }
      if (reason === 'timeout') {
        toast.error('Error: timeout');
      } else {
        toast.error('Operación cancelada');
      }
    } else {
      toast.error('Error de red');
    }
    throw err;
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
    untrack();
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
