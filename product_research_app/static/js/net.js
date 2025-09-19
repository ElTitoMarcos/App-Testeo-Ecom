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

if (typeof window !== 'undefined') {
  window.apiPing = async () => fetch('/healthz')
    .then((r) => r.json())
    .then((j) => console.log('healthz', j))
    .catch(console.error);
}
