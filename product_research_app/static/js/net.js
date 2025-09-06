export async function fetchJson(url, opts) {
  // Enhanced helper for JSON fetch requests.
  //
  // This version improves error handling by propagating server‑side error
  // messages back to the UI rather than always showing a generic network
  // failure.  It also falls back gracefully when the response is not
  // valid JSON.
  try {
    const options = Object.assign({}, opts || {});
    options.headers = Object.assign({}, options.headers);
    // Automatically set JSON content type when a body is provided (except
    // when using FormData).
    if (!(options.body instanceof FormData) && !options.headers['Content-Type']) {
      options.headers['Content-Type'] = 'application/json';
    }
    const response = await fetch(url, options);
    // Attempt to parse JSON payload; if parsing fails we still want to
    // inspect response.ok to determine success.
    let json;
    try {
      json = await response.json();
    } catch (_) {
      json = {};
    }
    // Treat any non‑2xx HTTP status or explicit ok=false as an error.  When
    // available, display the error message returned by the server.  If no
    // message is provided, fall back to the HTTP status text or a generic
    // error string.
    if (!response.ok || json.ok === false) {
      const msg = json.message || json.error || response.statusText || 'Error';
      const lp = json.log_path || '';
      toast.error(msg, {
        actionText: lp ? 'Copiar ruta' : '',
        onAction: () => navigator.clipboard.writeText(lp),
      });
      throw new Error(msg);
    }
    return json;
  } catch (err) {
    // Only report a network error when the request itself fails (e.g. the
    // server is unreachable or the connection is interrupted).
    toast.error('Error de red');
    throw err;
  }
}