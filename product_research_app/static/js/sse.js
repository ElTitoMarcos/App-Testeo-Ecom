const SSE_URL = '/events';
let eventSource = null;
let openPromise = null;
let sseAvailable;
const handlerStore = new Map();
const pendingRegistrations = [];

function parseData(raw) {
  if (raw == null) return null;
  const text = String(raw);
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch (err) {
    return text;
  }
}

function flushPending(es) {
  if (!es || !pendingRegistrations.length) return;
  const pending = pendingRegistrations.splice(0, pendingRegistrations.length);
  pending.forEach(({ eventName, entry }) => {
    const entries = handlerStore.get(eventName);
    if (entries && entries.includes(entry)) {
      es.addEventListener(eventName, entry.wrapped);
    }
  });
}

async function openEventSource() {
  if (eventSource) return eventSource;
  if (sseAvailable === false) return null;
  if (typeof window === 'undefined' || !('EventSource' in window)) {
    sseAvailable = false;
    return null;
  }
  if (openPromise) return openPromise;
  openPromise = (async () => {
    let ok = false;
    try {
      const resp = await fetch(SSE_URL, { method: 'HEAD' });
      ok = !!resp && resp.ok;
    } catch (err) {
      ok = false;
    }
    if (!ok) {
      sseAvailable = false;
      pendingRegistrations.length = 0;
      return null;
    }
    sseAvailable = true;
    const es = new EventSource(SSE_URL, { withCredentials: false });
    es.addEventListener('error', () => {
      // browsers will retry automatically when the connection drops
    });
    eventSource = es;
    flushPending(es);
    return es;
  })();
  openPromise.finally(() => {
    openPromise = null;
  });
  return openPromise;
}

function ensureEventSource() {
  if (eventSource) return eventSource;
  void openEventSource();
  return eventSource;
}

export function getEventSource() {
  return ensureEventSource();
}

function registerOrQueue(eventName, entry) {
  if (eventSource) {
    eventSource.addEventListener(eventName, entry.wrapped);
    return;
  }
  if (sseAvailable === false) {
    return;
  }
  pendingRegistrations.push({ eventName, entry });
  void openEventSource();
}

export function subscribe(eventName, handler) {
  const wrapped = (ev) => {
    handler({
      event: ev.type,
      data: parseData(ev.data),
      lastEventId: ev.lastEventId || null,
      rawEvent: ev,
    });
  };
  if (!handlerStore.has(eventName)) {
    handlerStore.set(eventName, []);
  }
  const entry = { original: handler, wrapped };
  handlerStore.get(eventName).push(entry);
  registerOrQueue(eventName, entry);
  return () => unsubscribe(eventName, handler);
}

function removePending(entry) {
  for (let i = pendingRegistrations.length - 1; i >= 0; i -= 1) {
    if (pendingRegistrations[i].entry === entry) {
      pendingRegistrations.splice(i, 1);
    }
  }
}

export function unsubscribe(eventName, handler) {
  const entries = handlerStore.get(eventName);
  if (!entries || !entries.length) return;
  let entryToRemove = null;
  for (let i = 0; i < entries.length; i += 1) {
    const entry = entries[i];
    if (entry.original === handler) {
      entryToRemove = entry;
      entries.splice(i, 1);
      break;
    }
  }
  if (entryToRemove) {
    if (eventSource) {
      eventSource.removeEventListener(eventName, entryToRemove.wrapped);
    } else {
      removePending(entryToRemove);
    }
  }
  if (!entries.length) {
    handlerStore.delete(eventName);
  }
}

const busRegistry = new Map();

window.SSEBus = {
  on(eventName, handler) {
    const wrapped = (detail) => handler(detail.data, detail);
    const unsubscribeFn = subscribe(eventName, wrapped);
    if (!busRegistry.has(eventName)) {
      busRegistry.set(eventName, new Map());
    }
    busRegistry.get(eventName).set(handler, unsubscribeFn);
    return () => this.off(eventName, handler);
  },
  off(eventName, handler) {
    const registry = busRegistry.get(eventName);
    if (!registry) return;
    const unsubscribeFn = registry.get(handler);
    if (typeof unsubscribeFn === 'function') {
      unsubscribeFn();
    }
    registry.delete(handler);
    if (registry.size === 0) {
      busRegistry.delete(eventName);
    }
  },
};

// Ensure connection is opened as soon as possible
ensureEventSource();
