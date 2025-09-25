const SSE_URL = '/events';
let eventSource = null;
const handlerStore = new Map();

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

function ensureEventSource() {
  if (typeof window === 'undefined') return null;
  if (!('EventSource' in window)) return null;
  if (eventSource) return eventSource;
  eventSource = new EventSource(SSE_URL, { withCredentials: false });
  eventSource.addEventListener('error', () => {
    // keep connection alive; browsers will retry automatically
  });
  return eventSource;
}

export function getEventSource() {
  return ensureEventSource();
}

export function subscribe(eventName, handler) {
  const es = ensureEventSource();
  if (!es) {
    return () => {};
  }
  const wrapped = (ev) => {
    handler({
      event: ev.type,
      data: parseData(ev.data),
      lastEventId: ev.lastEventId || null,
      rawEvent: ev,
    });
  };
  es.addEventListener(eventName, wrapped);
  if (!handlerStore.has(eventName)) {
    handlerStore.set(eventName, []);
  }
  handlerStore.get(eventName).push({ original: handler, wrapped });
  return () => unsubscribe(eventName, handler);
}

export function unsubscribe(eventName, handler) {
  const es = ensureEventSource();
  if (!es) return;
  const entries = handlerStore.get(eventName);
  if (!entries || !entries.length) return;
  for (let i = 0; i < entries.length; i += 1) {
    const entry = entries[i];
    if (entry.original === handler) {
      es.removeEventListener(eventName, entry.wrapped);
      entries.splice(i, 1);
      break;
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
