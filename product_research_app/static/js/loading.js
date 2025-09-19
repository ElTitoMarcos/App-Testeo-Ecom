const SSE_SUPPORTED = typeof window !== 'undefined' && typeof window.EventSource === 'function';

const OP_LABELS = {
  import: 'Import',
  enrich: 'Enriq.',
  delete: 'Elim.',
  weights: 'Pesos'
};

const OP_PRIORITY = {
  import: 1,
  enrich: 2,
  delete: 3,
  weights: 4
};

const COMPLETION_HOLD_MS = 2200;

const listeners = new Set();
const operations = new Map();

let eventSource = null;
let reconnectTimer = null;
let topBarSyncRaf = null;
let fallbackNotified = false;

function clampPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function toNumber(value) {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : 0;
  }
  if (typeof value === 'string') {
    const num = Number(value);
    return Number.isFinite(num) ? num : 0;
  }
  return 0;
}

function resolvePercent(event, previous) {
  if (event == null) return previous?.percent || 0;
  const direct = event.percent ?? event.pct;
  if (direct != null) return clampPercent(Number(direct));
  const total = toNumber(event.total);
  if (total > 0) {
    const processed = toNumber(event.processed ?? event.done ?? event.imported);
    return clampPercent((processed / total) * 100);
  }
  const imported = toNumber(event.imported);
  const queued = toNumber(event.queued);
  if (imported > 0 || queued > 0) {
    const totalGuess = imported + queued;
    if (totalGuess > 0) {
      return clampPercent((imported / totalGuess) * 100);
    }
  }
  if (event.status && ['done', 'completed', 'error'].includes(String(event.status))) {
    return 100;
  }
  return previous?.percent ?? 0;
}

function scheduleTopBarSync() {
  if (topBarSyncRaf) return;
  topBarSyncRaf = requestAnimationFrame(() => {
    topBarSyncRaf = null;
    const topBar = document.getElementById('topBar');
    if (!topBar) return;
    const rect = topBar.getBoundingClientRect();
    document.documentElement.style.setProperty('--topbar-height', `${Math.round(rect.height)}px`);
  });
}

function pruneOperations() {
  const now = Date.now();
  for (const [key, entry] of operations.entries()) {
    if (entry.completed && entry.completedAt && now - entry.completedAt > COMPLETION_HOLD_MS) {
      operations.delete(key);
    }
  }
}

function notify() {
  const snapshot = deriveState();
  for (const listener of listeners) {
    try {
      listener(snapshot);
    } catch (err) {
      console.error('progress listener failed', err);
    }
  }
}

function ensureSource() {
  if (!SSE_SUPPORTED) {
    if (!fallbackNotified) {
      fallbackNotified = true;
      console.warn('SSE no soportado; la barra global usará un estado estático.');
      document.documentElement?.setAttribute('data-sse-disabled', '1');
    }
    return;
  }
  if (eventSource) return;
  eventSource = new EventSource('/events');
  eventSource.onmessage = (ev) => {
    if (!ev.data) return;
    try {
      const payload = JSON.parse(ev.data);
      handleEvent(payload);
    } catch (err) {
      console.warn('Invalid progress payload', err);
    }
  };
  eventSource.onerror = () => {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    if (!reconnectTimer) {
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        ensureSource();
      }, 4000);
    }
  };
}

function handleEvent(event) {
  if (!event || !event.operation) return;
  const op = String(event.operation);
  const jobId = event.job_id ?? event.jobId ?? 'default';
  const key = `${op}:${jobId}`;
  const prev = operations.get(key);
  const now = Date.now();
  const percent = resolvePercent(event, prev);
  const status = event.status ?? prev?.status ?? null;
  const completed = percent >= 100 || (status && ['done', 'completed', 'error'].includes(String(status)));
  const entry = {
    key,
    operation: op,
    job_id: jobId,
    percent,
    status,
    message: event.message ?? prev?.message ?? '',
    enriched: toNumber(event.enriched ?? prev?.enriched ?? 0),
    failed: toNumber(event.failed ?? prev?.failed ?? 0),
    imported: toNumber(event.imported ?? prev?.imported ?? 0),
    queued: toNumber(event.queued ?? prev?.queued ?? 0),
    phase: event.phase ?? prev?.phase ?? op,
    eta_ms: event.eta_ms ?? prev?.eta_ms ?? null,
    updatedAt: now,
    completed,
    completedAt: completed ? (prev?.completed && prev?.completedAt ? prev.completedAt : now) : null
  };
  operations.set(key, entry);
  pruneOperations();
  notify();
}

function deriveState() {
  const now = Date.now();
  const items = Array.from(operations.values()).sort((a, b) => {
    if (a.completed !== b.completed) return a.completed ? 1 : -1;
    const pa = OP_PRIORITY[a.operation] ?? 50;
    const pb = OP_PRIORITY[b.operation] ?? 50;
    if (pa !== pb) return pa - pb;
    return (b.updatedAt ?? 0) - (a.updatedAt ?? 0);
  });
  const visible = items.filter((entry) => {
    if (!entry.completed) return true;
    if (!entry.completedAt) return true;
    return now - entry.completedAt <= COMPLETION_HOLD_MS;
  });
  const primary = visible[0] ?? null;
  const summary = visible
    .filter((entry) => entry.percent > 0 || entry.message)
    .slice(0, 3)
    .map((entry) => `${OP_LABELS[entry.operation] || entry.operation} ${entry.percent}%`)
    .join(' · ');
  return { entries: visible, primary, summary };
}

export function useSSEProgress() {
  ensureSource();
  return {
    subscribe(fn) {
      if (typeof fn !== 'function') return () => {};
      listeners.add(fn);
      try {
        fn(deriveState());
      } catch (err) {
        console.error('progress subscriber error', err);
      }
      return () => listeners.delete(fn);
    },
    getState: deriveState
  };
}

function formatAria(entry) {
  if (!entry) return 'Sin progreso activo';
  const label = OP_LABELS[entry.operation] || entry.operation;
  const phase = entry.phase ? ` (${entry.phase})` : '';
  const msg = entry.message ? `: ${entry.message}` : '';
  return `${label}${phase} ${entry.percent}%${msg}`;
}

function initGlobalProgressBar() {
  const host = document.getElementById('global-progress-bar');
  if (!host) return;
  const fill = host.querySelector('.global-progress__fill');
  const label = host.querySelector('.global-progress__label');
  const store = useSSEProgress();
  let hideTimer = null;

function render(state) {
    if (!SSE_SUPPORTED) {
      host.classList.add('is-visible', 'is-disabled');
      host.classList.remove('is-complete');
      host.setAttribute('aria-hidden', 'false');
      host.setAttribute('aria-valuenow', '0');
      host.setAttribute('aria-label', 'Seguimiento en vivo no disponible (SSE no soportado)');
      if (fill) fill.style.width = '0%';
      if (label) label.textContent = 'Seguimiento en vivo no disponible';
      scheduleTopBarSync();
      return;
    }
    const { primary, entries, summary } = state;
    const hasEntries = entries && entries.length > 0;
    if (!hasEntries) {
      host.classList.remove('is-visible', 'is-complete');
      host.setAttribute('aria-hidden', 'true');
      host.setAttribute('aria-valuenow', '0');
      if (fill) fill.style.width = '0%';
      if (label) label.textContent = '';
      scheduleTopBarSync();
      return;
    }
    const percent = clampPercent(primary?.percent ?? 0);
    if (fill) fill.style.width = `${percent}%`;
    host.setAttribute('aria-valuenow', String(percent));
    host.setAttribute('aria-hidden', 'false');
    host.setAttribute('aria-label', formatAria(primary));
    if (label) label.textContent = summary || formatAria(primary);
    host.classList.add('is-visible');
    const allCompleted = entries.every((entry) => entry.completed);
    host.classList.toggle('is-complete', allCompleted);
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
    if (allCompleted) {
      hideTimer = setTimeout(() => {
        host.classList.remove('is-visible', 'is-complete');
        host.setAttribute('aria-hidden', 'true');
        if (fill) fill.style.width = '0%';
        if (label) label.textContent = '';
        scheduleTopBarSync();
      }, COMPLETION_HOLD_MS + 200);
    }
    scheduleTopBarSync();
  }

  store.subscribe(render);
  render(store.getState());
}

export const LoadingHelpers = {
  start() {
    return {
      step() {},
      setStage() {},
      done() {}
    };
  }
};

function setup() {
  ensureSource();
  initGlobalProgressBar();
  scheduleTopBarSync();
  window.addEventListener('resize', scheduleTopBarSync, { passive: true });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setup);
} else {
  setup();
}

export default LoadingHelpers;
