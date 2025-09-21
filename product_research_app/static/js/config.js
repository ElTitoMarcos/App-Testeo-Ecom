import { LoadingHelpers } from './loading.js';

const DEFAULT_WEIGHT = 14;
const DEFAULT_ORDER = [
  'awareness',
  'desire',
  'revenue',
  'competition',
  'units_sold',
  'price',
  'oldness',
  'rating',
];
const METRIC_DEFS = [
  { key: 'awareness', label: 'Awareness', extremes: { left: 'Unaware', right: 'Most aware' } },
  { key: 'desire', label: 'Desire', extremes: { left: 'Menor deseo', right: 'Mayor deseo' } },
  { key: 'revenue', label: 'Revenue', extremes: { left: 'Menores ingresos', right: 'Mayores ingresos' } },
  { key: 'competition', label: 'Competition', extremes: { left: 'Menos competencia', right: 'Más competencia' } },
  { key: 'units_sold', label: 'Units Sold', extremes: { left: 'Menos ventas', right: 'Más ventas' } },
  { key: 'price', label: 'Price', extremes: { left: 'Más barato', right: 'Más caro' } },
  { key: 'oldness', label: 'Oldness (antigüedad)', extremes: { left: 'Más reciente', right: 'Más antiguo' } },
  { key: 'rating', label: 'Rating', extremes: { left: 'Peor rating', right: 'Mejor rating' } },
];
const METRIC_INDEX = new Map(METRIC_DEFS.map(def => [def.key, def]));
const METRIC_KEYS = METRIC_DEFS.map(def => def.key);
const ALIASES = {
  unitsSold: 'units_sold',
  unitssold: 'units_sold',
  orders: 'units_sold',
};

function normalizeKey(k) {
  if (k === undefined || k === null) return '';
  const str = String(k).trim();
  if (!str) return '';
  if (ALIASES[str]) return ALIASES[str];
  const lower = str.toLowerCase();
  if (ALIASES[lower]) return ALIASES[lower];
  const canonical = lower.replace(/[\s-]+/g, '_');
  if (ALIASES[canonical]) return ALIASES[canonical];
  if (METRIC_INDEX.has(str)) return str;
  if (METRIC_INDEX.has(lower)) return lower;
  if (METRIC_INDEX.has(canonical)) return canonical;
  return canonical;
}

function clampInt(value, min, max, fallback = min) {
  let num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  num = Math.round(num);
  if (Number.isNaN(num)) return fallback;
  if (num < min) return min;
  if (num > max) return max;
  return num;
}

function cloneState(state) {
  if (!state) return getDefaultState();
  return {
    order: Array.isArray(state.order) ? state.order.slice() : [],
    weights: { ...(state.weights || {}) },
    enabled: { ...(state.enabled || {}) },
  };
}

function getDefaultState() {
  const weights = {};
  const enabled = {};
  DEFAULT_ORDER.forEach(key => {
    weights[key] = DEFAULT_WEIGHT;
    enabled[key] = true;
  });
  return { order: DEFAULT_ORDER.slice(), weights, enabled };
}

function normalizeState(raw = {}) {
  const base = getDefaultState();
  const normalizedWeights = {};
  const srcWeights = raw.weights || raw.winner_weights || raw.factors || {};
  Object.entries(srcWeights).forEach(([rawKey, value]) => {
    const key = normalizeKey(rawKey);
    if (METRIC_INDEX.has(key)) {
      normalizedWeights[key] = clampInt(value, 0, 50, DEFAULT_WEIGHT);
    }
  });
  const normalizedEnabled = {};
  const srcEnabled = raw.weights_enabled || raw.enabled || raw.winner_weights_enabled || {};
  Object.entries(srcEnabled).forEach(([rawKey, value]) => {
    const key = normalizeKey(rawKey);
    if (METRIC_INDEX.has(key)) {
      normalizedEnabled[key] = value !== false;
    }
  });

  let orderCandidate = raw.order || raw.weights_order || raw.winner_order || [];
  const seen = new Set();
  const order = [];
  if (Array.isArray(orderCandidate) && orderCandidate.length) {
    orderCandidate.forEach(rawKey => {
      const key = normalizeKey(rawKey);
      if (METRIC_INDEX.has(key) && !seen.has(key)) {
        seen.add(key);
        order.push(key);
      }
    });
  }
  DEFAULT_ORDER.forEach(key => {
    if (!seen.has(key)) {
      seen.add(key);
      order.push(key);
    }
  });

  const weights = {};
  const enabled = {};
  order.forEach(key => {
    weights[key] = Object.prototype.hasOwnProperty.call(normalizedWeights, key)
      ? normalizedWeights[key]
      : clampInt(base.weights[key], 0, 50, DEFAULT_WEIGHT);
    enabled[key] = Object.prototype.hasOwnProperty.call(normalizedEnabled, key)
      ? normalizedEnabled[key]
      : base.enabled[key];
  });

  return { order, weights, enabled };
}

const SettingsCache = (() => {
  let cache = null;
  let inflight = null;

  const norm = (raw = {}) => normalizeState(raw);

  const get = async () => {
    if (cache) return cloneState(cache);
    if (inflight) return inflight;
    inflight = fetch('/api/config/winner-weights', { cache: 'no-store' })
      .then(res => res.json())
      .then(data => {
        cache = norm(data);
        return cloneState(cache);
      })
      .catch(err => {
        cache = norm({});
        throw err;
      })
      .finally(() => {
        inflight = null;
      });
    return inflight;
  };

  const set = (next) => {
    cache = norm(next || {});
  };

  const snapshot = () => cloneState(cache || norm({}));

  return { get, set, snapshot };
})();

let cacheState = getDefaultState();
let saveTimer = null;
let saveInFlight = null;
let pendingPreviousState = null;

document.addEventListener('DOMContentLoaded', () => {
  SettingsCache.get().catch(() => {});
});

function beginStateChange() {
  if (!pendingPreviousState) {
    pendingPreviousState = cloneState(cacheState);
  }
  return cloneState(pendingPreviousState);
}

function getPendingPreviousState() {
  return pendingPreviousState ? cloneState(pendingPreviousState) : null;
}

function clearPendingPreviousState() {
  pendingPreviousState = null;
}

function computeDiff(prev, next) {
  if (!prev || !next) {
    return { weightChanges: [], toggleChanges: [], orderMoves: [] };
  }
  const weightChanges = [];
  const toggleChanges = [];
  const orderMoves = [];
  const prevPositions = new Map();
  (prev.order || []).forEach((key, idx) => prevPositions.set(key, idx));

  next.order.forEach((key, idx) => {
    if (prev.weights) {
      const prevWeight = prev.weights[key];
      const nextWeight = next.weights[key];
      if (prevWeight !== undefined && nextWeight !== undefined && prevWeight !== nextWeight) {
        weightChanges.push({ key, from: prevWeight, to: nextWeight });
      }
    }
    const prevEnabled = prev.enabled ? prev.enabled[key] !== false : true;
    const nextEnabled = next.enabled ? next.enabled[key] !== false : true;
    if (prevEnabled !== nextEnabled) {
      toggleChanges.push({ key, from: prevEnabled, to: nextEnabled });
    }
    if (prevPositions.has(key)) {
      const fromIdx = prevPositions.get(key);
      if (fromIdx !== idx) {
        orderMoves.push({ key, from: fromIdx, to: idx });
      }
    }
  });

  return { weightChanges, toggleChanges, orderMoves };
}

function formatMetricLabel(key) {
  const def = METRIC_INDEX.get(key);
  return def ? def.label : key;
}

function formatOrder(order) {
  return (order || []).map(formatMetricLabel).join(' → ');
}

function applyHighlights(diff, options = {}) {
  if (!diff) return;
  const list = document.getElementById('weightsList');
  if (!list) return;

  diff.weightChanges.forEach(({ key }) => {
    const item = list.querySelector(`.weight-item[data-key="${key}"]`);
    if (!item) return;
    item.classList.add('highlight');
    setTimeout(() => item.classList.remove('highlight'), 900);
    const input = item.querySelector('.weight-input');
    if (input) {
      input.classList.add('changed');
      setTimeout(() => input.classList.remove('changed'), 900);
    }
  });

  diff.toggleChanges.forEach(({ key }) => {
    const item = list.querySelector(`.weight-item[data-key="${key}"]`);
    if (!item) return;
    item.classList.add('highlight');
    setTimeout(() => item.classList.remove('highlight'), 900);
  });

  diff.orderMoves.forEach(({ key, from, to }) => {
    const item = list.querySelector(`.weight-item[data-key="${key}"]`);
    if (!item) return;
    const cls = to < from ? 'moved-up' : 'moved-down';
    item.classList.add(cls);
    setTimeout(() => item.classList.remove(cls), 1200);
  });

  if (options.focusKey) {
    const focusEl = list.querySelector(`.weight-item[data-key="${options.focusKey}"]`);
    if (focusEl) focusEl.focus();
  }
}

function updateDiffPanel(diff) {
  const panel = document.getElementById('weightsDiffPanel');
  if (!panel) return;
  const body = panel.querySelector('.diff-body');
  if (!body) return;

  if (!diff || (!diff.weightChanges.length && !diff.orderMoves.length && !diff.toggleChanges.length)) {
    body.innerHTML = '';
    panel.hidden = true;
    panel.removeAttribute('open');
    return;
  }

  body.innerHTML = '';
  if (diff.weightChanges.length) {
    diff.weightChanges.forEach(({ key, from, to }) => {
      const row = document.createElement('div');
      row.className = 'diff-row';
      row.innerHTML = `<span>${formatMetricLabel(key)}</span><span>${from} → ${to}</span>`;
      body.appendChild(row);
    });
  }
  if (diff.orderMoves.length) {
    diff.orderMoves.forEach(({ key, from, to }) => {
      const row = document.createElement('div');
      row.className = 'diff-row';
      const direction = to < from ? '↑' : '↓';
      row.innerHTML = `<span class="diff-move">${formatMetricLabel(key)}</span><span>${direction} #${from + 1} → #${to + 1}</span>`;
      body.appendChild(row);
    });
  }
  if (diff.toggleChanges.length) {
    diff.toggleChanges.forEach(({ key, to }) => {
      const row = document.createElement('div');
      row.className = 'diff-row';
      row.innerHTML = `<span>${formatMetricLabel(key)}</span><span>${to ? 'Activado' : 'Desactivado'}</span>`;
      body.appendChild(row);
    });
  }
  panel.hidden = false;
  panel.open = true;
}

function syncListWithState(state) {
  const list = document.getElementById('weightsList');
  if (!list) return;
  const items = new Map(Array.from(list.children).map(el => [el.dataset.key, el]));
  const total = state.order.length;

  state.order.forEach((key, idx) => {
    const el = items.get(key);
    if (!el) return;
    list.appendChild(el);
    el.setAttribute('aria-posinset', idx + 1);
    el.setAttribute('aria-setsize', total);
    const rankEl = el.querySelector('.wi-rank');
    if (rankEl) rankEl.textContent = `#${idx + 1}`;
    const input = el.querySelector('.weight-input');
    if (input) {
      const val = clampInt(state.weights[key], 0, 50, DEFAULT_WEIGHT);
      if (Number(input.value) !== val) input.value = val;
      input.disabled = state.enabled[key] === false;
    }
    const toggle = el.querySelector('.wt-enabled');
    if (toggle) toggle.checked = state.enabled[key] !== false;
    el.classList.toggle('disabled', state.enabled[key] === false);
  });
}

function buildPayload(state) {
  const payloadWeights = {};
  const enabledMap = {};
  state.order.forEach(key => {
    payloadWeights[key] = clampInt(state.weights[key], 0, 50, DEFAULT_WEIGHT);
    enabledMap[key] = state.enabled[key] !== false;
  });
  return {
    weights: payloadWeights,
    order: state.order.slice(),
    weights_enabled: enabledMap,
  };
}

async function saveSettings(options = {}) {
  const stateToPersist = options.state ? cloneState(options.state) : cloneState(cacheState);
  const prevState = options.previousState ? cloneState(options.previousState) : getPendingPreviousState() || cloneState(cacheState);
  const reason = options.reason || 'weights';
  const toastMessage = options.toastMessage;
  const payload = buildPayload(stateToPersist);
  const host = options.host || document.querySelector('#progress-slot-global');
  const needsTracker = reason === 'ai' || reason === 'order' || reason === 'reset';
  const tracker = needsTracker ? LoadingHelpers.start('Guardando configuración', { host }) : null;

  const run = async () => {
    try {
      const res = await fetch('/api/config/winner-weights', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        __hostEl: host,
      });
      if (!res.ok) throw new Error('save_failed');
      const data = await res.json().catch(() => ({}));
      SettingsCache.set(data);
      cacheState = SettingsCache.snapshot();
      syncListWithState(cacheState);
      if (toastMessage && typeof toast !== 'undefined' && toast.success) {
        toast.success(toastMessage);
      }
      if (typeof reloadProductsLight === 'function') reloadProductsLight();
      else if (typeof reloadProducts === 'function') reloadProducts();
    } catch (err) {
      if (typeof toast !== 'undefined' && toast.error) {
        toast.error('No se pudo guardar la configuración');
      }
      cacheState = prevState;
      renderWeightsUI(cacheState, { updateDiffPanel: true, applyHighlights: false });
      throw err;
    } finally {
      if (tracker) tracker.done();
      clearPendingPreviousState();
    }
  };

  if (saveInFlight) {
    try {
      await saveInFlight;
    } catch (err) {
      // precedente fallido; continuar
    }
  }
  saveInFlight = run();
  try {
    await saveInFlight;
  } finally {
    saveInFlight = null;
  }
}

function markDirty(options = {}) {
  const immediate = options.immediate === true;
  const delay = typeof options.delay === 'number' ? options.delay : 700;
  clearTimeout(saveTimer);
  if (immediate) {
    return saveSettings(options);
  }
  saveTimer = setTimeout(() => {
    saveSettings(options);
  }, delay);
  return Promise.resolve();
}

function attachItemEvents(li, key) {
  const input = li.querySelector('.weight-input');
  if (input) {
    input.addEventListener('input', (ev) => {
      beginStateChange();
      let val = clampInt(ev.target.value, 0, 50, cacheState.weights[key] ?? DEFAULT_WEIGHT);
      if (!Number.isFinite(val)) val = cacheState.weights[key] ?? DEFAULT_WEIGHT;
      cacheState.weights[key] = val;
      ev.target.value = val;
      const base = getPendingPreviousState();
      if (base) {
        const diff = computeDiff(base, cacheState);
        updateDiffPanel(diff);
      }
      markDirty({ reason: 'weights' });
    });
    input.addEventListener('change', (ev) => {
      let val = clampInt(ev.target.value, 0, 50, cacheState.weights[key] ?? DEFAULT_WEIGHT);
      cacheState.weights[key] = val;
      ev.target.value = val;
      const base = getPendingPreviousState();
      if (base) {
        const diff = computeDiff(base, cacheState);
        updateDiffPanel(diff);
      }
      markDirty({ reason: 'weights' });
    });
  }

  const toggle = li.querySelector('.wt-enabled');
  if (toggle) {
    toggle.addEventListener('change', (ev) => {
      beginStateChange();
      const on = !!ev.target.checked;
      cacheState.enabled[key] = on;
      li.classList.toggle('disabled', !on);
      const base = getPendingPreviousState();
      if (base) {
        const diff = computeDiff(base, cacheState);
        updateDiffPanel(diff);
      }
      markDirty({ reason: 'weights' });
    });
  }

  li.addEventListener('keydown', (ev) => {
    if (ev.key === 'ArrowUp') {
      ev.preventDefault();
      moveMetric(key, -1);
    } else if (ev.key === 'ArrowDown') {
      ev.preventDefault();
      moveMetric(key, 1);
    }
  });
}

function renderWeightsUI(state, options = {}) {
  const list = document.getElementById('weightsList');
  if (!list) return { weightChanges: [], toggleChanges: [], orderMoves: [] };

  const prevState = options.prevState ? cloneState(options.prevState) : null;

  if (list.sortableInstance) {
    list.sortableInstance.destroy();
    list.sortableInstance = null;
  }

  if (state) {
    cacheState = normalizeState(state);
  } else {
    cacheState = normalizeState(cacheState);
  }
  SettingsCache.set(cacheState);

  list.innerHTML = '';
  const total = cacheState.order.length;
  cacheState.order.forEach((key, idx) => {
    const def = METRIC_INDEX.get(key) || { label: key, extremes: { left: '', right: '' } };
    const li = document.createElement('li');
    li.className = 'weight-item';
    if (cacheState.enabled[key] === false) li.classList.add('disabled');
    li.dataset.key = key;
    li.setAttribute('role', 'option');
    li.setAttribute('aria-posinset', idx + 1);
    li.setAttribute('aria-setsize', total);
    li.tabIndex = 0;

    const leftLabel = def.extremes?.left || '';
    const rightLabel = def.extremes?.right || '';

    li.innerHTML = `
      <div class="wi-head">
        <span class="wi-rank">#${idx + 1}</span>
        <span class="wi-title">${def.label}</span>
        <button type="button" class="wi-handle" aria-label="Mover ${def.label}">≡</button>
      </div>
      <div class="wi-controls">
        <input id="weight-${key}" class="weight-input" type="number" min="0" max="50" value="${clampInt(cacheState.weights[key], 0, 50, DEFAULT_WEIGHT)}" aria-label="Peso ${def.label}" />
        <label class="wi-toggle"><input type="checkbox" class="wt-enabled" ${cacheState.enabled[key] === false ? '' : 'checked'} aria-label="Activar ${def.label}" /></label>
      </div>
      <div class="wi-meta">
        <small>${leftLabel}</small>
        <small>${rightLabel}</small>
      </div>
    `;
    attachItemEvents(li, key);
    list.appendChild(li);
  });

  if (typeof Sortable !== 'undefined') {
    list.sortableInstance = Sortable.create(list, {
      handle: '.wi-handle',
      animation: 150,
      onEnd: handleSortableEnd,
    });
  }

  const diff = prevState ? computeDiff(prevState, cacheState) : { weightChanges: [], toggleChanges: [], orderMoves: [] };
  if (prevState && options.applyHighlights !== false) {
    applyHighlights(diff, { focusKey: options.focusKey });
  }
  if (options.updateDiffPanel) {
    updateDiffPanel(diff);
  }

  const card = document.getElementById('weightsCard');
  if (card) {
    card.style.display = '';
    card.classList.add('weights-section');
    card.classList.add('compact');
  }
  const footer = document.getElementById('weightsFooter');
  if (footer) footer.style.display = '';
  document.querySelector('.weights-section')?.classList.add('compact');

  return diff;
}

function handleSortableEnd(evt) {
  if (!evt || evt.oldIndex === evt.newIndex) return;
  const list = evt.to || document.getElementById('weightsList');
  if (!list) return;
  const prevState = beginStateChange();
  const items = Array.from(list.children);
  const newOrder = items.map(el => el.dataset.key).filter(Boolean);
  if (!newOrder.length) return;
  cacheState.order = newOrder;
  syncListWithState(cacheState);
  const diff = computeDiff(prevState, cacheState);
  updateDiffPanel(diff);
  applyHighlights(diff, { focusKey: cacheState.order[Math.min(evt.newIndex, cacheState.order.length - 1)] });
  const toastMessage = `Prioridad actualizada: ${formatOrder(cacheState.order)}`;
  markDirty({
    immediate: true,
    reason: 'order',
    toastMessage,
    previousState: prevState,
    state: cloneState(cacheState),
  });
}

function moveMetric(key, delta) {
  const idx = cacheState.order.indexOf(key);
  if (idx === -1) return;
  const target = idx + delta;
  if (target < 0 || target >= cacheState.order.length) return;
  const prevState = beginStateChange();
  const newOrder = cacheState.order.slice();
  const [moved] = newOrder.splice(idx, 1);
  newOrder.splice(target, 0, moved);
  cacheState.order = newOrder;
  syncListWithState(cacheState);
  const diff = computeDiff(prevState, cacheState);
  updateDiffPanel(diff);
  applyHighlights(diff, { focusKey: key });
  const toastMessage = `Prioridad actualizada: ${formatOrder(cacheState.order)}`;
  markDirty({
    immediate: true,
    reason: 'order',
    toastMessage,
    previousState: prevState,
    state: cloneState(cacheState),
  });
}

function resetWeights() {
  const prevState = beginStateChange();
  cacheState = getDefaultState();
  renderWeightsUI(cacheState, { prevState, updateDiffPanel: true });
  markDirty({
    immediate: true,
    reason: 'reset',
    toastMessage: 'Pesos restablecidos',
    previousState: prevState,
    state: cloneState(cacheState),
  });
}

const AWARE_MAP = {
  unaware: 0,
  'problem aware': 0.25,
  'solution aware': 0.5,
  'product aware': 0.75,
  'most aware': 1,
};
function awarenessValue(p) {
  const s = (p.awareness_level || '').toString().trim().toLowerCase();
  return AWARE_MAP[s] ?? 0.5;
}

function stratifiedSampleBy(arr, key, n) {
  if (!Array.isArray(arr) || arr.length <= n) return (arr || []).slice();
  const sorted = [...arr].sort((a, b) => Number(b[key] || 0) - Number(a[key] || 0));
  const out = [];
  for (let i = 0; i < n; i++) {
    const idx = Math.floor(i * (sorted.length - 1) / Math.max(1, (n - 1)));
    out.push(sorted[idx]);
  }
  return out;
}

async function adjustWeightsAI(ev) {
  const btn = ev?.currentTarget || document.getElementById('btnAiWeights');
  const modal = btn?.closest('.modal') || document.querySelector('.config-modal.modal');
  const host = modal?.querySelector('.modal-progress-slot') || modal || document.querySelector('#progress-slot-global');
  const tracker = LoadingHelpers.start('Ajustando pesos con IA', { host });
  if (btn) {
    btn.disabled = true;
    btn.setAttribute('aria-busy', 'true');
  }
  const num = (v) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  };

  let rollbackState = null;
  try {
    const products = Array.isArray(window.allProducts) ? window.allProducts : [];
    if (!products.length) {
      tracker.step(1, 'Sin productos');
      if (typeof toast !== 'undefined' && toast.info) toast.info('No hay productos cargados');
      return;
    }
    tracker.step(0.1, 'Preparando datos');

    const rows = products.map(p => {
      const ratingRaw = p.rating ?? (p.extras && p.extras.rating);
      const unitsRaw = p.units_sold ?? (p.extras && (p.extras['Item Sold'] || p.extras['Orders']));
      const revRaw = p.revenue ?? (p.extras && (p.extras['Revenue($)'] || p.extras['Revenue']));
      return {
        price: num(p.price),
        rating: num(ratingRaw),
        units_sold: num(unitsRaw),
        revenue: num(revRaw),
        desire: num(p.desire_magnitude),
        competition: num(p.competition_level),
        oldness: num(typeof computeOldnessDays === 'function' ? computeOldnessDays(p) : 0),
        awareness: num(typeof awarenessValue === 'function' ? awarenessValue(p) : 0),
      };
    });

    const revCount = rows.filter(r => r.revenue > 0).length;
    const targetName = (revCount >= Math.ceil(rows.length * 0.5)) ? 'revenue' : 'units_sold';
    tracker.step(0.25, 'Seleccionando muestra');

    const cfg = (window.userConfig && window.userConfig.aiCost) ? window.userConfig.aiCost : { costCapUSD: 0.25, estTokensPerItemIn: 300, estTokensPerItemOut: 80 };
    const estTokPerItem = (num(cfg.estTokensPerItemIn) + num(cfg.estTokensPerItemOut)) || 380;
    const pricePerK = 0.002;
    const maxByBudget = Math.max(30, Math.floor((cfg.costCapUSD || 0.25) / pricePerK * 1000 / Math.max(1, estTokPerItem)));
    const HARD_CAP = 500;
    const MAX = Math.min(HARD_CAP, Math.max(60, maxByBudget));

    let dataSample = rows.map(r => ({ ...r, target: r[targetName] }));
    if (dataSample.length > MAX) dataSample = stratifiedSampleBy(dataSample, targetName, MAX);

    tracker.step(0.45, 'Consultando IA');

    const res = await fetch('/api/config/winner-weights/ai?can_reorder=true', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: METRIC_KEYS, target: targetName, data_sample: dataSample }),
      __hostEl: host,
    });
    if (!res.ok) throw new Error('auto_weights_failed');
    const out = await res.json();

    const prevState = beginStateChange();
    rollbackState = cloneState(prevState);
    const mergedWeights = { ...cacheState.weights, ...(out.weights || out.winner_weights || {}) };
    const nextState = normalizeState({
      weights: mergedWeights,
      order: out.order || out.winner_order || cacheState.order,
      weights_enabled: cacheState.enabled,
    });

    cacheState = nextState;
    const diff = renderWeightsUI(cacheState, { prevState, updateDiffPanel: true });
    applyHighlights(diff);

    tracker.step(0.75, 'Guardando cambios');

    await markDirty({
      immediate: true,
      reason: 'ai',
      toastMessage: 'IA actualizó pesos (+orden) · guardado',
      previousState: prevState,
      state: cloneState(cacheState),
    });

    rollbackState = null;
    tracker.step(1, 'Completado');
  } catch (err) {
    if (rollbackState) {
      cacheState = rollbackState;
      renderWeightsUI(cacheState, { updateDiffPanel: true, applyHighlights: false });
    }
    if (!err || (err && err.message !== 'save_failed')) {
      if (typeof toast !== 'undefined' && toast.error) {
        toast.error('No se pudo ajustar por IA. Revisa tu API Key o inténtalo más tarde.');
      }
    }
  } finally {
    tracker.done();
    if (btn) {
      btn.disabled = false;
      btn.removeAttribute('aria-busy');
    }
    clearPendingPreviousState();
  }
}

function showSettingsModalShell() {
  const list = document.getElementById('weightsList');
  if (list) list.innerHTML = '';
}

async function hydrateSettingsModal() {
  try {
    const state = await SettingsCache.get();
    cacheState = cloneState(state);
    renderWeightsUI(cacheState, { applyHighlights: false, updateDiffPanel: false });
    updateDiffPanel({ weightChanges: [], toggleChanges: [], orderMoves: [] });
    console.debug('hydrateSettingsModal -> weights/order aplicados:', state);
  } catch (err) {
    // silencioso
  }
}

async function openConfigModal() {
  showSettingsModalShell();
  await hydrateSettingsModal();
  document.querySelector('#configModal')?.classList.add('ready');
  document.querySelector('#settings-modal')?.classList.add('ready');
  const resetBtn = document.getElementById('btnReset');
  if (resetBtn) resetBtn.onclick = resetWeights;
  const aiBtn = document.getElementById('btnAiWeights');
  if (aiBtn) aiBtn.onclick = adjustWeightsAI;
}

window.openConfigModal = openConfigModal;
window.loadWeights = hydrateSettingsModal;
window.resetWeights = resetWeights;
window.adjustWeightsAI = adjustWeightsAI;
window.markDirty = markDirty;
window.metricKeys = METRIC_KEYS;
