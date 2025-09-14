(() => {
  const log = (...a) => (console.debug('[AI-weights]', ...a), a);
  const info = (m) => (window.toast?.info?.(m) ?? window.showToast?.(m) ?? console.log(m));
  const errorT = (m) => (window.toast?.error?.(m) ?? window.showToast?.(m,'error') ?? console.error(m));

  function getRows() {
    try {
      if (typeof window.getAllFilteredRows === 'function') return window.getAllFilteredRows() || [];
    } catch {}
    return Array.isArray(window.products) ? window.products.slice() : [];
    }

  function buildSamples(max = 100) {
    const all = getRows();
    return all.slice(0, max).map(p => {
      const price = +p.price || 0;
      const units = +p.units_sold || 0;
      const revenue = Number.isFinite(+p.revenue) ? +p.revenue : (price * units);
      return {
        price,
        rating: +p.rating || 0,
        units_sold: units,
        revenue,
        desire: +p.desire || 0,
        competition: +p.competition || 0,
        oldness: +p.oldness || 0,
        awareness: +p.awareness || 0,
      };
    });
  }

  async function callAI(samples) {
    const res = await fetch('/api/config/winner-weights/ai', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ samples, success_key: 'revenue' })
    });
    if (!res.ok) {
      let j = {};
      try { j = await res.json(); } catch {}
      if (j.error === 'missing_api_key' && typeof window.openApiKeyModal === 'function') {
        window.openApiKeyModal();
      }
      throw new Error(`HTTP ${res.status}`);
    }
    return res.json();
  }

  function applyResult(data) {
    const weights = data?.winner_weights || {};
    const order   = data?.winner_order   || Object.keys(weights);
    log('applyResult', { weights, order, effective: data?.effective });

    // 1) sliders 0..100 crudos (no normalizar)
    for (const [k, v] of Object.entries(weights)) {
      window.setSliderValue?.(k, v);
      window.setToggleEnabled?.(k, true);
    }

    // 2) ordenar UI por prioridad (arriba = más importante)
    window.reorderWeightsUI?.(order);

    // 3) pintar “efectivos” si existe
    if (data?.effective?.int) {
      window.renderEffectiveBadges?.(data.effective.int);
    }
  }

  async function onClick(ev) {
    ev?.preventDefault?.();
    ev?.stopPropagation?.();
    const btn = ev?.currentTarget || document.querySelector('#btn-ai-weights') || document.querySelector('[data-action="ai-weights"]');
    try {
      btn && (btn.disabled = true, btn.classList?.add('is-loading'));
      info('Ajustando pesos con IA…');
      const samples = buildSamples(100);
      log('samples', samples.length, samples[0]);
      const data = await callAI(samples);
      applyResult(data);
      info('Pesos ajustados con IA');
    } catch (e) {
      console.error(e);
      errorT('Error al ajustar pesos con IA');
    } finally {
      btn && (btn.disabled = false, btn.classList?.remove('is-loading'));
    }
  }

  function bind() {
    const btn = document.querySelector('#btn-ai-weights') || document.querySelector('[data-action="ai-weights"]');
    if (!btn || btn.dataset.bound === '1') return;
    btn.dataset.bound = '1';
    btn.type = 'button';
    btn.addEventListener('click', onClick);
    log('handler bound to', btn);
  }

  function ready(fn){ document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', fn) : fn(); }
  ready(bind);
  document.addEventListener('app:config:open', bind); // por si el modal se monta tarde

  // helper manual en consola
  window.aiWeightsTest = () => onClick({ preventDefault(){}, stopPropagation(){}, currentTarget: document.querySelector('#btn-ai-weights') });
})();

