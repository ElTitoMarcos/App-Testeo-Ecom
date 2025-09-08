const EC_BATCH_SIZE = 10;
const EC_MODEL = "gpt-4o-mini-2024-07-18";

function getAllFilteredRows() {
  if (typeof window.getAllFilteredRows === 'function') {
    try {
      return window.getAllFilteredRows();
    } catch {
      return [];
    }
  }
  return Array.isArray(window.products) ? window.products.slice() : [];
}

function isEditing(pid, field) {
  const active = document.activeElement;
  if (!active) return false;
  const tr = active.closest('tr');
  if (!tr) return false;
  const cb = tr.querySelector('input.rowCheck');
  if (!cb || cb.dataset.id !== String(pid)) return false;
  const td = active.closest('td[data-key]');
  if (!td) return false;
  return td.dataset.key === field;
}

function applyUpdates(product, updates) {
  const applied = {};
  const row = document.querySelector(`input.rowCheck[data-id="${product.id}"]`)?.closest('tr');
  const map = {
    desire: 'td.ec-col-desire input',
    desire_magnitude: 'td.ec-col-desire-mag select',
    awareness_level: 'td.ec-col-awareness select',
    competition_level: 'td.ec-col-competition select'
  };
  Object.keys(map).forEach(k => {
    const nv = updates[k];
    if (nv === undefined || isEditing(product.id, k)) return;
    product[k] = nv;
    const el = row ? row.querySelector(map[k]) : null;
    if (el) {
      if (el.tagName === 'INPUT') el.value = nv || '';
      else el.value = nv || '';
    }
    applied[k] = nv;
  });
  if (Object.keys(applied).length) {
    fetch(`/products/${product.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(applied)
    }).catch(() => {});
    if (window.ecAutoFitColumns && window.gridRoot) ecAutoFitColumns(gridRoot);
  }
  return applied;
}

function chunkArray(arr, size) {
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

async function processBatch(items) {
  const res = await fetch('/api/ia/batch-columns', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EC_MODEL, items })
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { const err = await res.json(); if (err.error) msg = err.error; } catch {}
    throw new Error(msg);
  }
  const data = await res.json();
  let ok = 0;
  let ko = 0;
  const okMap = data.ok || {};
  const koMap = data.ko || {};
  Object.keys(okMap).forEach(id => {
    const product = (window.products || []).find(p => String(p.id) === String(id));
    if (product) {
      applyUpdates(product, okMap[id]);
      ok++;
    } else {
      ko++;
    }
  });
  ko += Object.keys(koMap).length;
  return { ok, ko };
}

window.handleCompletarIA = async function(opts = {}) {
  const ids = opts.ids;
  let all;
  if (ids && Array.isArray(ids)) {
    all = (Array.isArray(window.products) ? window.products : []).filter(p => ids.includes(p.id));
  } else {
    all = getAllFilteredRows();
  }
  if (all.length === 0) {
    if (!opts.silent) toast.info('No hay productos');
    return;
  }
  let okTotal = 0;
  const chunks = chunkArray(all, EC_BATCH_SIZE);
  for (const ch of chunks) {
    const payload = ch.map(p => ({
      id: p.id,
      name: p.name,
      category: p.category,
      price: p.price,
      rating: p.rating,
      units_sold: p.units_sold,
      revenue: p.revenue,
      conversion_rate: p.conversion_rate,
      launch_date: p.launch_date,
      date_range: p.date_range,
      image_url: p.image_url || null
    }));
      try {
        const { ok, ko } = await processBatch(payload);
        okTotal += ok;
        if (!opts.silent) toast.info(`IA lote: +${ok} / ${payload.length} (fallos ${ko})`, { duration: 2000 });
      } catch (e) {
        if (!opts.silent) toast.error(`IA lote: ${e.message}`, { duration: 2000 });
      }
  }
    if (!opts.silent) toast.info(`IA: ${okTotal}/${all.length} completados`);
  updateMasterState();
};

window.EC_IA = window.EC_IA || {};

(function(ns){
  ns.setLoading = function(btn, on){
    if (!btn) return;
    if (on){
      btn.disabled = true;
      btn.setAttribute('aria-disabled','true');
      btn.setAttribute('aria-busy','true');
      if (!btn.dataset.label) btn.dataset.label = btn.textContent.trim();
      btn.classList.add('ec-btn-loading');
      btn.innerHTML = '<span class="ec-spinner" aria-hidden="true"></span><span>Cargando…</span>';
      btn.dataset.loading = '1';
    } else {
      btn.disabled = false;
      btn.removeAttribute('aria-disabled');
      btn.removeAttribute('aria-busy');
      btn.classList.remove('ec-btn-loading');
      btn.innerHTML = btn.dataset.label || 'Completar columnas (IA)';
      btn.dataset.loading = '';
    }
  };

  ns.runCompletarIA = async function(opts){
      if (typeof window.handleCompletarIA === 'function'){
        return await window.handleCompletarIA(opts);
      }
      if (typeof window.runCompletarIA === 'function'){
        return await window.runCompletarIA(opts);
      }
      if (window.toast && toast.error) toast.error('No se encontró el flujo de IA.');
      else console.error('No se encontró el flujo de IA.');
    };

  ns._bound = ns._bound || null;
  ns.bindButton = function(){
    const btn = document.getElementById('btn-completar-ia');
    if (!btn) return;
    if (ns._bound){
      btn.removeEventListener('click', ns._bound);
    }
    ns._bound = async function(ev){
      ev.preventDefault();
      const b = ev.currentTarget;
      if (b.dataset.loading === '1') return;
      try{
        ns.setLoading(b, true);
        await ns.runCompletarIA();
      } catch(err){
        console.error(err);
        if (window.toast && toast.error) toast.error('IA: error inesperado');
      } finally {
        ns.setLoading(b, false);
      }
    };
    btn.addEventListener('click', ns._bound);
  };

  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', ns.bindButton, {once:true});
  } else {
    ns.bindButton();
  }
  if (!ns._observer){
    ns._observer = new MutationObserver(() => {
      ns.bindButton();
    });
    ns._observer.observe(document.body, {childList:true, subtree:true});
  }
})(window.EC_IA);
