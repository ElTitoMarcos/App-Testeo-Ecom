const EC_MODEL = 'gpt-4o-mini-2024-07-18';
const CTX_BUDGET = 48000;
const ITEM_BASE = 420;
const ITEM_IMG = 180;

async function getAllFilteredRows() {
  const all = window.allProducts || [];
  if (typeof window.applyCurrentFilters === 'function') {
    return window.applyCurrentFilters(all);
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

function splitByBudget(items) {
  let budget = 800;
  let current = [];
  const chunks = [];
  items.forEach(p => {
    const cost = ITEM_BASE + (p.image_url ? ITEM_IMG : 0);
    if (budget + cost > CTX_BUDGET && current.length) {
      chunks.push(current);
      current = [];
      budget = 800;
    }
    current.push(p);
    budget += cost;
  });
  if (current.length) chunks.push(current);
  return chunks;
}

async function processBatch(items) {
  const res = await fetch('/api/ia/batch-columns', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EC_MODEL, items })
  });
  if (res.status === 503) {
    try {
      const err = await res.json();
      if (err.error === 'OPENAI_MISSING') throw new Error('OPENAI_MISSING');
      throw new Error(err.error || res.statusText);
    } catch {
      throw new Error('OPENAI_MISSING');
    }
  }
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

window.handleCompletarIA = async function() {
  if (window.OPENAI_ENABLED === false) {
    if (window.toast && toast.error) toast.error('Falta API de OpenAI. Añádela en Configuración para usar IA.');
    return;
  }
  const items = await getAllFilteredRows();
  if (!items || !items.length) {
    toast.info('No hay productos en la lista actual.');
    return;
  }
  const est = items.reduce((s, p) => s + ITEM_BASE + (p.image_url ? ITEM_IMG : 0), 800);
  const chunks = est <= CTX_BUDGET ? [items] : splitByBudget(items);
  let okTotal = 0;
  let aborted = false;
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
      toast.info(`IA lote: +${ok} / ${payload.length} (fallos ${ko})`, { duration: 2000 });
    } catch (e) {
      if (e.message === 'OPENAI_MISSING') {
        if (window.toast && toast.error) toast.error('Falta API de OpenAI. Añádela en Configuración para usar IA.');
        aborted = true;
        break;
      }
      toast.error(`IA lote: ${e.message}`, { duration: 2000 });
    }
  }
  if (!aborted) {
    toast.info(`IA: ${okTotal}/${items.length} completados`);
    updateMasterState();
  }
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

  ns.runCompletarIA = async function(){
    if (window.OPENAI_ENABLED === false) {
      if (window.toast && toast.error) toast.error('Falta API de OpenAI. Añádela en Configuración para usar IA.');
      return;
    }
    if (typeof window.handleCompletarIA === 'function'){
      return await window.handleCompletarIA({silent:true});
    }
    if (typeof window.runCompletarIA === 'function'){
      return await window.runCompletarIA();
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
