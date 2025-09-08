const EC_BATCH_SIZE = 10;
const EC_MODEL = "gpt-4o-mini-2024-07-18";
let EC_IA_LOADING = false;

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

function applyBatchGridUpdates(okMap) {
  const ids = Object.keys(okMap || {});
  ids.forEach(id => {
    const product = (window.products || []).find(p => String(p.id) === String(id));
    if (product) applyUpdates(product, okMap[id]);
  });
  return ids.length;
}

async function handleCompletarIA(){
  if (EC_IA_LOADING) return;
  EC_IA_LOADING = true;
  const btn = document.getElementById('btn-completar-ia');
  if (btn){
    btn.disabled = true;
    btn.setAttribute('aria-disabled','true');
    btn.setAttribute('aria-busy','true');
    btn.dataset.label = btn.dataset.label || btn.textContent.trim();
    btn.innerHTML = '<span class="ec-spinner" aria-hidden="true"></span><span>Cargando…</span>';
  }
  try{
    const items = await getAllFilteredRows();
    if (!items || !items.length){
      toast.info('No hay productos en la lista actual.');
      return;
    }
    const CHUNK = window.EC_BATCH_SIZE || EC_BATCH_SIZE;
    let okTotal = 0, koTotal = 0;
    for (let i=0; i<items.length; i+=CHUNK){
      const slice = items.slice(i, i+CHUNK);
      const payload = slice.map(p => ({
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
        image_url: (p.image_url && /^https?:/i.test(p.image_url)) ? p.image_url : null
      }));
      const res = await fetch('/api/ia/batch-columns', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ model: window.EC_MODEL || EC_MODEL, items: payload })
      });
      if (!res.ok){
        koTotal += slice.length;
        toast.error(`IA lote falló (${res.status})`);
        continue;
      }
      let data;
      try { data = await res.json(); } catch { data = null; }
      if (!data){
        koTotal += slice.length;
        toast.error('IA lote: JSON inválido');
        continue;
      }
      const oks = Object.keys(data.ok || {});
      const kos = Object.keys(data.ko || {});
      okTotal += applyBatchGridUpdates(data.ok);
      koTotal += kos.length;
      toast.info(`IA lote: +${oks.length} / ${slice.length} (fallos ${kos.length})`);
    }
    toast.success(`IA: ${okTotal}/${items.length} completados`);
    if (typeof updateMasterState === 'function') updateMasterState();
  } catch(err){
    console.error(err);
    toast.error('IA: error inesperado');
  } finally {
    if (btn){
      btn.disabled = false;
      btn.removeAttribute('aria-disabled');
      btn.removeAttribute('aria-busy');
      btn.innerHTML = btn.dataset.label || 'Completar columnas (IA)';
    }
    EC_IA_LOADING = false;
  }
}

document.getElementById('btn-completar-ia')?.addEventListener('click', handleCompletarIA);
