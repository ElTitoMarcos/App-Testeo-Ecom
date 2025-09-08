const EC_BATCH_SIZE = 10;
const EC_MODEL = "gpt-4o-mini-2024-07-18";
const btn = document.getElementById('btn-completar-ia');

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

if (btn) {
  btn.addEventListener('click', async () => {
    const all = getAllFilteredRows();
    if (all.length === 0) {
      toast.info('No hay productos');
      return;
    }
    btn.disabled = true;
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
        toast.info(`IA lote: +${ok} / ${payload.length} (fallos ${ko})`, { duration: 2000 });
      } catch (e) {
        toast.error(`IA lote: ${e.message}`, { duration: 2000 });
      }
    }
    btn.disabled = false;
    toast.info(`IA: ${okTotal}/${all.length} completados`);
    updateMasterState();
  });
}
