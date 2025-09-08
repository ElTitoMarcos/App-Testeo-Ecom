const EC_BA_CONCURRENCY = 3;
const btn = document.getElementById('btn-completar-ia');

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

async function processProduct(product) {
  const payload = {
    id: product.id,
    name: product.name,
    category: product.category,
    price: product.price,
    rating: product.rating,
    units_sold: product.units_sold,
    revenue: product.revenue,
    conversion_rate: product.conversion_rate,
    launch_date: product.launch_date,
    date_range: product.date_range,
    image_url: product.image_url || null,
    desire: product.desire,
    desire_magnitude: product.desire_magnitude,
    awareness_level: product.awareness_level,
    competition_level: product.competition_level
  };
  try {
    const res = await fetch('/api/ba/insights', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product: payload, model: 'gpt-4o-mini-2024-07-18' })
    });
    if (!res.ok) {
      let msg = res.statusText;
      try { const err = await res.json(); if (err.error) msg = err.error; } catch {}
      throw new Error(msg);
    }
    const data = await res.json();
    applyUpdates(product, data.grid_updates || {});
    toast.success(`Completado ID ${product.id}`, { duration: 2000 });
    return true;
  } catch (e) {
    const msg = e && e.message ? e.message : 'Error';
    toast.error(`Falló ID ${product.id}: ${msg}`, { duration: 2000 });
    return false;
  }
}

async function runQueue(products) {
  let ok = 0;
  const total = products.length;
  const queue = products.slice();
  async function worker() {
    while (queue.length) {
      const p = queue.shift();
      if (await processProduct(p)) ok++;
    }
  }
  const workers = [];
  for (let i = 0; i < EC_BA_CONCURRENCY; i++) workers.push(worker());
  await Promise.all(workers);
  toast.info(`IA: ${ok}/${total}`);
}

if (btn) {
  btn.addEventListener('click', async () => {
    if (selection.size === 0) {
      toast.info('Selecciona al menos un producto');
      return;
    }
    btn.disabled = true;
    const ids = Array.from(selection);
    const products = ids.map(id => (window.products || []).find(p => String(p.id) === id)).filter(Boolean);
    await runQueue(products);
    btn.disabled = false;
    updateMasterState();
  });
}
