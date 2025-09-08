const EC_BA_CONCURRENCY = 3;
const btn = document.getElementById('btn-ba-gpt');

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
    const resp = await fetch('/api/ba/insights', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product: payload, model: 'gpt-4o-mini' })
    });
    if (!resp.ok) throw new Error();
    const data = await resp.json();
    const updates = data.grid_updates || {};
    const applied = {};
    ['desire', 'desire_magnitude', 'awareness_level', 'competition_level'].forEach(k => {
      if (updates[k] !== undefined && !isEditing(product.id, k)) {
        product[k] = updates[k];
        applied[k] = updates[k];
      }
    });
    if (Object.keys(applied).length) {
      renderTable();
      try {
        await fetch(`/products/${product.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(applied)
        });
      } catch (e) {}
    }
    toast.success(`BA listo: ID ${product.id}`, { duration: 2000 });
    return true;
  } catch (e) {
    toast.error(`BA falló: ID ${product.id}`, { duration: 2000 });
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
  toast.info(`BA: ${ok}/${total}`);
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
