function num(v) { return Number(v || 0); }
const fmtEu = (v) => {
  if (!isFinite(v)) return '-';
  if (v >= 1e6) return '€' + (v / 1e6).toFixed(2).replace('.', ',') + ' M';
  if (v >= 1e3) return '€' + (v / 1e3).toFixed(1).replace('.', ',') + ' K';
  return '€' + v.toFixed(2).replace('.', ',');
};
const fmtUnits = (v) => {
  v = num(v);
  if (v >= 1e6) return (v / 1e6).toFixed(2).replace('.', ',') + ' M';
  if (v >= 1e3) return (v / 1e3).toFixed(1).replace('.', ',') + ' K';
  return v.toLocaleString('es-ES');
};
const SEP = ' - - - ';
function joinSep(list){ return list.filter(Boolean).join(SEP); }
const qtile = (arr, q) => {
  if (!arr.length) return 0;
  const a = [...arr].sort((x, y) => x - y);
  const i = (a.length - 1) * q;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  if (lo === hi) return a[lo];
  return a[lo] * (hi - i) + a[hi] * (i - lo);
};

function extraordinaryProducts(products) {
  const prices = products.map((p) => num(p.price)).filter(Boolean);
  const units = products.map((p) => num(p.units_sold || p.units)).filter(Boolean);
  const ratings = products.map((p) => num(p.rating)).filter(Boolean);

  const p75 = qtile(prices, 0.75);
  const u75 = qtile(units, 0.75);
  const r85 = qtile(ratings, 0.85);

  const priceyAndSell = products
    .filter((p) => num(p.price) >= p75 && num(p.units_sold || p.units) >= u75)
    .slice(0, 3)
    .map((p) => `${p.name} (${fmtEu(num(p.revenue))})`);

  const fanFavs = products
    .filter((p) => num(p.rating) >= Math.max(4.6, r85) && num(p.units_sold || p.units) >= u75)
    .slice(0, 3)
    .map((p) => `${p.name} (${num(p.rating).toFixed(2).replace('.', ',')}★)`);

  return { priceyAndSell, fanFavs };
}

function categoryBullets(categories){
  if (!categories?.length) return [];
  const total = categories.reduce((s,c)=> s + num(c.revenue), 0) || 1;
  const topRev = [...categories]
    .sort((a,b)=> num(b.revenue) - num(a.revenue))
    .slice(0,3)
    .map(c => {
      const pct = ((100 * num(c.revenue)) / total).toFixed(1).replace('.', ',');
      return `${c.path || c.name} (${pct}%)`;
    })
    .filter(Boolean);
  return topRev.length ? [`Top categorías por ingresos: ${joinSep(topRev)}`] : [];
}

function productBullets(products){
  if (!products?.length) return [];
  const prepared = products.map((p) => ({
    ...p,
    name: p.name || p.title || p.product_name || 'Producto',
    revenue: num(p.revenue),
    units_sold: num(p.units_sold ?? p.units ?? p.quantity ?? p.total_units),
    units: num(p.units ?? p.units_sold ?? p.quantity ?? p.total_units),
    price: num(p.price),
    rating: num(p.rating)
  }));

  const topByRevenue = [...prepared]
    .sort((a,b)=> b.revenue - a.revenue)
    .slice(0,3)
    .filter((p) => p.revenue > 0)
    .map((p) => `${p.name} (${fmtEu(p.revenue)})`);
  const topByUnits = [...prepared]
    .sort((a,b)=> b.units_sold - a.units_sold)
    .slice(0,3)
    .filter((p) => p.units_sold > 0)
    .map((p) => `${p.name} (${fmtUnits(p.units_sold)} uds)`);
  const { priceyAndSell, fanFavs } = extraordinaryProducts(prepared);

  const out = [];
  if (topByRevenue.length) out.push(`Productos top por ingresos: ${joinSep(topByRevenue)}`);
  if (topByUnits.length)   out.push(`Productos top por unidades: ${joinSep(topByUnits)}`);
  if (priceyAndSell.length) out.push(`Caros y venden mucho: ${joinSep(priceyAndSell)}`);
  if (fanFavs.length)       out.push(`Favoritos por valoración y ventas: ${joinSep(fanFavs)}`);
  return out;
}

// writeInsights: 4–6 viñetas máximo (sin bullets anidados)
function writeInsights(lines){
  const box = document.getElementById('insightsContent');
  if (!box) return;
  const trimmed = lines.filter(Boolean).slice(0, 6);
  if (!trimmed.length){
    box.innerHTML = '<p class="muted">Sin datos para generar insights.</p>';
    return;
  }
  box.innerHTML = '<ul>' + trimmed.map((l) => `<li>${l}</li>`).join('') + '</ul>';
}

function getData() {
  const agg = window.__latestTrendsData?.categoriesAgg || [];
  const all = window.__latestTrendsData?.allProducts || [];
  return { agg, all };
}

document.addEventListener('click', (ev) => {
  if (ev.target?.id !== 'btnLocalInsights') return;
  const { agg, all } = getData();
  const lines = [...categoryBullets(agg), ...productBullets(all)];
  writeInsights(lines);
});

export {};
