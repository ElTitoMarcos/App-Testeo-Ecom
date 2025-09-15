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
const pct = (value, total) => {
  const totalNum = Number(total);
  const val = num(value);
  const pctVal = totalNum > 0 ? (100 * val) / totalNum : 0;
  return pctVal.toFixed(1).replace('.', ',') + '%';
};
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

function buildCategoryList(categories){
  if (!categories?.length) return [];
  const total = categories.reduce((sum, c) => sum + num(c.revenue), 0);
  return [...categories]
    .sort((a, b) => num(b.revenue) - num(a.revenue))
    .slice(0, 3)
    .map((c) => `${c.path || c.name} (${pct(c.revenue, total)})`)
    .filter(Boolean);
}

function buildProductLists(products){
  if (!products?.length) {
    return { prodTopRev: [], prodTopUnits: [], priceyAndSell: [], fanFavs: [] };
  }
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
    .sort((a, b) => b.revenue - a.revenue)
    .slice(0, 3)
    .filter((p) => p.revenue > 0);
  const topByUnits = [...prepared]
    .sort((a, b) => b.units_sold - a.units_sold)
    .slice(0, 3)
    .filter((p) => p.units_sold > 0);
  const { priceyAndSell, fanFavs } = extraordinaryProducts(prepared);

  return {
    prodTopRev: topByRevenue.map((p) => `${p.name} (${fmtEu(p.revenue)})`),
    prodTopUnits: topByUnits.map((p) => `${p.name} (${fmtUnits(p.units_sold)} uds)`),
    priceyAndSell,
    fanFavs
  };
}

function renderBlock(title, items){
  if (!items?.length) return '';
  const lis = items.map((t) => `<li>${t}</li>`).join('');
  return `<div class="insight-block"><div class="insight-title">${title}</div><ul>${lis}</ul></div>`;
}

function writeInsightsBlocks({ catTop = [], prodTopRev = [], prodTopUnits = [], priceyAndSell = [], fanFavs = [] }){
  const box = document.getElementById('insightsContent');
  if (!box) return;
  const html =
    renderBlock('Top categorías por ingresos:', catTop) +
    renderBlock('Productos top por ingresos:', prodTopRev) +
    renderBlock('Productos top por unidades:', prodTopUnits) +
    renderBlock('Caros y venden mucho:', priceyAndSell) +
    renderBlock('Favoritos por valoración y ventas:', fanFavs);
  box.innerHTML = html || '<p class="muted">Sin datos para generar insights.</p>';
}

function getData() {
  const scope = window.__latestTrendsData;
  if (scope?.categoriesAgg?.length || scope?.allProducts?.length) {
    return scope;
  }
  if (typeof window.computeTrendsScope === 'function') {
    return window.computeTrendsScope();
  }
  return { categoriesAgg: [], allProducts: [] };
}

document.addEventListener('click', (ev) => {
  if (ev.target?.id !== 'btnLocalInsights') return;
  const { categoriesAgg, allProducts } = getData();
  const catTop = buildCategoryList(categoriesAgg);
  const { prodTopRev, prodTopUnits, priceyAndSell, fanFavs } = buildProductLists(allProducts);
  const blocks = { catTop, prodTopRev, prodTopUnits, priceyAndSell, fanFavs };
  writeInsightsBlocks(blocks);
});

export {};
