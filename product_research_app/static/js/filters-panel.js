const toNumber = (raw) => {
  if (raw == null || raw === '') return NaN;
  let s = String(raw).trim();
  let mul = 1;
  if (/m$/i.test(s)) { mul = 1e6; s = s.replace(/m/i,''); }
  if (/k$/i.test(s)) { mul = 1e3; s = s.replace(/k/i,''); }
  s = s.replace(/[€$]/g,'').replace(/\./g,'').replace(',', '.').replace('%','');
  const v = parseFloat(s);
  return isNaN(v) ? NaN : v * mul;
};

const normLevel = (s) => (s || '').toLowerCase().replace(/[\s-]/g, '');
const toPercent = (raw) => {
  const n = toNumber(raw);
  return isNaN(n) ? NaN : n;
};

const parseIdQuery = (txt) => {
  const out = new Set();
  (txt||'').split(',').map(s=>s.trim()).forEach(tok=>{
    if (!tok) return;
    const m = tok.match(/^(\d+)\s*-\s*(\d+)$/);
    if (m){ let a=+m[1], b=+m[2]; if (a>b) [a,b]=[b,a]; for(let i=a;i<=b;i++) out.add(i); }
    else if (/^\d+$/.test(tok)) out.add(+tok);
  });
  return out;
};

const val = (id) => document.getElementById(id)?.value?.trim() ?? '';

const awarenessLabel = (value) => {
  if (!value) return '';
  const select = document.getElementById('f-awareness');
  if (!select) return value;
  const option = Array.from(select.options).find(opt => opt.value === value);
  return option?.textContent?.trim() || value;
};
export function readFilters(){
  return {
    ids: parseIdQuery(val('f-id')),
    category: val('f-category').toLowerCase(),
    priceMin: toNumber(val('f-price-min')),
    priceMax: toNumber(val('f-price-max')),
    ratingMin: toNumber(val('f-rating-min')),
    ratingMax: toNumber(val('f-rating-max')),
    unitsMin: toNumber(val('f-units-min')),
    unitsMax: toNumber(val('f-units-max')),
    revenueMin: toNumber(val('f-revenue-min')),
    revenueMax: toNumber(val('f-revenue-max')),
    convMin: toPercent(val('f-conv-min')),
    convMax: toPercent(val('f-conv-max')),
    dateFrom: val('f-date-from'),
    dateTo: val('f-date-to'),
    rangeText: val('f-range-text').toLowerCase(),
    desireMag: val('f-desire-mag'),
    awareness: normLevel(val('f-awareness')), // ya viene normalizado del select
    competition: val('f-competition'),
    scoreMin: toNumber(val('f-score-min')),
    scoreMax: toNumber(val('f-score-max')),
  };
}

const getAppState = () => {
  window.appState = window.appState || {};
  return window.appState;
};

const extraValue = (p, ...keys) => {
  if (!p || !p.extras) return undefined;
  for (const key of keys) {
    if (p.extras[key] != null && p.extras[key] !== '') {
      return p.extras[key];
    }
  }
  return undefined;
};

const F = {
  id: p => Number(p?.id ?? p?.ID ?? p?.idx ?? extraValue(p, 'ID') ?? NaN),
  category: p => ((p?.category_path || p?.category || p?.path || extraValue(p, 'Category', 'Categoría')) || '').toLowerCase(),
  price: p => toNumber(p?.price ?? extraValue(p, 'Price', 'Precio', 'Product Price')), 
  rating: p => toNumber(p?.rating ?? extraValue(p, 'Product Rating', 'Rating')), 
  units: p => toNumber(p?.units_sold ?? p?.units ?? p?.total_units ?? p?.quantity ?? extraValue(p, 'Item Sold', 'Units Sold', 'Orders', 'Sales Units', 'Quantity')),
  revenue: p => toNumber(p?.revenue ?? extraValue(p, 'Revenue($)', 'Revenue', 'Ingresos', 'GMV')),
  conv: p => toNumber(p?.conversion_rate ?? p?.conversion ?? extraValue(p, 'Conversion Rate', 'Conv Rate', 'Conversion %', 'CVR')),
  date: p => (p?.launch_date || p?.date || extraValue(p, 'Launch Date', 'Fecha Lanzamiento', 'LaunchDate') || ''),
  range: p => ((p?.range_label || p?.range || p?.date_range || extraValue(p, 'Date Range', 'Rango Fechas')) || '').toLowerCase(),
  desireMag: p => (p?.desire_magnitude || p?.desireMag || extraValue(p, 'Desire Magnitude') || '').trim(),
  awareness: p => (p?.awareness_level || p?.awareness || extraValue(p, 'Awareness Level') || '').trim(),
  competition: p => (p?.competition_level || p?.competition || extraValue(p, 'Competition Level') || '').trim(),
  score: p => toNumber(p?.winner_score ?? p?.score ?? extraValue(p, 'Winner Score')), 
};

const inRange = (v, min, max) => {
  const hasMin = !isNaN(min);
  const hasMax = !isNaN(max);
  if (!hasMin && !hasMax) return true;
  if (!Number.isFinite(v)) return false;
  if (hasMin && v < min) return false;
  if (hasMax && v > max) return false;
  return true;
};

const inDate = (iso, from, to) => {
  if (!iso) return false;
  if (from && iso < from) return false;
  if (to && iso > to) return false;
  return true;
};

export function applyFilters(products, filters){
  const list = Array.isArray(products) ? products : [];
  const f = filters || readFilters();
  const idSet = f.ids instanceof Set ? f.ids : parseIdQuery(f.ids);
  return list.filter(p => {
    if (idSet && idSet.size){
      const pid = F.id(p);
      if (!idSet.has(pid)) return false;
    }
    if (f.category && !F.category(p).includes(f.category)) return false;
    if (!inRange(F.price(p),   f.priceMin,   f.priceMax)) return false;
    if (!inRange(F.rating(p),  f.ratingMin,  f.ratingMax)) return false;
    if (!inRange(F.units(p),   f.unitsMin,   f.unitsMax)) return false;
    if (!inRange(F.revenue(p), f.revenueMin, f.revenueMax)) return false;
    if (!inRange(F.conv(p),    f.convMin,    f.convMax)) return false;
    if (!inRange(F.score(p),   f.scoreMin,   f.scoreMax)) return false;
    if ((f.dateFrom || f.dateTo) && !inDate(F.date(p), f.dateFrom, f.dateTo)) return false;
    if (f.rangeText && !F.range(p).includes(f.rangeText)) return false;
    if (f.desireMag && F.desireMag(p) !== f.desireMag) return false;
    if (f.awareness) {
      if (normLevel(F.awareness(p)) !== f.awareness) return false;
    }
    if (f.competition && F.competition(p) !== f.competition) return false;
    return true;
  });
}

const drawerEl = () => document.getElementById('filtersDrawer');
const toggleDrawer = () => {
  const drawer = drawerEl();
  if (!drawer) return;
  drawer.classList.toggle('hidden');
};
const closeDrawer = () => {
  const drawer = drawerEl();
  if (!drawer) return;
  drawer.classList.add('hidden');
};

const isTextInput = (el) => {
  if (!el) return false;
  const tag = el.tagName ? el.tagName.toLowerCase() : '';
  return tag === 'input' || tag === 'textarea' || tag === 'select' || el.isContentEditable === true;
};

const fmtNumber = (n) => {
  if (!Number.isFinite(n)) return '';
  const abs = Math.abs(n);
  if (abs >= 1e6) return `${(n / 1e6).toFixed(abs >= 1e8 ? 0 : 1).replace(/\.0$/, '')}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(abs >= 1e5 ? 0 : 1).replace(/\.0$/, '')}K`;
  return n.toLocaleString('es-ES', { maximumFractionDigits: n % 1 ? 2 : 0 });
};
const fmtCurrency = (n) => {
  const base = fmtNumber(n);
  return base ? `€${base}` : '';
};
const chipContainer = document.getElementById('activeFilterChips');

function buildActiveChips(filters){
  if (!chipContainer) return;
  chipContainer.innerHTML = '';
  const chips = [];
  const pushChip = (label, clear) => {
    if (!label || typeof clear !== 'function') return;
    chips.push({ label, clear });
  };

  const idRaw = val('f-id');
  if (filters.ids && filters.ids.size && idRaw) {
    pushChip(`ID: ${idRaw}`, () => {
      const el = document.getElementById('f-id');
      if (el) el.value = '';
    });
  }

  const categoryRaw = val('f-category');
  if (categoryRaw) {
    pushChip(`Categoría: ${categoryRaw}`, () => {
      const el = document.getElementById('f-category');
      if (el) el.value = '';
    });
  }

  if (Number.isFinite(filters.priceMin)) {
    pushChip(`Precio ≥ ${fmtCurrency(filters.priceMin)}`, () => {
      const el = document.getElementById('f-price-min');
      if (el) el.value = '';
    });
  }
  if (Number.isFinite(filters.priceMax)) {
    pushChip(`Precio ≤ ${fmtCurrency(filters.priceMax)}`, () => {
      const el = document.getElementById('f-price-max');
      if (el) el.value = '';
    });
  }

  if (Number.isFinite(filters.ratingMin)) {
    pushChip(`Rating ≥ ${fmtNumber(filters.ratingMin)}`, () => {
      const el = document.getElementById('f-rating-min');
      if (el) el.value = '';
    });
  }
  if (Number.isFinite(filters.ratingMax)) {
    pushChip(`Rating ≤ ${fmtNumber(filters.ratingMax)}`, () => {
      const el = document.getElementById('f-rating-max');
      if (el) el.value = '';
    });
  }

  if (Number.isFinite(filters.unitsMin)) {
    pushChip(`Unidades ≥ ${fmtNumber(filters.unitsMin)}`, () => {
      const el = document.getElementById('f-units-min');
      if (el) el.value = '';
    });
  }
  if (Number.isFinite(filters.unitsMax)) {
    pushChip(`Unidades ≤ ${fmtNumber(filters.unitsMax)}`, () => {
      const el = document.getElementById('f-units-max');
      if (el) el.value = '';
    });
  }

  if (Number.isFinite(filters.revenueMin)) {
    pushChip(`Ingresos ≥ ${fmtCurrency(filters.revenueMin)}`, () => {
      const el = document.getElementById('f-revenue-min');
      if (el) el.value = '';
    });
  }
  if (Number.isFinite(filters.revenueMax)) {
    pushChip(`Ingresos ≤ ${fmtCurrency(filters.revenueMax)}`, () => {
      const el = document.getElementById('f-revenue-max');
      if (el) el.value = '';
    });
  }

  if (Number.isFinite(filters.convMin)) {
    pushChip(`Conv. ≥ ${fmtNumber(filters.convMin)}%`, () => {
      const el = document.getElementById('f-conv-min');
      if (el) el.value = '';
    });
  }
  if (Number.isFinite(filters.convMax)) {
    pushChip(`Conv. ≤ ${fmtNumber(filters.convMax)}%`, () => {
      const el = document.getElementById('f-conv-max');
      if (el) el.value = '';
    });
  }

  if (filters.dateFrom) {
    pushChip(`Desde ${filters.dateFrom}`, () => {
      const el = document.getElementById('f-date-from');
      if (el) el.value = '';
    });
  }
  if (filters.dateTo) {
    pushChip(`Hasta ${filters.dateTo}`, () => {
      const el = document.getElementById('f-date-to');
      if (el) el.value = '';
    });
  }

  const rangeRaw = val('f-range-text');
  if (rangeRaw) {
    pushChip(`Rango: ${rangeRaw}`, () => {
      const el = document.getElementById('f-range-text');
      if (el) el.value = '';
    });
  }

  if (filters.desireMag) {
    pushChip(`Desire: ${filters.desireMag}`, () => {
      const el = document.getElementById('f-desire-mag');
      if (el) el.value = '';
    });
  }
  if (filters.awareness) {
    pushChip(`Awareness: ${awarenessLabel(filters.awareness)}`, () => {
      const el = document.getElementById('f-awareness');
      if (el) el.value = '';
    });
  }
  if (filters.competition) {
    pushChip(`Competencia: ${filters.competition}`, () => {
      const el = document.getElementById('f-competition');
      if (el) el.value = '';
    });
  }

  if (Number.isFinite(filters.scoreMin)) {
    pushChip(`Score ≥ ${Math.round(filters.scoreMin)}`, () => {
      const el = document.getElementById('f-score-min');
      if (el) el.value = '';
    });
  }
  if (Number.isFinite(filters.scoreMax)) {
    pushChip(`Score ≤ ${Math.round(filters.scoreMax)}`, () => {
      const el = document.getElementById('f-score-max');
      if (el) el.value = '';
    });
  }

  chips.forEach(({ label, clear }) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    const text = document.createElement('span');
    text.textContent = label;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = '×';
    btn.setAttribute('aria-label', 'Quitar filtro');
    btn.addEventListener('click', () => {
      clear();
      const updated = readFilters();
      getAppState().filters = updated;
      document.dispatchEvent(new CustomEvent('filters-changed', { detail: updated }));
    });
    chip.append(text, btn);
    chipContainer.appendChild(chip);
  });
}

const btnFilters = document.getElementById('btnFilters');
if (btnFilters) {
  btnFilters.addEventListener('click', (e) => {
    e.preventDefault();
    toggleDrawer();
  });
}
const closeBtn = document.getElementById('closeFilters');
if (closeBtn) {
  closeBtn.addEventListener('click', (e) => {
    e.preventDefault();
    closeDrawer();
  });
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeDrawer();
    return;
  }
  if (isTextInput(e.target)) return;
  if (e.key === '/' && !e.metaKey && !e.ctrlKey) {
    e.preventDefault();
    document.getElementById('searchInput')?.focus();
    return;
  }
  if (e.key && e.key.toLowerCase() === 'f' && !e.metaKey && !e.ctrlKey && !e.altKey) {
    e.preventDefault();
    toggleDrawer();
    return;
  }
  if (e.key && e.key.toLowerCase() === 'g' && !e.metaKey && !e.ctrlKey && !e.altKey) {
    e.preventDefault();
    document.getElementById('groupSelect')?.focus();
  }
});

document.addEventListener('click', (e) => {
  const target = e.target;
  if (!target) return;
  const id = target.id;
  const text = target.textContent?.trim();
  if (id === 'applyFilters' || id === 'apply-filters' || /^Aplicar$/i.test(text || '')) {
    const filters = readFilters();
    getAppState().filters = filters;
    document.dispatchEvent(new CustomEvent('filters-changed', { detail: filters }));
    closeDrawer();
  }
  if (id === 'clearFilters' || id === 'clear-filters' || /^Limpiar$/i.test(text || '')) {
    document.querySelectorAll('.filters-grid input').forEach(el => { el.value = ''; });
    document.querySelectorAll('.filters-grid select').forEach(el => { el.value = ''; });
    const filters = readFilters();
    getAppState().filters = filters;
    document.dispatchEvent(new CustomEvent('filters-changed', { detail: filters }));
  }
});

const initialFilters = getAppState().filters || readFilters();
getAppState().filters = initialFilters;
buildActiveChips(initialFilters);

document.addEventListener('filters-changed', (e) => {
  const filters = e?.detail || readFilters();
  buildActiveChips(filters);
});

export { buildActiveChips };
