let filtersState = {
  priceMin: null,
  priceMax: null,
  dateMin: '',
  dateMax: '',
  ratingMin: null,
  category: '',
};

const idMap = {
  priceMin: 'filterPriceMin',
  priceMax: 'filterPriceMax',
  dateMin: 'filterDateMin',
  dateMax: 'filterDateMax',
  ratingMin: 'filterRatingMin',
  category: 'filterCategory'
};

function toggleDrawer() {
  document.getElementById('filtersDrawer').classList.toggle('hidden');
}

function closeDrawer() {
  document.getElementById('filtersDrawer').classList.add('hidden');
}

function applyCurrentFilters(list) {
  const dMin = filtersState.dateMin ? parseDate(filtersState.dateMin) : null;
  const dMax = filtersState.dateMax ? parseDate(filtersState.dateMax) : null;
  return list.filter(item => {
    if (filtersState.priceMin !== null && !isNaN(filtersState.priceMin)) {
      if (item.price === null || item.price === undefined || item.price < filtersState.priceMin) return false;
    }
    if (filtersState.priceMax !== null && !isNaN(filtersState.priceMax)) {
      if (item.price === null || item.price === undefined || item.price > filtersState.priceMax) return false;
    }
    if (dMin || dMax) {
      let launch = '';
      if (item.extras && item.extras['Launch Date']) launch = item.extras['Launch Date'];
      const dLaunch = parseDate(launch);
      if (dMin && dLaunch && dLaunch < dMin) return false;
      if (dMax && dLaunch && dLaunch > dMax) return false;
    }
    if (filtersState.ratingMin !== null && !isNaN(filtersState.ratingMin)) {
      const ratingVal = item.extras && item.extras['Product Rating'] ? parseFloat(String(item.extras['Product Rating']).replace(/[^0-9.]+/g,'')) : null;
      if (ratingVal === null || ratingVal < filtersState.ratingMin) return false;
    }
    if (filtersState.category) {
      const cat = (item.category || '').toString().toLowerCase();
      if (!cat.includes(filtersState.category)) return false;
    }
    return true;
  });
}

window.applyCurrentFilters = list => applyCurrentFilters(list);

function applyFiltersFromState() {
  const filtered = applyCurrentFilters(allProducts);
  // Mutate the global products array in place so renderTable sees the filtered list
  products.splice(0, products.length, ...filtered);
  window.products = products;
  buildActiveChips(filtersState);
  if (typeof startProgress === 'function') startProgress();
  selection.clear();
  updateMasterState();
  renderTable();
}

function buildActiveChips(state) {
  const container = document.getElementById('activeFilterChips');
  if (!container) return;
  container.innerHTML = '';
  const chips = [];
  if (state.priceMin !== null && !isNaN(state.priceMin)) chips.push(['priceMin', `≥ ${state.priceMin}`]);
  if (state.priceMax !== null && !isNaN(state.priceMax)) chips.push(['priceMax', `≤ ${state.priceMax}`]);
  if (state.dateMin) chips.push(['dateMin', `Desde ${state.dateMin}`]);
  if (state.dateMax) chips.push(['dateMax', `Hasta ${state.dateMax}`]);
  if (state.ratingMin !== null && !isNaN(state.ratingMin)) chips.push(['ratingMin', `Rating ≥ ${state.ratingMin}`]);
  if (state.category) chips.push(['category', `Cat: ${state.category}`]);
  chips.forEach(([key, label]) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = label;
    const btn = document.createElement('button');
    btn.textContent = '×';
    btn.onclick = () => {
      if (['priceMin','priceMax','ratingMin'].includes(key)) {
        filtersState[key] = null;
      } else {
        filtersState[key] = '';
      }
      document.getElementById(idMap[key]).value = '';
      applyFiltersFromState();
    };
    chip.appendChild(btn);
    container.appendChild(chip);
  });
}

document.getElementById('btnFilters')?.addEventListener('click', toggleDrawer);
document.getElementById('closeFilters')?.addEventListener('click', closeDrawer);
document.getElementById('applyFilters')?.addEventListener('click', () => {
  const pMinVal = document.getElementById('filterPriceMin').value;
  const pMaxVal = document.getElementById('filterPriceMax').value;
  const rMinVal = document.getElementById('filterRatingMin').value;
  filtersState.priceMin = pMinVal ? parseFloat(pMinVal) : null;
  filtersState.priceMax = pMaxVal ? parseFloat(pMaxVal) : null;
  filtersState.dateMin = document.getElementById('filterDateMin').value;
  filtersState.dateMax = document.getElementById('filterDateMax').value;
  filtersState.ratingMin = rMinVal ? parseFloat(rMinVal) : null;
  filtersState.category = document.getElementById('filterCategory').value.trim().toLowerCase();
  applyFiltersFromState();
  closeDrawer();
});

document.getElementById('clearFilters')?.addEventListener('click', () => {
  document.getElementById('filterPriceMin').value = '';
  document.getElementById('filterPriceMax').value = '';
  document.getElementById('filterDateMin').value = '';
  document.getElementById('filterDateMax').value = '';
  document.getElementById('filterRatingMin').value = '';
  document.getElementById('filterCategory').value = '';
  filtersState = { priceMin: null, priceMax: null, dateMin: '', dateMax: '', ratingMin: null, category: '' };
  applyFiltersFromState();
});

document.addEventListener('keydown', (e) => {
  if (e.key === '/') {
    e.preventDefault();
    document.getElementById('searchInput')?.focus();
  }
  if (e.key.toLowerCase() === 'f') {
    e.preventDefault();
    toggleDrawer();
  }
  if (e.key.toLowerCase() === 'g') {
    e.preventDefault();
    document.getElementById('groupSelect')?.focus();
  }
  if (e.key === 'Escape') {
    closeDrawer();
  }
});

function updateHeaderHeight() {
  const topBar = document.getElementById('topBar');
  if (topBar) {
    document.documentElement.style.setProperty('--header-h', `${topBar.offsetHeight}px`);
  }
}
window.addEventListener('load', updateHeaderHeight);
window.addEventListener('resize', updateHeaderHeight);
