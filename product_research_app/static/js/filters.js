let filtersState = {
  priceMin: null,
  priceMax: null,
  dateMin: '',
  dateMax: '',
  ratingMin: null,
  category: '',
  scoreMin: null,
};

const idMap = {
  priceMin: 'filterPriceMin',
  priceMax: 'filterPriceMax',
  dateMin: 'filterDateMin',
  dateMax: 'filterDateMax',
  ratingMin: 'filterRatingMin',
  category: 'filterCategory',
  scoreMin: 'filterScoreMin'
};

function toggleDrawer() {
  document.getElementById('filtersDrawer').classList.toggle('hidden');
}

function closeDrawer() {
  document.getElementById('filtersDrawer').classList.add('hidden');
}

function applyFiltersFromState() {
  const dMin = filtersState.dateMin ? parseDate(filtersState.dateMin) : null;
  const dMax = filtersState.dateMax ? parseDate(filtersState.dateMax) : null;
  products = allProducts.filter(item => {
    if (!isNaN(filtersState.priceMin)) {
      if (item.price === null || item.price === undefined || item.price < filtersState.priceMin) return false;
    }
    if (!isNaN(filtersState.priceMax)) {
      if (item.price === null || item.price === undefined || item.price > filtersState.priceMax) return false;
    }
    if (dMin || dMax) {
      let launch = '';
      if (item.extras && item.extras['Launch Date']) launch = item.extras['Launch Date'];
      const dLaunch = parseDate(launch);
      if (dMin && dLaunch && dLaunch < dMin) return false;
      if (dMax && dLaunch && dLaunch > dMax) return false;
    }
    if (!isNaN(filtersState.ratingMin)) {
      const ratingVal = item.extras && item.extras['Product Rating'] ? parseFloat(String(item.extras['Product Rating']).replace(/[^0-9.]+/g,'')) : null;
      if (ratingVal === null || ratingVal < filtersState.ratingMin) return false;
    }
    if (filtersState.category) {
      const cat = (item.category || '').toString().toLowerCase();
      if (!cat.includes(filtersState.category)) return false;
    }
    if (!isNaN(filtersState.scoreMin)) {
      const sc = item.score;
      if (sc === null || sc === undefined || sc < filtersState.scoreMin) return false;
    }
    return true;
  });
  buildActiveChips(filtersState);
  if (typeof startProgress === 'function') startProgress();
  renderTable();
}

function buildActiveChips(state) {
  const container = document.getElementById('activeFilterChips');
  if (!container) return;
  container.innerHTML = '';
  const chips = [];
  if (!isNaN(state.priceMin)) chips.push(['priceMin', `≥ ${state.priceMin}`]);
  if (!isNaN(state.priceMax)) chips.push(['priceMax', `≤ ${state.priceMax}`]);
  if (state.dateMin) chips.push(['dateMin', `Desde ${state.dateMin}`]);
  if (state.dateMax) chips.push(['dateMax', `Hasta ${state.dateMax}`]);
  if (!isNaN(state.ratingMin)) chips.push(['ratingMin', `Rating ≥ ${state.ratingMin}`]);
  if (state.category) chips.push(['category', `Cat: ${state.category}`]);
  if (!isNaN(state.scoreMin)) chips.push(['scoreMin', `Score ≥ ${state.scoreMin}`]);
  chips.forEach(([key, label]) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = label;
    const btn = document.createElement('button');
    btn.textContent = '×';
    btn.onclick = () => {
      if (['priceMin','priceMax','ratingMin','scoreMin'].includes(key)) {
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
document.getElementById('applyFilters')?.addEventListener('click', () => {
  filtersState.priceMin = parseFloat(document.getElementById('filterPriceMin').value);
  filtersState.priceMax = parseFloat(document.getElementById('filterPriceMax').value);
  filtersState.dateMin = document.getElementById('filterDateMin').value;
  filtersState.dateMax = document.getElementById('filterDateMax').value;
  filtersState.ratingMin = parseFloat(document.getElementById('filterRatingMin').value);
  filtersState.category = document.getElementById('filterCategory').value.trim().toLowerCase();
  filtersState.scoreMin = parseFloat(document.getElementById('filterScoreMin').value);
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
  document.getElementById('filterScoreMin').value = '';
  filtersState = { priceMin: null, priceMax: null, dateMin: '', dateMax: '', ratingMin: null, category: '', scoreMin: null };
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

