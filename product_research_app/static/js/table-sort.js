const table = document.getElementById('trendsTable');
const thead = table ? table.querySelector('thead') : null;
const tbody = table ? table.querySelector('tbody') : null;

function toNumber(value) {
  if (value == null) return NaN;
  let str = String(value).trim();
  if (!str) return NaN;
  str = str.replace(/[€$]/g, '').replace(/\s+/g, '');
  let factor = 1;
  const suffix = str.match(/(k|m)$/i);
  if (suffix) {
    const letter = suffix[1].toLowerCase();
    if (letter === 'k') factor = 1e3;
    if (letter === 'm') factor = 1e6;
    str = str.slice(0, -1);
  }
  str = str.replace(/\./g, '').replace(/,/g, '.');
  const num = parseFloat(str);
  if (!Number.isFinite(num)) return NaN;
  return num * factor;
}

function getCellValue(row, index) {
  const cell = row.children[index];
  return cell ? cell.textContent.trim() : '';
}

function sortRows(th) {
  if (!thead || !tbody) return;
  const index = Array.from(th.parentNode.children).indexOf(th);
  if (index < 0) return;
  const type = th.dataset.type || 'text';
  const currentSort = th.getAttribute('aria-sort');
  const nextSort = currentSort === 'ascending' ? 'descending' : 'ascending';

  thead.querySelectorAll('th').forEach(header => header.removeAttribute('aria-sort'));
  th.setAttribute('aria-sort', nextSort);

  const rows = Array.from(tbody.querySelectorAll('tr'));
  const dirMultiplier = nextSort === 'ascending' ? 1 : -1;

  rows.sort((rowA, rowB) => {
    const valueA = getCellValue(rowA, index);
    const valueB = getCellValue(rowB, index);

    if (type === 'num') {
      const numA = toNumber(valueA);
      const numB = toNumber(valueB);
      const nanA = Number.isNaN(numA);
      const nanB = Number.isNaN(numB);
      if (nanA && nanB) return 0;
      if (nanA) return 1;
      if (nanB) return -1;
      if (numA === numB) return 0;
      return (numA - numB) * dirMultiplier;
    }

    const result = valueA.localeCompare(valueB, 'es', { numeric: true, sensitivity: 'base' });
    return dirMultiplier * result;
  });

  const fragment = document.createDocumentFragment();
  rows.forEach(row => fragment.appendChild(row));
  tbody.appendChild(fragment);
}

if (thead && tbody) {
  thead.addEventListener('click', (event) => {
    const th = event.target.closest('th[data-key]');
    if (!th || !thead.contains(th)) return;
    sortRows(th);
  });
}

export {};
