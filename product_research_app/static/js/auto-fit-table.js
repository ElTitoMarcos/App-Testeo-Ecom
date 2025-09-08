/* auto-fit table columns + larger image — cambios mínimos */
(function () {
  const TABLE_ID = 'productTable';

  function headerKey(text) {
    const t = (text || '').toLowerCase().trim();
    if (t.startsWith('id')) return 'id';
    if (t.includes('imagen') || t.includes('image')) return 'img';
    if (t.includes('nombre') || t.includes('name') || t.includes('title')) return 'name';
    if (t.includes('categor')) return 'category';
    if (t.includes('precio') || t.includes('price')) return 'price';
    if (t.includes('rating') || t.includes('valoración')) return 'rating';
    if (t.includes('unidades') || t.includes('sold') || t.includes('ventas')) return 'sales';
    if (t.includes('ingres') || t.includes('revenue')) return 'revenue';
    return 'other';
  }

  /* Porcentajes objetivo con imagen más ancha.
     Suma de fijas ≈ 90%; el 10% restante se reparte entre "other". */
  const TARGET = {
    id: 4, img: 9.5, name: 26.5, category: 17, price: 8, rating: 6, sales: 9.5, revenue: 9.5
  };

  function buildColgroup(table) {
    const thead = table.querySelector('thead');
    if (!thead) return;
    const headers = [...thead.querySelectorAll('th')];
    const keys = headers.map(h => headerKey(h.textContent));

    const fixedSum = keys.reduce((sum, k) => sum + (TARGET[k] || 0), 0);
    const othersCount = keys.filter(k => k === 'other').length;
    const leftover = Math.max(0, 100 - fixedSum);
    const otherPct = othersCount ? leftover / othersCount : 0;

    let cg = table.querySelector('colgroup.__autoFit');
    if (cg) cg.remove();
    cg = document.createElement('colgroup');
    cg.className = '__autoFit';

    keys.forEach(k => {
      const col = document.createElement('col');
      const pct = (TARGET[k] || otherPct);
      col.style.width = pct.toFixed(3) + '%';
      if (k === 'img') {
        /* pista de mínimo; el td/imagen garantizan el tamaño visual */
        col.style.minWidth = 'var(--img-size)';
      }
      cg.appendChild(col);
    });

    table.insertBefore(cg, table.firstElementChild);
  }

  function decorateCells(table) {
    const thead = table.querySelector('thead');
    if (!thead) return;
    const headers = [...thead.querySelectorAll('th')];
    const keys = headers.map(h => headerKey(h.textContent));

    [...table.tBodies].forEach(tb => {
      [...tb.rows].forEach(row => {
        [...row.cells].forEach((td, i) => {
          const k = keys[i] || 'other';
          td.classList.add(
            'cell',
            k === 'img' ? 'cell-img' :
            k === 'name' ? 'cell-name' :
            k === 'category' ? 'cell-category' :
            k === 'price' ? 'cell-price' :
            k === 'rating' ? 'cell-rating' :
            k === 'sales' ? 'cell-sales' :
            k === 'revenue' ? 'cell-revenue' :
            k === 'id' ? 'cell-id' : 'cell-other'
          );

          /* Tooltip con el texto completo si la celda puede truncar */
          if (!td.title) {
            const full = td.textContent && td.textContent.trim();
            if (full) td.title = full;
          }
        });
      });
    });
  }

  function fitProductsTable() {
    const table = document.getElementById(TABLE_ID);
    if (!table) return;
    buildColgroup(table);
    decorateCells(table);
  }

  // Ejecuta al cargar y en resize (con debounce para no recalcular de más)
  let to = null;
  function onResize() {
    clearTimeout(to);
    to = setTimeout(fitProductsTable, 120);
  }

  document.addEventListener('DOMContentLoaded', fitProductsTable);
  window.addEventListener('resize', onResize);
})();

