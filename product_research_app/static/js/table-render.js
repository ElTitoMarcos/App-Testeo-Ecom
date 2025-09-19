(function(){
  const insertQueue = [];
  const patchQueue = [];
  let rafId = null;
  const pendingHighlights = new Set();

  const schedule = () => {
    if (rafId !== null) return;
    rafId = requestAnimationFrame(flush);
  };

  const TableStore = window.TableStore || null;
  const ensureStore = () => TableStore;

  const registerVisible = (id) => {
    const idStr = String(id);
    if (!Array.isArray(window.currentPageIds)) return;
    if (!window.currentPageIds.includes(idStr)) {
      window.currentPageIds.push(idStr);
    }
  };

  const unregisterVisible = (id) => {
    const idStr = String(id);
    if (!Array.isArray(window.currentPageIds)) return;
    const idx = window.currentPageIds.indexOf(idStr);
    if (idx >= 0) {
      window.currentPageIds.splice(idx, 1);
    }
  };

  const syncCollections = (row) => {
    if (!row) return;
    if (!Array.isArray(window.allProducts)) {
      window.allProducts = [];
    }
    if (!Array.isArray(window.products)) {
      window.products = [];
    }
    const idStr = String(row.id);
    const idxAll = window.allProducts.findIndex((p) => String(p.id) === idStr);
    if (idxAll >= 0) {
      window.allProducts[idxAll] = row;
    } else {
      window.allProducts.push(row);
    }
    if (!ensureStore()) return;
    const shouldShow = ensureStore().shouldRender(row);
    const idxVisible = window.products.findIndex((p) => String(p.id) === idStr);
    if (shouldShow) {
      if (idxVisible >= 0) {
        window.products[idxVisible] = row;
      } else {
        window.products.push(row);
      }
    } else if (idxVisible >= 0) {
      window.products.splice(idxVisible, 1);
    }
  };

  const appendRows = (rows) => {
    if (!rows.length || !window.tbodyElement) return;
    const frag = document.createDocumentFragment();
    for (const row of rows) {
      const tr = window.ensureRowElement ? window.ensureRowElement(row.id) : null;
      if (!tr) continue;
      if (typeof window.renderRow === 'function') {
        window.renderRow(tr, row);
      }
      tr.classList.add('row-new');
      pendingHighlights.add(tr);
      frag.appendChild(tr);
      registerVisible(row.id);
    }
    if (frag.childNodes.length) {
      window.tbodyElement.appendChild(frag);
    }
  };

  const removeRow = (id) => {
    const el = document.getElementById(`row-${id}`);
    if (el && el.parentNode) {
      el.parentNode.removeChild(el);
    }
    unregisterVisible(id);
  };

  const updateRowIA = (row) => {
    if (!row) return;
    const el = window.ensureRowElement ? window.ensureRowElement(row.id) : null;
    if (!el) return;
    if (!el.isConnected && window.tbodyElement) {
      if (typeof window.renderRow === 'function') {
        window.renderRow(el, row);
      }
      el.classList.add('row-new');
      pendingHighlights.add(el);
      window.tbodyElement.appendChild(el);
      registerVisible(row.id);
      return;
    }
    if (typeof window.renderIAColumns === 'function') {
      window.renderIAColumns(el, row);
    }
    el.classList.add('row-updated');
    pendingHighlights.add(el);
  };

  const flush = () => {
    rafId = null;
    if (!ensureStore() || !window.tbodyElement) {
      insertQueue.length = 0;
      patchQueue.length = 0;
      pendingHighlights.clear();
      return;
    }
    if (insertQueue.length) {
      const payload = insertQueue.splice(0, insertQueue.length);
      ensureStore().upsertMany(payload);
      const renderable = [];
      for (const entry of payload) {
        const full = ensureStore().get(entry.id);
        if (!full) continue;
        syncCollections(full);
        if (ensureStore().shouldRender(full)) {
          renderable.push(full);
        }
      }
      appendRows(renderable);
    }
    if (patchQueue.length) {
      const payload = patchQueue.splice(0, patchQueue.length);
      ensureStore().patchMany(payload);
      for (const entry of payload) {
        const full = ensureStore().get(entry.id);
        if (!full) continue;
        syncCollections(full);
        if (!ensureStore().shouldRender(full)) {
          removeRow(entry.id);
          continue;
        }
        updateRowIA(full);
      }
    }
    if (typeof window.refreshColumns === 'function') {
      window.refreshColumns();
    }
    if (typeof window.applyColumnVisibility === 'function') {
      window.applyColumnVisibility();
    }
    if (typeof window.updateResultsBadge === 'function') {
      window.updateResultsBadge(window.products ? window.products.length : undefined);
    }
    if (typeof window.updateMasterState === 'function') {
      window.updateMasterState();
    }
    if (pendingHighlights.size) {
      setTimeout(() => {
        pendingHighlights.forEach((el) => {
          el.classList.remove('row-new', 'row-updated');
        });
        pendingHighlights.clear();
      }, 1500);
    }
  };

  window.LiveTable = {
    onImportBatch(rows) {
      if (!Array.isArray(rows) || rows.length === 0) return;
      insertQueue.push(...rows);
      schedule();
    },
    onEnrichBatch(updates) {
      if (!Array.isArray(updates) || updates.length === 0) return;
      patchQueue.push(...updates);
      schedule();
    },
  };
})();
