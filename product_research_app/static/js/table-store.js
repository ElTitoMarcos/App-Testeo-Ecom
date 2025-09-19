(function(){
  const records = new Map();
  const order = [];
  let filters = null;
  let activeGroupId = -1;

  const isPlainObject = (value) => {
    return value && typeof value === 'object' && !Array.isArray(value);
  };

  const normalizeId = (rawId) => {
    if (rawId === null || rawId === undefined) return null;
    const num = Number(rawId);
    if (Number.isFinite(num)) return num;
    return String(rawId);
  };

  const mergeRecord = (existing, next) => {
    const base = existing ? { ...existing } : {};
    if (!next || typeof next !== 'object') return base;
    for (const [key, value] of Object.entries(next)) {
      if (value === undefined) continue;
      if (key === 'extras' && isPlainObject(value)) {
        const prev = isPlainObject(base.extras) ? base.extras : {};
        base.extras = { ...prev, ...value };
        continue;
      }
      if (isPlainObject(value) && isPlainObject(base[key])) {
        base[key] = { ...base[key], ...value };
      } else {
        base[key] = value;
      }
    }
    if (!('id' in base) && next.id !== undefined) {
      base.id = next.id;
    }
    if (!base.name && base.title) {
      base.name = base.title;
    }
    if (!base.title && base.name) {
      base.title = base.name;
    }
    return base;
  };

  const upsertMany = (rows) => {
    if (!Array.isArray(rows)) return;
    for (const row of rows) {
      if (!row) continue;
      const normalizedId = normalizeId(row.id);
      if (normalizedId === null || normalizedId === undefined || normalizedId === '') continue;
      const key = String(normalizedId);
      const existing = records.get(key);
      const merged = mergeRecord(existing, row);
      if (!('id' in merged)) {
        merged.id = normalizedId;
      }
      records.set(key, merged);
      if (!existing) {
        order.push(key);
      }
    }
  };

  const patchMany = (updates) => {
    upsertMany(updates);
  };

  const replaceAll = (rows) => {
    records.clear();
    order.length = 0;
    upsertMany(rows);
  };

  const setFilters = (next) => {
    if (!next) {
      filters = null;
    } else {
      filters = { ...next };
    }
  };

  const setGroup = (groupId) => {
    if (groupId === null || groupId === undefined || groupId === '' || Number(groupId) === -1) {
      activeGroupId = -1;
      return;
    }
    const num = Number(groupId);
    activeGroupId = Number.isNaN(num) ? groupId : num;
  };

  const matchesGroup = (row) => {
    if (activeGroupId === null || activeGroupId === undefined || activeGroupId === -1) return true;
    if (!row) return false;
    const target = activeGroupId;
    const groupsArray = row.groups || row.group_ids || row.groupIds;
    if (Array.isArray(groupsArray)) {
      return groupsArray.map((g) => Number(g)).includes(Number(target));
    }
    const gid = row.group_id ?? row.groupId ?? row.groupID;
    if (gid === null || gid === undefined || gid === '') return false;
    return Number(gid) === Number(target) || String(gid) === String(target);
  };

  const shouldRender = (row) => {
    if (!row) return false;
    if (!matchesGroup(row)) return false;
    if (!filters) return true;
    const apply = window.applyFilters;
    if (typeof apply !== 'function') return true;
    try {
      const result = apply([row], filters);
      return Array.isArray(result) ? result.length > 0 : !!result;
    } catch (err) {
      console.warn('[TableStore] filter evaluation error', err);
      return true;
    }
  };

  const entries = () => order.map((id) => records.get(id)).filter(Boolean);
  const get = (id) => records.get(String(id));

  const remove = (id) => {
    const key = String(id);
    if (!records.has(key)) return;
    records.delete(key);
    const idx = order.indexOf(key);
    if (idx >= 0) order.splice(idx, 1);
  };

  const clear = () => {
    records.clear();
    order.length = 0;
  };

  window.TableStore = {
    upsertMany,
    patchMany,
    replaceAll,
    setFilters,
    setGroup,
    entries,
    get,
    remove,
    clear,
    shouldRender,
  };
})();
