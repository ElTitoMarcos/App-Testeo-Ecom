import { fetchJson } from './net.js';

const deleteInFlight = {};

export async function listGroups() {
  const lists = await fetchJson('/lists');
  window.listCache = lists;
  return lists;
}

export async function createGroup(name) {
  const res = await fetchJson('/create_list', {
    method: 'POST',
    body: JSON.stringify({ name })
  });
  await listGroups();
  document.dispatchEvent(new CustomEvent('groups-updated'));
  return res;
}

export async function deleteGroup(id, opts = {}) {
  if (deleteInFlight[id]) return;
  deleteInFlight[id] = true;
  const { mode = 'remove', targetGroupId = null } = opts;
  try {
    await fetchJson('/delete_list', {
      method: 'POST',
      body: JSON.stringify({ id, mode, targetGroupId })
    });
    window.listCache = (window.listCache || []).filter(g => g.id !== id);
    if (window.groupsMap) delete window.groupsMap[id];
    if (window.groupsList) window.groupsList = (window.groupsList || []).filter(g => g.id !== id);
    try {
      const cache = JSON.parse(localStorage.getItem('groupsCache') || '[]');
      localStorage.setItem('groupsCache', JSON.stringify(cache.filter(g => g.id !== id)));
    } catch (e) {
      /* ignore */
    }
    if (window.currentGroupFilter === id) {
      const next = mode === 'move' && targetGroupId ? targetGroupId : -1;
      window.currentGroupFilter = next;
      if (next === -1) {
        try {
          localStorage.removeItem('prapp.currentGroupId');
          localStorage.removeItem('prapp.currentGroupName');
        } catch (e) {}
      }
      if (window.applyGroupFilter) {
        await window.applyGroupFilter(next, { skipProgress: true });
      } else if (next === -1 && window.fetchProducts) {
        await window.fetchProducts({ skipProgress: true, groupId: null });
      }
    }
    document.dispatchEvent(new CustomEvent('groups-updated'));
  } finally {
    delete deleteInFlight[id];
  }
}

export default { listGroups, createGroup, deleteGroup };

// expose for non-module consumers
window.groupsService = { listGroups, createGroup, deleteGroup };
