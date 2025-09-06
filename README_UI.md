# UI Migration Notes

The interface now has two sticky rows: the global app bar, a flexible `.app-topbar` with the search box and main actions, and a secondary `.table-toolbar` for filters and selection info.

## Control mapping
- app-topbar/left: `searchInput`, `searchBtn`
- app-topbar/right: `groupSelect`, `btnAddToGroup`, `sendPrompt`, `btnColumns`, `btnExport`, `btnDelete`, `newListName`, `createListBtn`
- table-toolbar/left: `btnFilters`, `activeFilterChips`, `listMeta`
- table-toolbar/right: `selCount`, `legendBtn`
- table header (first column): `selectAll`

## Shortcuts
- `/` — focus search box
- `f` — toggle filters drawer
- `g` — focus group selector
