# UI Migration Notes

The interface now has two sticky rows: the global app bar and a unified `controlBar` that replaces the old page and table toolbars.

## Control mapping
- controlBar/left: `searchInput`, `searchBtn`, `btnFilters`, `activeFilterChips`, `listMeta`
- controlBar/right: `legendBtn`, `selCount`, `groupSelect`, `btnAddToGroup`, `sendPrompt`, `btnColumns`, `btnExport`, `btnDelete`
- table header (first column): `selectAll`

## Shortcuts
- `/` — focus search box
- `f` — toggle filters drawer
- `g` — focus group selector
