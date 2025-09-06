# UI Migration Notes

The interface now features two persistent toolbars:

- **pageBar** — page-level controls
- **tableToolbar** — table-level actions

## Control mapping
- pageBar/left: `searchInput`, `searchBtn`, `btnFilters`, `activeFilterChips`, `listMeta`
- pageBar/right: `newListName`, `createListBtn`, `groupSelect`, `sendPrompt`
- tableToolbar/left: `selectAll`
- tableToolbar/right: `btnColumns`, `btnAddToGroup`, `btnExport`, `btnDelete`

## Shortcuts
- `/` — focus search box
- `f` — toggle filters drawer
- `g` — focus group selector
