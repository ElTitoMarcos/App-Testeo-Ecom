# Scripts y utilidades

## Importador unificado

El endpoint `/upload` ahora acepta archivos `.csv` y `.xlsx` con el mismo flujo rápido:

- Los Excel se leen con `pandas` y se convierten a registros en memoria.
- Los datos se insertan en bloque mediante `importer_fast.fast_import_records`.
- El progreso reporta etapas como `parse_xlsx` y `db_bulk_insert` para que la barra avance.

No se lanzan tareas de IA ni de scoring durante la importación masiva; solo se cargan los registros. Usa el estatus (`task_id`) que devuelve `/upload` para seguir el avance desde el frontend.
