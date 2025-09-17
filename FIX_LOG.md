# FIX LOG

## Fase 1 — Hallazgos
- **Causa raíz /_ai_status 404:** La ruta existe en `product_research_app/web_app.py`, pero solo se inicializa `AI_STATUS` cuando se invoca `safe_run_post_import_auto`. El flujo de importación XLSX (`_process_import_job`) nunca llama a ese runner y, tras finalizar, no hay estado asociado al `task_id`, por lo que el `GET /_ai_status` devuelve 404.
- **Columnas IA vacías tras importación:** En la importación XLSX se invoca el legado `ai_columns.fill_ai_columns` (sin seguimiento de estado) en lugar de `run_post_import_auto`. No se ejecuta el runner que debería rellenar `ai_desire`, `ai_desire_label`, `desire_magnitude`, etc., y la UI (`_ensure_desire` en `product_research_app/web_app.py` y `preprocessProducts` en `product_research_app/static/index.html`) termina registrando `desire_missing=true`.
- **Runner existente:** `product_research_app/ai/runner.py` ya implementa lotes para deseo/imputación/winner score y actualiza las columnas correctas vía `UPDATE products ...`, pero depende de que se le pase la lista de IDs recién importados y de que se inicialice el estado. También carece de helpers `set_error` / trazas extendidas solicitadas.
- **Esquema:** `PRAGMA table_info(products)` confirma la presencia de las columnas objetivo (`ai_desire`, `ai_desire_label`, `desire_magnitude`, `review_count`, `image_count`, `winner_score`).

## Archivos a tocar en Fase 2
- `product_research_app/web_app.py`
- `product_research_app/ai/ai_status.py`
- `product_research_app/ai/runner.py`
- `product_research_app/ai/gpt_orchestrator.py`
- `product_research_app/config.py` (si hace falta exponer defaults AI_*)
- `product_research_app/services/importer_fast.py` / `_process_import_job` para disparo runner
- `product_research_app/static/index.html` (solo si hay que ajustar lectura legacy)
- Nuevo `scripts/smoke_ai_auto.py`

## Plan Fase 2
1. Añadir helpers de estado (`set_error`, notas) y exponer `AI_STATUS` conforme a requisitos, además de inicializar estado al lanzar runner en background y responder `/_ai_status` con alias si hiciera falta.
2. Modificar el flujo POST `/upload` para que **todas** las importaciones, incluido XLSX, lancen `run_post_import_auto` en segundo plano (thread/Executor) tras el bulk insert, usando `init_status`/`update_status` y manejando errores.
3. Ajustar `run_post_import_auto` para respetar configuraciones de lotes/llamadas, actualizar columnas con fallback heurístico y propagar errores vía `set_error`, dejando trazas claras. Sincronizar `desire_summary` si existiese.
4. Asegurar compatibilidad del esquema (ALTER TABLE condicional + copia desde `desire_summary` si aplica).
5. Implementar orquestador JSON-only (`run_desire_batch`, `run_imputacion_batch`, `run_weights_once`) con manejo de reintentos, límites y conteo de llamadas.
6. Añadir logs descriptivos de progreso por lote y notas específicas cuando falte deseo tras el runner.
7. Crear `scripts/smoke_ai_auto.py` para validar subida, polling de `_import_status`/`/_ai_status` y verificar columnas rellenadas.

## Fase 2 — Cambios aplicados
- Estado IA en memoria ampliado (`ai_status.set_error`, `poll_interval`, timestamps) y handler `/_ai_status` con alias `/ai_status` devolviendo `state="IDLE"` en 404.
- `safe_run_post_import_auto` inicializa estado, captura excepciones con traza, actualiza tablas de import (`import_jobs`) y lanza runner solo si no estamos bajo Pytest; en escenarios de test marca el job como completado sin disparar hilos.
- `_process_import_job` elimina el flujo legado de `ai_columns`, calcula Winner Score inicial, lanza el runner en hilo (o se salta en pruebas) y mantiene compatibilidad con `start_import_job_ai`.
- Runner IA ahora usa conexión dedicada, respeta límites configurables, rellena `ai_desire/label/desire_magnitude`, sincroniza `desire_summary`, marca `ai_columns_completed_at`, actualiza `review_count/image_count`, ejecuta Winner Score y registra avances/errores vía `set_error`.
- Migración defensiva en `database.initialize_database` para copiar `desire_summary` a `ai_desire` en bases antiguas.
- Logs de `_ensure_desire` incluyen razones y timestamp de actualización.
- Nuevo script `scripts/smoke_ai_auto.py` para smoke test end-to-end.

## Validación
- `pytest` (tras `pip install -r requirements.txt`).
- Smoke manual: `python scripts/smoke_ai_auto.py --base-url http://127.0.0.1:8000` (requiere servidor activo y API key válida).

