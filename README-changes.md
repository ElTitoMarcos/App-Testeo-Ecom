# Cambios aplicados

## Resumen del incidente
- **`finish_reason=length` / `reason=json_truncated` en batches grandes**: el modelo cortaba las respuestas JSON cuando el presupuesto de tokens era insuficiente. Ahora `call_prompt_task_async` amplía automáticamente el `max_completion_tokens`, añade instrucciones de respuesta mínima y reintenta antes de fallar. En `ai_columns` se divide el lote según el presupuesto estimado de tokens, se escalona hasta 1 ítem y, si aun así se trunca, se registra un `ko.json_truncated` sin detener el resto del job.
- **`InvalidJSONError: La respuesta JSON está vacía` al refinar Desire**: el parser sanitiza fences, detecta vacíos y fuerza un reintento con una orden estricta de “solo JSON válido”. Si el segundo intento no aporta datos, se devuelve un JSON mínimo con `low_confidence` para que el job continúe.
- **Ruido de handshakes TLS en el servidor HTTP**: el `QuietHandlerMixin` identifica paquetes que comienzan por `0x16 0x03 0x01` (ClientHello) y suprime el log de error, dejándolo como `DEBUG` opcional.

## Configuración nueva/modificada
- `product_research_app/gpt.py`
  - `TOKENS_POR_ITEM_ESTIMADOS` (`PRAPP_GPT_TOKENS_POR_ITEM_ESTIMADOS`, fallback a `PRAPP_AI_COLUMNS_TOKENS_PER_ITEM`, default 220): estimación de salida por ítem usada para calcular el tamaño máximo seguro de lote.
  - `MIN_COMPLETION_TOKENS_JSON` (`PRAPP_GPT_MIN_COMPLETION_TOKENS_JSON`, default 700) y `MAX_COMPLETION_TOKENS_JSON` (`PRAPP_GPT_MAX_COMPLETION_TOKENS_JSON`, default 1200): límites inferiores/superiores del `max_completion_tokens` cuando se exige JSON estricto. Ajusta estos valores si el modelo necesita más o menos presupuesto de salida.
- `product_research_app/services/ai_columns.py`
  - Usa los límites anteriores para decidir cuántos ítems caben en un batch (`⌊MAX * 0.9 / TOKENS_POR_ITEM_ESTIMADOS⌋`). Si personalizas las constantes en `gpt.py`, esta lógica se adapta automáticamente.

## Cómo ejecutar las pruebas
```bash
pip install -r requirements.txt
pytest product_research_app/tests/test_ai_columns_microbatch.py \
       product_research_app/tests/test_ai_columns_refine.py \
       product_research_app/tests/test_http_quiet.py \
       product_research_app/tests/test_gpt_messages.py -q
pytest -q
```

## Runbook de verificación manual
1. Arranca la app y lanza un “AI fill” con suficientes productos para generar batches grandes.
2. Comprueba en los logs que, ante cualquier `finish_reason=length`/`json_truncated`, se observan reintentos con batches más pequeños y sin degradar el resto del trabajo.
3. Fuerza un refine manual (p. ej. con un borrador vacío) y confirma que, tras dos respuestas vacías, se guarda un JSON mínimo con `low_confidence`.
4. Envía un ClientHello TLS al puerto HTTP (por ejemplo con `openssl s_client -connect localhost:PUERTO`) y verifica que no aparecen entradas `code 400` en los logs del servidor (solo un mensaje `DEBUG` opcional).
