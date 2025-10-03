# Cambios aplicados

## Resumen del incidente
- **Salida sin JSON / finish_reason=length**: las respuestas largas del modelo truncaban los batches grandes. Se añadió microbatcheo recursivo que detecta `finish_reason="length"`, divide el lote y vuelve a intentar hasta llegar a un solo producto antes de declarar fallo, registrando el error como `ko.json_truncated`.
- **Errores `json_parse`**: ahora se vuelve a intentar con lotes más pequeños incluso tras un primer reintento para garantizar JSON válido, manteniendo las configuraciones de tokens.
- **Ruido `Bad request version`**: el `QuietHandlerMixin` filtra las conexiones TLS mal dirigidas comprobando el primer byte del handshake y evita que aparezcan como errores HTTP 400 en los logs.

## Configuración nueva/modificada
- `TOKENS_POR_ITEM_ESTIMADOS` (`PRAPP_AI_COLUMNS_TOKENS_PER_ITEM`, default 96): estimación del presupuesto de tokens por producto. Ajusta esta constante si el modelo devuelve respuestas más largas o cortas; se usa con un margen del 15 % (`TOKEN_BUDGET_MARGIN`) para calcular el tamaño máximo inicial del batch.

## Cómo ejecutar las pruebas
```bash
pip install -r requirements.txt
pytest product_research_app/tests/test_ai_columns_microbatch.py product_research_app/tests/test_http_quiet.py -q
pytest -q
```

## Runbook de verificación manual
1. Arrancar la aplicación como de costumbre (`python -m product_research_app.app` o script equivalente).
2. Lanzar un llenado de columnas de IA con varios productos (>40) y comprobar en los logs que:
   - No aparecen mensajes `Salida sin JSON (finish_reason=length …)`; si los hay, deben ir seguidos de reintentos con lotes más pequeños.
   - Las entradas de error para truncados se etiquetan como `ko.json_truncated` en la telemetría.
3. Ejecutar un escaneo de puertos/sondeo TLS contra el puerto HTTP y verificar que no se registran entradas `code 400, message Bad request version …`.
