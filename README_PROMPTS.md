# Prompt Maestro v3

La capa HTTP expone `POST /api/gpt/<task>` para orquestar las tareas A, B, C, D, E y E_auto.
Cada tarea reutiliza el **Prompt Maestro v3** definido en `product_research_app/prompts/registry.py` y construye el mensaje
SYSTEM+USER con `product_research_app.gpt.build_messages`.

## Tareas disponibles
- **A** – Radiografía del mercado: síntesis de oportunidades y riesgos a partir de `context_json`.
- **B** – Ajuste de ponderaciones: genera pesos 0-100 y orden recomendado desde `aggregates` (respuesta JSON-only).
- **C** – Ángulos creativos: tabla con ángulos de venta usando `context_json`.
- **D** – Plan de validación: experimentos priorizados usando `context_json`.
- **E** – Resumen ejecutivo: recomendación ejecutiva a partir de `context_json`.
- **E_auto** – Decisión automática: clasifica lotes de productos usando `data` (respuesta JSON-only).

## Ejemplos de payload

### Task A
```http
POST /api/gpt/A
Content-Type: application/json

{
  "context_json": {
    "products": [
      {"id": "sku-101", "name": "Botella térmica", "trend": "alza", "margin": 0.42},
      {"id": "sku-102", "name": "Lámpara minimalista", "trend": "estable", "margin": 0.31},
      {"id": "sku-103", "name": "Silla ergonómica", "trend": "baja", "margin": 0.55}
    ],
    "period": "2024-Q3"
  }
}
```

### Task B
```http
POST /api/gpt/B
Content-Type: application/json

{
  "aggregates": {
    "momentum_avg": 0.67,
    "saturation_index": 0.48,
    "margin_p90": 0.62,
    "validation_rate": 0.54
  }
}
```

### Task E_auto
```http
POST /api/gpt/E_auto
Content-Type: application/json

{
  "data": {
    "items": [
      {
        "id": "sku-101",
        "name": "Botella térmica",
        "signals": {"trend": 0.82, "reviews": 0.74, "margin": 0.41}
      },
      {
        "id": "sku-102",
        "name": "Lámpara minimalista",
        "signals": {"trend": 0.55, "reviews": 0.68, "margin": 0.28}
      }
    ]
  }
}
```

## Respuestas y validaciones
- Las tareas **B** y **E_auto** son JSON-only. El backend solicita `response_format=json_schema` y valida contra los esquemas
  del registry. Si el modelo no soporta `response_format`, se aplica un fallback que extrae el primer bloque JSON y valida.
- Si la respuesta JSON contiene texto adicional o no se ajusta al esquema, el servidor devuelve **422**.
- Errores del proveedor (credenciales, red) devuelven **502** con `error="openai_error"`.
- El payload de éxito siempre incluye `{ "ok": true, "task": <task>, "content": ..., "raw": <respuesta_bruta> }`.

## Notas operativas
- El handler acepta opcionalmente `context_json`, `aggregates` y `data`; cualquier otra clave se ignora.
- El API key y el modelo se toman de `config.json` (o `OPENAI_API_KEY`).
- Los prompts añaden `prompt_version: prompt-maestro-v3` para facilitar auditorías y compatibilidad futura.
