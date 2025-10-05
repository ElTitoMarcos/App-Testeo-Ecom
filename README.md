# AI runtime tuning

The AI enrichment flow now uses configurable micro-batching and parallel HTTP calls. Runtime parameters can be adjusted through environment variables (or `config.json`) without touching the code.

## Environment flags

| Variable | Default | Description |
| --- | --- | --- |
| `AI_MODEL` | `gpt-5-mini` | Override the chat completion model used for column filling. |
| `PRAPP_AI_TRIAGE_MODEL` | `gpt-5-mini` | Model used by the quick triage phase; set it to `gpt-5-mini` to keep reasoning quality consistent end-to-end. |
| `AI_MICROBATCH` | `12` | Number of products per request. Requests are further reduced automatically if the token estimate would exceed `AI_TPM`. |
| `AI_PARALLELISM` | `8` | Maximum concurrent requests sent to the model. |
| `AI_TRUNC_TITLE` | `180` | Character cap applied to product titles before building the prompt. |
| `AI_TRUNC_DESC` | `800` | Character cap for descriptions and bullet lists in the prompt. |
| `AI_TIMEOUT` | `45` | Total HTTP timeout (seconds) per OpenAI request. |
| `AI_RPM` | model-specific | Soft limit for requests per minute enforced via an async semaphore. Defaults to the active model's published rate limit (e.g. 500 RPM for `gpt-5-mini`, 150 RPM for `gpt-4o`). |
| `AI_TPM` | model-specific | Soft limit for prompt tokens per minute; when unset it tracks the active model's tokens-per-minute limit (500k TPM for `gpt-5-mini`, 30k for `gpt-4o`). |

All variables also exist under `config["ai"]` so they can be persisted in `product_research_app/config.json` when running locally.

## Calibration cache

Calibrations for desire/competition percentiles are cached once per weights version in `product_research_app/ai_calibration_cache.json`. When insufficient data is available the fallback percentiles are persisted so subsequent jobs skip the heavy recalculation step.

## Instrumentation

Each micro-batch emits a log entry summarising prompt size and latency. A final summary aggregates per-request latency percentiles so a 100 item job should show ~8-9 parallel calls:

```
ai_columns.request: req_id=001 items=12 prompt_tokens_est=3100 start=2024-06-01T15:24:01Z end=2024-06-01T15:24:10Z duration=8.74s status=ok retries=0
ai_columns.request: req_id=002 items=12 prompt_tokens_est=2950 start=2024-06-01T15:24:01Z end=2024-06-01T15:24:11Z duration=9.18s status=ok retries=0
...
run_ai_fill_job: job=42 total=100 ok=92 cached=8 ko=0 cost=0.2150 pending=0 error=None duration=12.03s latency_p50=8.90s latency_p95=10.21s requests=9
```

Logs tagged `ai_columns.request` are useful to verify concurrency in smoke tests: you should see eight or nine overlapping requests for 100 products with the defaults.

## GPT-5 Mini runtime profile

`gpt-5-mini` is now the default model for enrichment flows. The runtime automatically applies the published capacity limits for each supported model. For `gpt-5-mini` this means:

* 500k prompt tokens per minute and 5M tokens per day.
* Up to ~400k context tokens (the job runner keeps a 10% safety margin, capping batches at 360k prompt+completion tokens unless you override `PRAPP_OPENAI_CONTEXT_SAFE_TOKENS`).
* Extra sampling parameters such as `temperature`, `top_p`, or penalties are ignored to comply with the reasoning API. Completion budgets use `max_completion_tokens` instead of `max_tokens` automatically.

Because of the larger context window the batcher increases its defaults when `gpt-5-mini` is active: micro-batches grow to 64 items, the per-item completion budget doubles to 256 tokens, and the per-request completion cap rises to 8,000 tokens. Models with smaller context windows keep the previous limits.

### Recommended `.env` overrides

You can customise the runtime via environment variables. A sample configuration tuned for `gpt-5-mini` looks like this:

```env
# Límites para GPT-5-Mini
AI_MODEL=gpt-5-mini
PRAPP_AI_TRIAGE_MODEL=gpt-5-mini
AI_TPM=500000
AI_RPM=500
AI_MICROBATCH=64
PRAPP_OPENAI_CONTEXT_SAFE_TOKENS=380000
PRAPP_AI_COLUMNS_MAX_TOKENS_PER_ITEM=256
```

Adjust these numbers based on the access tier in your OpenAI account.

Make sure the cost estimator is aligned with the active reasoning model. In `product_research_app/config.json`, set `aiCost.model` to `gpt-5-mini` (or update the persisted configuration through the UI) so that projected spend reflects the GPT-5 Mini rates.

## Auto-open y variables de entorno

El arranque por `python -m product_research_app` respeta varias variables útiles para entornos locales:

| Variable | Default | Descripción |
| --- | --- | --- |
| `PRAPP_HOST` | `127.0.0.1` | Host donde se enlaza el servidor Flask. |
| `PRAPP_PORT` | `8000` | Puerto TCP expuesto por la aplicación. |
| `PRAPP_AUTO_OPEN` | `1` | Controla la apertura automática del navegador (usa `0`, `false` o `no` para desactivarlo). |
| `PRAPP_BROWSER_URL` | `http://{host}:{port}/` | URL que se abrirá cuando el puerto esté listo. |

Además de `/healthz`, el backend expone `/health` para comprobaciones rápidas de estado (`200` + `{"status": "ok"}`).

### Variables de entorno de arranque

- `PRAPP_HOST` (por defecto `127.0.0.1`)
- `PRAPP_PORT` (por defecto `8000`)
- `PRAPP_AUTO_OPEN` (`1` por defecto; `0|false|no` para desactivar)
- `PRAPP_BROWSER_URL` (URL preferida a abrir; si no responde 200, se prueban `/app`, `/ui`, `/index.html`, `/dashboard`, `/`)
- `PRAPP_ROOT_REDIRECT` (si se define y no es `/`, la raíz `/` redirige ahí)

> El dev server de Werkzeug puede recibir handshakes TLS en un puerto HTTP (p.ej. de antivirus o probes HTTP/2); esos 400 se silencian para no ensuciar logs.
