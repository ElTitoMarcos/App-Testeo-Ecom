# AI runtime tuning

The AI enrichment flow now uses configurable micro-batching and parallel HTTP calls. Runtime parameters can be adjusted through environment variables (or `config.json`) without touching the code.

## Environment flags

| Variable | Default | Description |
| --- | --- | --- |
| `AI_MODEL` | `gpt-5-mini` | Override the chat completion model used for column filling. |
| `AI_MICROBATCH` | `24` | Number of products per request. Requests are further reduced automatically if the token estimate would exceed `AI_TPM`. |
| `AI_PARALLELISM` | `24` | Maximum concurrent requests sent to the model. |
| `AI_TRUNC_TITLE` | `180` | Character cap applied to product titles before building the prompt. |
| `AI_TRUNC_DESC` | `800` | Character cap for descriptions and bullet lists in the prompt. |
| `AI_TIMEOUT` | `45` | Total HTTP timeout (seconds) per OpenAI request. |
| `AI_RPM` | `600` | Soft limit for requests per minute enforced via an async semaphore. Set to `0` to disable. |
| `AI_TPM` | `450000` | Soft limit for prompt tokens per minute; batches are truncated when the estimated prompt would exceed this value. |

All variables also exist under `config["ai"]` so they can be persisted in `product_research_app/config.json` when running locally.

The defaults assume the GPT-5 Mini batch queue with 500k tokens per minute and a 5M tokens per day allowance. If you need to tighten the guardrails for testing or quotas, lower `AI_TPM`/`AI_RPM` or override `PRAPP_OPENAI_TPD`; otherwise you can leave the defaults to enjoy minimal artificial throttling.

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

Logs tagged `ai_columns.request` are useful to verify concurrency in smoke tests: with the GPT-5 Mini defaults you should see the system happily running a few dozen overlapping requests for 100 products without tripping the new 500k TPM / 5M TPD allowances. The start/end summary logs also emit `tpm_cap`, `rpm_cap` and `tpd_cap` fields so alerting rules can be updated around the higher quotas.

## Troubleshooting

- Mensajes code 400, Bad request version ('JJ…') pueden aparecer si algún cliente intenta TLS contra el puerto HTTP. No afectan al pipeline.
