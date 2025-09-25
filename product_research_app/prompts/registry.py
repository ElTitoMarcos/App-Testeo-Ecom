"""Registro de prompts para Prompt Maestro v4.

Fecha de actualización: 2025-09-25.
prompt_version = "prompt-maestro-v4".
"""

from __future__ import annotations

from typing import Any, Dict

PROMPT_VERSION = "prompt-maestro-v4"
PROMPT_RELEASE_DATE = "2025-09-25"

PROMPT_MASTER_V3_SYSTEM = """SYSTEM — PROMPT MAESTRO v3\nFecha de publicación: 2024-09-15\nIdentificador: prompt-maestro-v3\nEres Prompt Maestro v3, director de orquesta de la investigación de productos. Orquesta análisis, síntesis y recomendaciones fiables.\n\nReglas núcleo:\n1. Trabaja en español neutro, preciso y accionable.\n2. Nunca inventes datos ni referencias: si faltan, indica la carencia con claridad.\n3. No encierres respuestas JSON en bloques de código ni escapes comillas salvo que el esquema lo exija.\n4. Respeta los formatos solicitados (texto, listas, tablas o JSON) sin añadir emojis, banners ni notas superfluas.\n5. No repitas el texto de entrada salvo cuando la tarea lo pida explícitamente.\n6. Limpia HTML, Markdown u otras secuencias peligrosas antes de razonar; evita propagar código o scripts.\n7. Lee los bloques etiquetados (### CONTEXT_JSON, ### AGGREGATES, ### DATA) como JSON UTF-8 válido y preserva los identificadores tal cual.\n\nFallbacks oficiales:\n- Si la entrada es ilegible o falta información crítica, responde literalmente "ERROR: entrada inválida".\n- Si la tarea requiere datos inexistentes, responde "SIN DATOS".\n- Si no puedes garantizar la estructura pedida, responde "ERROR: formato".\n\nCuando la tarea lo pida, incluye el campo prompt_version con el valor "prompt-maestro-v3" sin alterarlo."""

PROMPT_MASTER_V4_SYSTEM = """SYSTEM — PROMPT MAESTRO v4
Fecha: 2025-09-25 • ID: prompt-maestro-v4

Rol: director de orquesta de investigación de productos. Produce análisis fiables y accionables.

Reglas núcleo (cost-aware):
1) Español neutro, conciso. 
2) No inventes datos; si faltan: “SIN DATOS”.
3) Respeta EXACTAMENTE los formatos pedidos; no añadas texto extra.
4) Limpia HTML/MD y evita arrastrar código.
5) Lee ### CONTEXT_JSON, ### AGGREGATES y ### DATA como JSON UTF-8 válido; preserva IDs.
6) Usa límites de caracteres cuando se indiquen.
7) En tareas JSON-only: imprime SOLO el JSON, sin fences ni comentarios.
8) Incluye siempre: "prompt_version":"prompt-maestro-v4" cuando se solicite.
Fallbacks globales:
- Entrada ilegible: "ERROR: entrada inválida"
- No puedes garantizar estructura: "ERROR: formato"
"""

PROMPT_DESIRE = """TAREA DESIRE — Extracción de Deseo (v4.2-L)
Objetivo: inferir el deseo dominante y su fuerza a partir de ### CONTEXT_JSON y/o ### DATA. Usa SOLO el material provisto.

Instinto (elige 1): health | sex | status | belonging | control | comfort
Magnitud: media de {scope, urgency, staying_power} (0–100)
Awareness: problem | solution | product | most
Competencia: low | mid | high
Estacionalidad: window ∈ {jan, feb, mar_apr, may, jun, jul_aug, sep, oct, nov, dec}

DESIRE STATEMENT — reglas:
- EXTENSIÓN OBLIGATORIA: entre 220 y 360 caracteres. Si quedas corto, añade cláusulas concisas separadas por “;”.
- Forma: 2–4 frases cortas que incluyan: resultado funcional + beneficio emocional + micro-escena de uso + diferenciador/objeción neutralizada.
- Estilo telegráfico, sin hype, sin claims médicos/ilegales, sin marcas.

signals: 3–8 tokens/rasgos del input que respalden el deseo.

SALIDA JSON estricta:
{
  "prompt_version": "prompt-maestro-v4",
  "desire_primary": "<health|sex|status|belonging|control|comfort>",
  "desire_statement": "<=360 chars, >=220 chars",
  "desire_magnitude": {
    "scope": <0-100>, "urgency": <0-100>, "staying_power": <0-100>, "overall": <0-100>
  },
  "awareness_level": "<problem|solution|product|most>",
  "competition_level": "<low|mid|high>",
  "competition_reason": "<=140 chars",
  "seasonality_hint": { "window": "<jan|feb|mar_apr|may|jun|jul_aug|sep|oct|nov|dec>", "confidence": <0-100> },
  "elevation_strategy": "<=140 chars",
  "signals": ["<token>", "..."]
}
Reglas:
- "overall" = round((scope+urgency+staying_power)/3).
- Sin señales → signals=[], y usa "SIN DATOS" en statement y reason.
- No añadas campos ni comentarios."""

PROMPT_A = """TAREA A — Radiografía del mercado\nObjetivo: sintetizar oportunidades y riesgos clave del dataset recibido.\nUsa exclusivamente ### CONTEXT_JSON.\n\nLímites:\n- Diagnóstico: 2–3 frases (≤80 palabras)\n- Oportunidades: hasta 3 viñetas\n- Riesgos: ≥1 viñeta\n- Próximos pasos: 2–3 acciones\n\nFormato:\nDiagnóstico\nHallazgos\nRiesgos\nPróximos pasos\nprompt_version: prompt-maestro-v4\n\nFallbacks:\n- <1 producto válido: “SIN DATOS” y listas vacías.\n- JSON ilegible: “ERROR: entrada inválida”."""

PROMPT_B = """TAREA B — Ponderaciones cuantitativas (JSON-only)\nObjetivo: convertir ### AGGREGATES en pesos 0–100 y orden de prioridad.\n\nSalida (SOLO JSON):\n{\n  "prompt_version":"prompt-maestro-v4",\n  "weights": { "market_momentum":<0-100>, "market_saturation":<0-100>, "offer_strength":<0-100>, "social_proof":<0-100>, "margin_quality":<0-100>, "logistics_ease":<0-100>, "validation_signal":<0-100>, "overall_priority":<0-100> },\n  "order": ["market_momentum","market_saturation","offer_strength","social_proof","margin_quality","logistics_ease","validation_signal","overall_priority"],\n  "notes":"<=180 chars"\n}\nReglas: valores 0–100; order sin duplicados; notes ≤180 chars.\nFallback: sin datos → todos 0, order=[], notes="SIN DATOS"."""

PROMPT_C = """TAREA C — Ángulos y mensajes (cost-aware)\nObjetivo: proponer 3 ángulos de venta accionables.\nUsa ### CONTEXT_JSON (pains, deseos, objeciones).\n\nFormato (tabla texto):\nÁngulo | Mensaje (≤120 chars) | Canal | Deseo (health/sex/status/belonging/control/comfort)\nNota final: prompt_version: prompt-maestro-v4\n\nFallback: sin señales → tabla vacía + “SIN DATOS”."""

PROMPT_D = """TAREA D — Plan de validación\nObjetivo: diseñar hasta 4 experimentos priorizados.\n\nFormato:\n1) Nombre\n- Hipótesis (≤120 chars)\n- Métrica de éxito\n- Recursos (personas/€ aprox.)\n- Costo: low|mid|high\n- ETA: días\n- Riesgo (≤120 chars)\nCierra con: prompt_version: prompt-maestro-v4\n\nFallback: sin contexto → “SIN DATOS”."""

PROMPT_E = """TAREA E — Resumen ejecutivo\nBloques: Situación (≤80 palabras), Oportunidad (≤80), Recomendación (≤80)\nConvicción: Alto|Medio|Bajo\nprompt_version: prompt-maestro-v4\nFallback: datos insuficientes → “SIN DATOS” en cada bloque y Convicción: “Baja”."""

PROMPT_E_AUTO = """TAREA E_auto — Decisión por lote (JSON-only)\nLee ### DATA. Para cada item:\n- status: aprobado|revisar|descartar\n- score, confidence: 0–100\n- summary: frase breve\n- reason: texto breve o null\n- next_step: texto breve o null\n- signals: tokens clave\n\nSalida:\n{\n  "prompt_version":"prompt-maestro-v4",\n  "items":[ { "id":<string|number>, "status":"...", "score":0-100, "confidence":0-100, "summary":"...", "reason":<string|null>, "next_step":<string|null>, "signals":["..."] }, ... ]\n}\nReglas: sin campos extra; signals puede estar vacío.\nFallback: ### DATA vacío → items=[], y si generas registros, reason="SIN DATOS"."""

_TASK_PROMPTS: Dict[str, str] = {
    "A": PROMPT_A,
    "B": PROMPT_B,
    "C": PROMPT_C,
    "D": PROMPT_D,
    "E": PROMPT_E,
    "E_auto": PROMPT_E_AUTO,
    "DESIRE": PROMPT_DESIRE,
}

JSON_ONLY: Dict[str, bool] = {
    "A": False,
    "B": True,
    "C": False,
    "D": False,
    "E": False,
    "E_auto": True,
    "DESIRE": True,
}

_TASK_B_METRICS = [
    "market_momentum",
    "market_saturation",
    "offer_strength",
    "social_proof",
    "margin_quality",
    "logistics_ease",
    "validation_signal",
    "overall_priority",
]

JSON_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "B": {
        "name": "prompt_maestro_v4_task_b",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["prompt_version", "weights", "order", "notes"],
            "properties": {
                "prompt_version": {"type": "string"},
                "weights": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(_TASK_B_METRICS),
                    "properties": {
                        metric: {"type": "number", "minimum": 0, "maximum": 100}
                        for metric in _TASK_B_METRICS
                    },
                },
                "order": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(_TASK_B_METRICS)},
                    "minItems": len(_TASK_B_METRICS),
                    "maxItems": len(_TASK_B_METRICS),
                    "uniqueItems": True,
                },
                "notes": {
                    "type": "string",
                    "maxLength": 180,
                },
            },
        },
    },
    "E_auto": {
        "name": "prompt_maestro_v4_task_e_auto",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["prompt_version", "items"],
            "properties": {
                "prompt_version": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "id",
                            "status",
                            "score",
                            "confidence",
                            "summary",
                            "reason",
                            "next_step",
                            "signals",
                        ],
                        "properties": {
                            "id": {"type": ["string", "number"]},
                            "status": {
                                "type": "string",
                                "enum": ["aprobado", "revisar", "descartar"],
                            },
                            "score": {"type": "number", "minimum": 0, "maximum": 100},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 100},
                            "summary": {"type": "string"},
                            "reason": {"type": ["string", "null"]},
                            "next_step": {"type": ["string", "null"]},
                            "signals": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 0,
                            },
                        },
                    },
                },
            },
        },
    },
    "DESIRE": {
        "name": "prompt_maestro_v4_task_desire",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "prompt_version",
                "desire_primary",
                "desire_statement",
                "desire_magnitude",
                "awareness_level",
                "competition_level",
                "competition_reason",
                "seasonality_hint",
                "elevation_strategy",
                "signals",
            ],
            "properties": {
                "prompt_version": {"type": "string"},
                "desire_primary": {
                    "type": "string",
                    "enum": [
                        "health",
                        "sex",
                        "status",
                        "belonging",
                        "control",
                        "comfort",
                    ],
                },
                "desire_statement": {
                    "type": "string",
                    "minLength": 220,
                    "maxLength": 360,
                },
                "desire_magnitude": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "scope",
                        "urgency",
                        "staying_power",
                        "overall",
                    ],
                    "properties": {
                        "scope": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "urgency": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "staying_power": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "overall": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                        },
                    },
                },
                "awareness_level": {
                    "type": "string",
                    "enum": [
                        "problem",
                        "solution",
                        "product",
                        "most",
                    ],
                },
                "competition_level": {
                    "type": "string",
                    "enum": ["low", "mid", "high"],
                },
                "competition_reason": {
                    "type": "string",
                    "maxLength": 140,
                },
                "seasonality_hint": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["window", "confidence"],
                    "properties": {
                        "window": {
                            "type": "string",
                            "enum": [
                                "jan",
                                "feb",
                                "mar_apr",
                                "may",
                                "jun",
                                "jul_aug",
                                "sep",
                                "oct",
                                "nov",
                                "dec",
                            ],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                        },
                    },
                },
                "elevation_strategy": {
                    "type": "string",
                    "maxLength": 140,
                },
                "signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                },
            },
        },
    },
}


def _normalize_task(task: str) -> str:
    if not isinstance(task, str):
        raise KeyError("task must be a string")
    token = task.strip()
    if not token:
        raise KeyError("task must not be empty")
    normalized = token.replace("-", "_")
    upper = normalized.upper()
    if upper == "E_AUTO" or upper == "EAUTO":
        return "E_auto"
    stripped = upper.replace("_", "")
    if stripped == "DESIRE":
        return "DESIRE"
    if upper in {"A", "B", "C", "D", "E"}:
        return upper
    raise KeyError(f"Unknown task: {task}")


def get_system_prompt(task: str) -> str:
    """Return the system prompt for Prompt Maestro v4."""
    _normalize_task(task)
    return PROMPT_MASTER_V4_SYSTEM


def get_task_prompt(task: str) -> str:
    """Return the user prompt template for the given task."""
    canonical = _normalize_task(task)
    return _TASK_PROMPTS[canonical]


def normalize_task(task: str) -> str:
    """Public helper returning the canonical identifier for a task."""
    return _normalize_task(task)


def is_json_only(task: str) -> bool:
    """Return True if the task must respond strictly in JSON."""
    canonical = _normalize_task(task)
    return JSON_ONLY.get(canonical, False)


def get_json_schema(task: str) -> Dict[str, Any] | None:
    """Return the JSON schema associated with a task, if any."""
    canonical = _normalize_task(task)
    return JSON_SCHEMAS.get(canonical)


__all__ = [
    "PROMPT_MASTER_V3_SYSTEM",
    "PROMPT_MASTER_V4_SYSTEM",
    "PROMPT_DESIRE",
    "PROMPT_A",
    "PROMPT_B",
    "PROMPT_C",
    "PROMPT_D",
    "PROMPT_E",
    "PROMPT_E_AUTO",
    "PROMPT_VERSION",
    "PROMPT_RELEASE_DATE",
    "JSON_ONLY",
    "JSON_SCHEMAS",
    "get_system_prompt",
    "get_task_prompt",
    "normalize_task",
    "is_json_only",
    "get_json_schema",
]
