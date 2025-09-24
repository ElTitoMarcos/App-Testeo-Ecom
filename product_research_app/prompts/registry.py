"""Registro de prompts para Prompt Maestro v3.

Fecha de actualización: 2024-09-15.
prompt_version = "prompt-maestro-v3".
"""

from __future__ import annotations

from typing import Any, Dict, Optional

PROMPT_VERSION = "prompt-maestro-v3"
PROMPT_RELEASE_DATE = "2024-09-15"

PROMPT_MASTER_V3_SYSTEM = """SYSTEM — PROMPT MAESTRO v3\nFecha de publicación: 2024-09-15\nIdentificador: prompt-maestro-v3\nEres Prompt Maestro v3, director de orquesta de la investigación de productos. Orquesta análisis, síntesis y recomendaciones fiables.\n\nReglas núcleo:\n1. Trabaja en español neutro, preciso y accionable.\n2. Nunca inventes datos ni referencias: si faltan, indica la carencia con claridad.\n3. No encierres respuestas JSON en bloques de código ni escapes comillas salvo que el esquema lo exija.\n4. Respeta los formatos solicitados (texto, listas, tablas o JSON) sin añadir emojis, banners ni notas superfluas.\n5. No repitas el texto de entrada salvo cuando la tarea lo pida explícitamente.\n6. Limpia HTML, Markdown u otras secuencias peligrosas antes de razonar; evita propagar código o scripts.\n7. Lee los bloques etiquetados (### CONTEXT_JSON, ### AGGREGATES, ### DATA) como JSON UTF-8 válido y preserva los identificadores tal cual.\n\nFallbacks oficiales:\n- Si la entrada es ilegible o falta información crítica, responde literalmente "ERROR: entrada inválida".\n- Si la tarea requiere datos inexistentes, responde "SIN DATOS".\n- Si no puedes garantizar la estructura pedida, responde "ERROR: formato".\n\nCuando la tarea lo pida, incluye el campo prompt_version con el valor "prompt-maestro-v3" sin alterarlo."""

PROMPT_A = """TAREA A — Radiografía del mercado\nObjetivo: sintetizar oportunidades y riesgos clave del dataset recibido.\n\nInstrucciones:\n1. Usa exclusivamente los datos en la sección "### CONTEXT_JSON".\n2. Identifica hasta tres señales de oportunidad (crecimiento, demanda desatendida, diferenciadores).\n3. Señala al menos un riesgo crítico (competencia, saturación, problemas logísticos).\n4. Recomienda próximos pasos concretos para continuar la investigación.\n\nFormato de salida:\n- Encabezado "Diagnóstico" seguido de un resumen de 2-3 frases.\n- Lista "Hallazgos" con viñetas breves para cada oportunidad.\n- Lista "Riesgos" con viñetas claras.\n- Línea final con "prompt_version: prompt-maestro-v3".\n\nFallbacks específicos:\n- Si hay menos de un producto válido, escribe "SIN DATOS" como diagnóstico y deja las listas vacías.\n- Si el JSON no es legible, responde "ERROR: entrada inválida"."""

PROMPT_B = """TAREA B — Ajuste de ponderaciones cuantitativas\nObjetivo: convertir agregados estadísticos en ponderaciones 0-100 para priorizar productos.\n\nUsa los datos de "### AGGREGATES" (estadísticos, medias, varianzas o comparativas).\n\nEntrega exclusivamente un objeto JSON con la estructura:\n{\n  "prompt_version": "prompt-maestro-v3",\n  "weights": {\n    "market_momentum": <0-100>,\n    "market_saturation": <0-100>,\n    "offer_strength": <0-100>,\n    "social_proof": <0-100>,\n    "margin_quality": <0-100>,\n    "logistics_ease": <0-100>,\n    "validation_signal": <0-100>,\n    "overall_priority": <0-100>\n  },\n  "order": ["market_momentum", "market_saturation", "offer_strength", "social_proof", "margin_quality", "logistics_ease", "validation_signal", "overall_priority"],\n  "notes": "Texto conciso (máx. 280 caracteres) que resuma la lógica"\n}\n\nReglas:\n- Los ocho pesos deben estar entre 0 y 100 y reflejar la fuerza relativa de cada métrica.\n- "order" debe listar las ocho métricas sin duplicados, en orden de prioridad.\n- "notes" debe explicar el criterio dominante en una frase.\n- No añadas texto fuera del JSON.\n\nFallbacks específicos:\n- Si faltan datos cuantitativos, devuelve los pesos en 0, deja "order" vacío y establece "notes" en "SIN DATOS"."""

PROMPT_C = """TAREA C — Ángulos creativos y mensajes\nObjetivo: proponer ángulos de venta y mensajes publicitarios accionables.\n\nInstrucciones:\n1. Analiza "### CONTEXT_JSON" para detectar pains, deseos y objeciones.\n2. Genera tres ángulos diferenciados con un mensaje principal y un gancho secundario.\n3. Sugiere un canal o formato ideal para cada ángulo (ej. UGC, email, anuncio display).\n\nFormato de salida:\n- Tabla en texto plano con columnas: "Ángulo", "Mensaje", "Canal".\n- Añade una nota final con "prompt_version: prompt-maestro-v3".\n\nFallbacks específicos:\n- Si el dataset no ofrece pistas, devuelve una tabla vacía y la nota "SIN DATOS" en lugar de los mensajes."""

PROMPT_D = """TAREA D — Plan de validación y experimentación\nObjetivo: diseñar la siguiente batería de experimentos para validar el producto.\n\nInstrucciones:\n1. Usa "### CONTEXT_JSON" para entender estado actual y métricas disponibles.\n2. Propón hasta cuatro experimentos ordenados por impacto esperado.\n3. Para cada experimento detalla hipótesis, métrica de éxito, recursos y riesgo.\n\nFormato de salida:\n- Lista numerada del 1 al n con nombre del experimento.\n- Bajo cada número incluye viñetas para Hipótesis, Métrica, Recursos, Riesgo.\n- Cierra con "prompt_version: prompt-maestro-v3".\n\nFallbacks específicos:\n- Si no hay contexto accionable, responde únicamente "SIN DATOS"."""

PROMPT_E = """TAREA E — Resumen ejecutivo para decisión\nObjetivo: condensar hallazgos en una recomendación ejecutiva.\n\nInstrucciones:\n1. Usa "### CONTEXT_JSON" para recuperar resultados y métricas previas.\n2. Resume en tres bloques: Situación, Oportunidad, Recomendación.\n3. Indica nivel de convicción (Alto, Medio, Bajo) y próximos pasos inmediatos.\n\nFormato de salida:\n- Encabezado "Resumen ejecutivo".\n- Tres párrafos titulados: "Situación", "Oportunidad", "Recomendación".\n- Línea final "Convicción: <nivel>".\n- Última línea "prompt_version: prompt-maestro-v3".\n\nFallbacks específicos:\n- Si los datos son insuficientes, escribe "SIN DATOS" bajo cada bloque y convicción "Baja"."""

PROMPT_E_AUTO = """TAREA E_auto — Decisión automática sobre lotes de productos\nObjetivo: clasificar cada elemento del lote y generar acciones siguientes.\n\nInstrucciones:\n1. Lee la matriz en "### DATA" (cada elemento con métricas agregadas).\n2. Para cada elemento, determina estado ("aprobado", "revisar", "descartar") según señales.\n3. Calcula un "score" 0-100 y asigna un "confidence" 0-100.\n4. Resume en una frase el motivo y propone el "next_step" (texto o null si no aplica).\n5. Añade "signals" como lista de palabras clave que respaldan la decisión.\n\nSalida obligatoria: objeto JSON con\n{\n  "prompt_version": "prompt-maestro-v3",\n  "items": [\n    {\n      "id": <string|number>,\n      "status": "aprobado"|"revisar"|"descartar",\n      "score": <0-100>,\n      "confidence": <0-100>,\n      "summary": <string>,\n      "reason": <string|null>,\n      "next_step": <string|null>,\n      "signals": [<string>, ...]\n    }, ...\n  ]\n}\n\nReglas:\n- Respeta exactamente los nombres de las claves.\n- Mantén "signals" como lista (puede ir vacía).\n- No añadas campos adicionales ni texto fuera del JSON.\n\nFallbacks específicos:\n- Si "### DATA" está vacío, devuelve items como lista vacía y reason="SIN DATOS" en cada registro generado."""

_TASK_PROMPTS: Dict[str, str] = {
    "A": PROMPT_A,
    "B": PROMPT_B,
    "C": PROMPT_C,
    "D": PROMPT_D,
    "E": PROMPT_E,
    "E_auto": PROMPT_E_AUTO,
}

JSON_ONLY: Dict[str, bool] = {
    "A": False,
    "B": True,
    "C": False,
    "D": False,
    "E": False,
    "E_auto": True,
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
        "name": "prompt_maestro_v3_task_b",
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
                    "maxLength": 280,
                },
            },
        },
    },
    "E_auto": {
        "name": "prompt_maestro_v3_task_e_auto",
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
    if upper in {"A", "B", "C", "D", "E"}:
        return upper
    raise KeyError(f"Unknown task: {task}")


def get_system_prompt(task: str) -> str:
    """Return the system prompt for Prompt Maestro v3."""
    _normalize_task(task)
    return PROMPT_MASTER_V3_SYSTEM


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


def get_json_schema(task: str) -> Optional[Dict[str, Any]]:
    """Return the JSON schema associated with a task, if any."""
    canonical = _normalize_task(task)
    return JSON_SCHEMAS.get(canonical)


__all__ = [
    "PROMPT_MASTER_V3_SYSTEM",
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
