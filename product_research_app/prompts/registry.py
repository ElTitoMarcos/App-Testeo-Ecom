"""Registro de prompts para Prompt Maestro v3.

Fecha de actualización: 2024-10-05.
prompt_version = "prompt-maestro-v3".
"""

from __future__ import annotations

from typing import Any, Dict

PROMPT_VERSION = "prompt-maestro-v3"
PROMPT_RELEASE_DATE = "2024-10-05"

PROMPT_MASTER_V3_SYSTEM = """SYSTEM — PROMPT MAESTRO v3\nFecha de publicación: 2024-09-15\nIdentificador: prompt-maestro-v3\nEres Prompt Maestro v3, director de orquesta de la investigación de productos. Orquesta análisis, síntesis y recomendaciones fiables.\n\nReglas núcleo:\n1. Trabaja en español neutro, preciso y accionable.\n2. Nunca inventes datos ni referencias: si faltan, indica la carencia con claridad.\n3. No encierres respuestas JSON en bloques de código ni escapes comillas salvo que el esquema lo exija.\n4. Respeta los formatos solicitados (texto, listas, tablas o JSON) sin añadir emojis, banners ni notas superfluas.\n5. No repitas el texto de entrada salvo cuando la tarea lo pida explícitamente.\n6. Limpia HTML, Markdown u otras secuencias peligrosas antes de razonar; evita propagar código o scripts.\n7. Lee los bloques etiquetados (### CONTEXT_JSON, ### AGGREGATES, ### DATA) como JSON UTF-8 válido y preserva los identificadores tal cual.\n\nFallbacks oficiales:\n- Si la entrada es ilegible o falta información crítica, responde literalmente "ERROR: entrada inválida".\n- Si la tarea requiere datos inexistentes, responde "SIN DATOS".\n- Si no puedes garantizar la estructura pedida, responde "ERROR: formato".\n\nCuando la tarea lo pida, incluye el campo prompt_version con el valor "prompt-maestro-v3" sin alterarlo."""

PROMPT_A = """TAREA A — Radiografía del mercado\nObjetivo: sintetizar oportunidades y riesgos clave del dataset recibido.\n\nInstrucciones:\n1. Usa exclusivamente los datos en la sección "### CONTEXT_JSON".\n2. Identifica hasta tres señales de oportunidad (crecimiento, demanda desatendida, diferenciadores).\n3. Señala al menos un riesgo crítico (competencia, saturación, problemas logísticos).\n4. Recomienda próximos pasos concretos para continuar la investigación.\n\nFormato de salida:\n- Encabezado "Diagnóstico" seguido de un resumen de 2-3 frases.\n- Lista "Hallazgos" con viñetas breves para cada oportunidad.\n- Lista "Riesgos" con viñetas claras.\n- Línea final con "prompt_version: prompt-maestro-v3".\n\nFallbacks específicos:\n- Si hay menos de un producto válido, escribe "SIN DATOS" como diagnóstico y deja las listas vacías.\n- Si el JSON no es legible, responde "ERROR: entrada inválida"."""

PROMPT_B = """TAREA B — Ajuste de ponderaciones cuantitativas\nObjetivo: convertir agregados estadísticos en ponderaciones 0-100 para priorizar productos.\n\nUsa los datos de "### AGGREGATES" (estadísticos, medias, varianzas o comparativas).\n\nEntrega exclusivamente un objeto JSON con la estructura:\n{\n  "prompt_version": "prompt-maestro-v3",\n  "weights": {\n    "market_momentum": <0-100>,\n    "market_saturation": <0-100>,\n    "offer_strength": <0-100>,\n    "social_proof": <0-100>,\n    "margin_quality": <0-100>,\n    "logistics_ease": <0-100>,\n    "validation_signal": <0-100>,\n    "overall_priority": <0-100>\n  },\n  "order": ["market_momentum", "market_saturation", "offer_strength", "social_proof", "margin_quality", "logistics_ease", "validation_signal", "overall_priority"],\n  "notes": "Texto conciso (máx. 280 caracteres) que resuma la lógica"\n}\n\nReglas:\n- Los ocho pesos deben estar entre 0 y 100 y reflejar la fuerza relativa de cada métrica.\n- "order" debe listar las ocho métricas sin duplicados, en orden de prioridad.\n- "notes" debe explicar el criterio dominante en una frase.\n- No añadas texto fuera del JSON.\n\nFallbacks específicos:\n- Si faltan datos cuantitativos, devuelve los pesos en 0, deja "order" vacío y establece "notes" en "SIN DATOS"."""

PROMPT_C = """TAREA C — Ángulos creativos y mensajes\nObjetivo: proponer ángulos de venta y mensajes publicitarios accionables.\n\nInstrucciones:\n1. Analiza "### CONTEXT_JSON" para detectar pains, deseos y objeciones.\n2. Genera tres ángulos diferenciados con un mensaje principal y un gancho secundario.\n3. Sugiere un canal o formato ideal para cada ángulo (ej. UGC, email, anuncio display).\n\nFormato de salida:\n- Tabla en texto plano con columnas: "Ángulo", "Mensaje", "Canal".\n- Añade una nota final con "prompt_version: prompt-maestro-v3".\n\nFallbacks específicos:\n- Si el dataset no ofrece pistas, devuelve una tabla vacía y la nota "SIN DATOS" en lugar de los mensajes."""

PROMPT_D = """TAREA D — Plan de validación y experimentación\nObjetivo: diseñar la siguiente batería de experimentos para validar el producto.\n\nInstrucciones:\n1. Usa "### CONTEXT_JSON" para entender estado actual y métricas disponibles.\n2. Propón hasta cuatro experimentos ordenados por impacto esperado.\n3. Para cada experimento detalla hipótesis, métrica de éxito, recursos y riesgo.\n\nFormato de salida:\n- Lista numerada del 1 al n con nombre del experimento.\n- Bajo cada número incluye viñetas para Hipótesis, Métrica, Recursos, Riesgo.\n- Cierra con "prompt_version: prompt-maestro-v3".\n\nFallbacks específicos:\n- Si no hay contexto accionable, responde únicamente "SIN DATOS"."""

PROMPT_E = """TAREA E — Resumen ejecutivo para decisión\nObjetivo: condensar hallazgos en una recomendación ejecutiva.\n\nInstrucciones:\n1. Usa "### CONTEXT_JSON" para recuperar resultados y métricas previas.\n2. Resume en tres bloques: Situación, Oportunidad, Recomendación.\n3. Indica nivel de convicción (Alto, Medio, Bajo) y próximos pasos inmediatos.\n\nFormato de salida:\n- Encabezado "Resumen ejecutivo".\n- Tres párrafos titulados: "Situación", "Oportunidad", "Recomendación".\n- Línea final "Convicción: <nivel>".\n- Última línea "prompt_version: prompt-maestro-v3".\n\nFallbacks específicos:\n- Si los datos son insuficientes, escribe "SIN DATOS" bajo cada bloque y convicción "Baja"."""

PROMPT_E_AUTO = """TAREA E_auto — Decisión automática sobre lotes de productos\nObjetivo: clasificar cada elemento del lote y generar acciones siguientes.\n\nInstrucciones:\n1. Lee la matriz en "### DATA" (cada elemento con métricas agregadas).\n2. Para cada elemento, determina estado ("aprobado", "revisar", "descartar") según señales.\n3. Calcula un "score" 0-100 y asigna un "confidence" 0-100.\n4. Resume en una frase el motivo y propone el "next_step" (texto o null si no aplica).\n5. Añade "signals" como lista de palabras clave que respaldan la decisión.\n\nSalida obligatoria: objeto JSON con\n{\n  "prompt_version": "prompt-maestro-v3",\n  "items": [\n    {\n      "id": <string|number>,\n      "status": "aprobado"|"revisar"|"descartar",\n      "score": <0-100>,\n      "confidence": <0-100>,\n      "summary": <string>,\n      "reason": <string|null>,\n      "next_step": <string|null>,\n      "signals": [<string>, ...]\n    }, ...\n  ]\n}\n\nReglas:\n- Respeta exactamente los nombres de las claves.\n- Mantén "signals" como lista (puede ir vacía).\n- No añadas campos adicionales ni texto fuera del JSON.\n\nFallbacks específicos:\n- Si "### DATA" está vacío, devuelve items como lista vacía y reason="SIN DATOS" en cada registro generado."""

PROMPT_DESIRE = """TAREA DESIRE — JTBD Outcome (v6.4 • 3–4 líneas, anti-producto)
Objetivo: redactar el DESEO HUMANO subyacente, no la cosa. Usa ###CONTEXT_JSON/###DATA y, si hay navegador, consulta hasta 5 fuentes (foros/reseñas/comentarios) para extraer señales; si no hay navegador, continúa solo con el input.

Marco Evolve + JTBD (compacto):
- instincts: health|sex|status|belonging|control|comfort   # elige 1
- job-to-be-done: progreso deseado en situación concreta + emoción buscada + obstáculo eliminado
- tech_problems (0–1): complexity|overwhelm|fragility|maintenance|incompatibility|obsolescence
- overall = round((scope+urgency+staying_power)/3)

DESIRE STATEMENT — reglas duras (único por item):
- 280–420 caracteres (≈3–4 líneas), 2–4 frases o cláusulas “;”.
- Empieza con un verbo de resultado (lograr, recuperar, sentir, mantener, simplificar, proteger…).
- Incluye: resultado funcional + emoción + micro-escena (cuándo/dónde/para quién) + fricción neutralizada (tiempo, caos, dolor, vergüenza…).
- PROHIBIDO: marcas o modelos; nombres de producto/categoría (crema, champú, aspiradora, cuchillos, cepillo, parches, figura, etc.); packs/cantidades; medidas (ml/W/cm); materiales; hype; claims médicos. Si aparecen, REESCRIBE hasta eliminarlos.
- Evita muletillas genéricas (“sin esfuerzo”, “premium”, etc.) y la repetición de ideas.

Estacionalidad:
- seasonality_hint.window ∈ {jan,feb,mar_apr,may,jun,jul_aug,sep,oct,nov,dec} según el tipo de deseo.

SALIDA JSON (estricta):
{
  "prompt_version":"prompt-maestro-v4",
  "desire_primary":"<health|sex|status|belonging|control|comfort>",
  "desire_statement":"<=420, >=280",
  "desire_magnitude":{"scope":0-100,"urgency":0-100,"staying_power":0-100,"overall":0-100},
  "awareness_level":"<problem|solution|product|most>",
  "competition_level":"<low|mid|high>",
  "competition_reason":"<=140",
  "seasonality_hint":{"window":"<jan|feb|mar_apr|may|jun|jul_aug,sep,oct,nov,dec>","confidence":0-100},
  "elevation_strategy":"<=140",
  "signals":["t1","t2","t3"]
}

Reglas finales:
- Antes de imprimir, AUTOCHEQUEA: si tu texto contiene nombres de producto/categoría, medidas o marcas, reescribe el “desire_statement”.
- No añadas campos ni comentarios."""

PROMPT_TODO_TERRENO = """TAREA TODO_TERRENO — Generación de código todo-terreno
Objetivo: producir código ejecutable según los parámetros recibidos.

Recibirás un JSON en "### REQUEST" con los campos:
- "lenguaje" (str) — lenguaje de programación destino.
- "objetivo" (str) — descripción del problema a resolver.
- "version" (str) — versión o runtime a respetar.
- "entrada" (str) — formato de entrada esperado.
- "salida" (str) — formato de salida esperado.
- "requisitos" (array[str]) — restricciones funcionales obligatorias.
- "casos_borde" (array[str] | str) — escenarios límite que debes manejar.
- "estilo" (str) — guía de estilo o patrones a seguir.

Instrucciones:
1. Sustituye cada marcador {…} del esqueleto con los valores de "### REQUEST".
2. Diseña funciones pequeñas, nombres claros y sin dependencias innecesarias.
3. Asegura validaciones de entrada y cobertura de los casos borde listados.
4. Añade comentarios concisos solo cuando aporten claridad.

Formato de salida:
- Entrega exclusivamente el código final dentro de ```<lenguaje>``` sin texto adicional.

Fallbacks específicos:
- Si falta "lenguaje" u "objetivo", responde exactamente "SIN DATOS"."""

PROMPT_REFACTOR = """TAREA REFACTOR — Mejora incremental de código
Objetivo: refactorizar el código suministrado manteniendo el comportamiento.

En "### REQUEST" recibirás:
- "lenguaje" (str).
- "principios" (array[str]) — reglas adicionales a priorizar.
- "codigo" (str) — implementación actual.

Instrucciones:
1. Refactoriza aplicando SRP y DRY, además de los principios extra indicados.
2. Extrae funciones o estructuras auxiliares cuando simplifiquen la lectura.
3. Conserva la API pública (firmas exportadas, nombres públicos) sin cambios.
4. No introduzcas dependencias nuevas.

Formato de salida:
- Devuelve únicamente el código refactorizado dentro de ```<lenguaje>```.

Fallbacks específicos:
- Si "codigo" está vacío o falta, responde "SIN DATOS"."""

PROMPT_BUGFIX = """TAREA BUGFIX — Corrección mínima segura
Objetivo: solucionar el bug descrito con el menor cambio posible.

Datos en "### REQUEST":
- "lenguaje" (str).
- "lenguaje_pruebas" (str).
- "descripcion_bug" (str) — síntoma o causa observada.
- "codigo" (str) — estado actual.

Instrucciones:
1. Identifica la causa raíz y coméntala en una línea al inicio del archivo resultante.
2. Aplica el fix más acotado que resuelva el problema sin regresiones.
3. Añade una prueba unitaria que falle antes del fix y pase después.
4. Mantén estilos y convenciones existentes.

Formato de salida:
- Primero el archivo corregido en ```<lenguaje>```.
- Luego las pruebas en ```<lenguaje_pruebas>```.

Fallbacks específicos:
- Si no se provee "codigo", responde "SIN DATOS"."""

PROMPT_UNIT_TESTS = """TAREA UNIT_TESTS — Generación de pruebas
Objetivo: escribir pruebas unitarias exhaustivas para el código dado.

Entrada "### REQUEST":
- "lenguaje" (str) — lenguaje de las pruebas.
- "framework" (str) — framework de testing.
- "codigo" (str) — implementación a cubrir.
- "cobertura" (array[str]) — lista de escenarios clave (felices, borde, error).

Instrucciones:
1. Usa los escenarios de "cobertura" como guía mínima; añade otros si aportan valor.
2. Emplea nombres de test descriptivos y aislados.
3. Configura los fixtures o mocks necesarios sin dependencias externas.
4. Respeta el estilo idiomático del framework indicado.

Formato de salida:
- Devuelve solo el archivo de pruebas dentro de ```<lenguaje>```.

Fallbacks específicos:
- Si "framework" o "codigo" faltan, responde "SIN DATOS"."""

PROMPT_DOCSTRINGS = """TAREA DOCSTRINGS — Documentar y tipar
Objetivo: añadir docstrings y anotaciones de tipo sin alterar la lógica.

Datos en "### REQUEST":
- "lenguaje" (str).
- "estilo_doc" (str) — formato de docstring (ej. Google, NumPy, JSDoc).
- "codigo" (str) — fuente actual.

Instrucciones:
1. Aplica el estilo solicitado en todas las funciones o clases públicas.
2. Añade anotaciones de tipo donde sean inferibles sin introducir dependencias nuevas.
3. No modifiques la lógica ni el flujo de control existente.
4. Mantén el orden original de definiciones.

Formato de salida:
- Entrega el código actualizado dentro de ```<lenguaje>```.

Fallbacks específicos:
- Si falta "codigo", responde "SIN DATOS"."""

PROMPT_MIGRATION = """TAREA MIGRATION — Conversión entre lenguajes
Objetivo: trasladar la implementación manteniendo el comportamiento.

"### REQUEST" provee:
- "origen" (str) — lenguaje de entrada.
- "destino" (str) — lenguaje objetivo.
- "codigo" (str) — código original.
- "dependencias" (array[str]) — paquetes requeridos en el destino.

Instrucciones:
1. Reescribe el código de manera idiomática en el lenguaje destino.
2. Añade al inicio comentarios con instrucciones de instalación de dependencias nuevas, si las hay.
3. Conserva estructura y comportamiento; adapta APIs o librerías equivalentes.
4. No mezcles fragmentos de ambos lenguajes en la salida.

Formato de salida:
- Devuelve solo el código final en ```<destino>```.

Fallbacks específicos:
- Si "codigo" está vacío, responde "SIN DATOS"."""

PROMPT_API_ENDPOINT = """TAREA API_ENDPOINT — Implementación de endpoint
Objetivo: crear un endpoint conforme al framework y esquema especificados.

Entrada en "### REQUEST":
- "framework" (str) — por ejemplo FastAPI, Express.
- "lenguaje" (str).
- "metodo" (str) — verbo HTTP.
- "ruta" (str).
- "request_schema" (str) — descripción o JSON Schema del body.
- "response_schema" (str) — estructura esperada en 200.
- "errores" (array[str]) — lista de errores que manejar.

Instrucciones:
1. Define el endpoint con validación del request body según el esquema.
2. Implementa respuestas 200 y errores declarados con mensajes claros.
3. Añade manejo de excepciones genéricas y específicas donde aplique.
4. Evita dependencias innecesarias y configura el archivo listo para ejecución.

Formato de salida:
- Devuelve un único archivo listo para correr en ```<lenguaje>```.

Fallbacks específicos:
- Si falta "framework" o "ruta", responde "SIN DATOS"."""

PROMPT_SQL_QUERY = """TAREA SQL_QUERY — Consulta a partir de esquema
Objetivo: redactar una consulta SQL precisa según el objetivo recibido.

En "### REQUEST" se incluye:
- "esquema" (str) — definición de tablas o columnas.
- "objetivo" (str).
- "dialecto" (str) — motor objetivo.

Instrucciones:
1. Usa CTEs si facilitan la claridad o reutilización.
2. Ajusta la sintaxis al dialecto indicado sin extensiones propietarias.
3. Asegura que la consulta responda exactamente al objetivo planteado.
4. No devuelvas explicaciones ni comentarios fuera del SQL.

Formato de salida:
- Entrega solo la consulta encerrada en ```sql```.

Fallbacks específicos:
- Si falta "esquema" u "objetivo", responde "SIN DATOS"."""

PROMPT_CLI_TOOL = """TAREA CLI_TOOL — Herramienta de línea de comandos
Objetivo: generar una CLI compacta que cumpla el objetivo indicado.

Datos "### REQUEST":
- "lenguaje" (str).
- "objetivo" (str).
- "flags" (array[str]) — banderas soportadas.

Instrucciones:
1. Implementa la CLI en un solo archivo.
2. Procesa las banderas indicadas y produce salida JSON válida en stdout.
3. Maneja errores de entrada con mensajes claros y códigos adecuados.
4. Evita dependencias externas salvo que se especifique lo contrario.

Formato de salida:
- Devuelve únicamente el código dentro de ```<lenguaje>```.

Fallbacks específicos:
- Si "lenguaje" u "objetivo" faltan, responde "SIN DATOS"."""

PROMPT_JSON_REPORT = """TAREA JSON_REPORT — Generación estricta de JSON
Objetivo: producir un JSON estricto con el puntaje por ítem.

Entrada "### REQUEST":
- "items" (array[object]) — cada objeto describe una entrada a evaluar.

Instrucciones:
1. Para cada entrada genera "id" (número), "score" (0..1 con dos decimales) y "reason" (≤12 palabras).
2. Usa números con dos decimales redondeados; valores fuera de rango deben ajustarse al límite más cercano.
3. "reason" debe ser concisa y basada en los datos suministrados.
4. No añadas texto fuera del JSON ni comentarios.

Formato de salida:
- Responde únicamente con un objeto JSON válido que siga la estructura solicitada.

Fallbacks específicos:
- Si "items" está vacío, responde {"items": []}."""

_TASK_PROMPTS: Dict[str, str] = {
    "A": PROMPT_A,
    "B": PROMPT_B,
    "C": PROMPT_C,
    "D": PROMPT_D,
    "E": PROMPT_E,
    "E_auto": PROMPT_E_AUTO,
    "DESIRE": PROMPT_DESIRE,
    "TODO_TERRENO": PROMPT_TODO_TERRENO,
    "REFACTOR": PROMPT_REFACTOR,
    "BUGFIX": PROMPT_BUGFIX,
    "UNIT_TESTS": PROMPT_UNIT_TESTS,
    "DOCSTRINGS": PROMPT_DOCSTRINGS,
    "MIGRATION": PROMPT_MIGRATION,
    "API_ENDPOINT": PROMPT_API_ENDPOINT,
    "SQL_QUERY": PROMPT_SQL_QUERY,
    "CLI_TOOL": PROMPT_CLI_TOOL,
    "JSON_REPORT": PROMPT_JSON_REPORT,
}

JSON_ONLY: Dict[str, bool] = {
    "A": False,
    "B": True,
    "C": False,
    "D": False,
    "E": False,
    "E_auto": True,
    "DESIRE": True,
    "TODO_TERRENO": False,
    "REFACTOR": False,
    "BUGFIX": False,
    "UNIT_TESTS": False,
    "DOCSTRINGS": False,
    "MIGRATION": False,
    "API_ENDPOINT": False,
    "SQL_QUERY": False,
    "CLI_TOOL": False,
    "JSON_REPORT": True,
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
    "JSON_REPORT": {
        "name": "prompt_maestro_v3_task_json_report",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "score", "reason"],
                        "properties": {
                            "id": {"type": "number"},
                            "score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "multipleOf": 0.01,
                            },
                            "reason": {
                                "type": "string",
                                "maxLength": 120,
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
                "prompt_version": {"type": "string", "enum": ["prompt-maestro-v4"]},
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
                    "minLength": 280,
                    "maxLength": 420,
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
                        "scope": {"type": "number", "minimum": 0, "maximum": 100},
                        "urgency": {"type": "number", "minimum": 0, "maximum": 100},
                        "staying_power": {"type": "number", "minimum": 0, "maximum": 100},
                        "overall": {"type": "number", "minimum": 0, "maximum": 100},
                    },
                },
                "awareness_level": {
                    "type": "string",
                    "enum": ["problem", "solution", "product", "most"],
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
                    "maxItems": 3,
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
    normalized = token.replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    upper = normalized.upper()
    task_map = {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
        "E": "E",
        "DESIRE": "DESIRE",
        "E_AUTO": "E_auto",
        "EAUTO": "E_auto",
        "TODO_TERRENO": "TODO_TERRENO",
        "REFACTOR": "REFACTOR",
        "BUGFIX": "BUGFIX",
        "BUG_FIX": "BUGFIX",
        "UNIT_TESTS": "UNIT_TESTS",
        "UNITTESTS": "UNIT_TESTS",
        "DOCSTRINGS": "DOCSTRINGS",
        "DOCS": "DOCSTRINGS",
        "MIGRATION": "MIGRATION",
        "MIGRATE": "MIGRATION",
        "API_ENDPOINT": "API_ENDPOINT",
        "ENDPOINT": "API_ENDPOINT",
        "SQL_QUERY": "SQL_QUERY",
        "SQL": "SQL_QUERY",
        "CLI_TOOL": "CLI_TOOL",
        "CLI": "CLI_TOOL",
        "JSON_REPORT": "JSON_REPORT",
        "JSON": "JSON_REPORT",
    }
    canonical = task_map.get(upper)
    if canonical is None:
        raise KeyError(f"Unknown task: {task}")
    return canonical


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


def get_json_schema(task: str) -> Dict[str, Any] | None:
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
    "PROMPT_DESIRE",
    "PROMPT_TODO_TERRENO",
    "PROMPT_REFACTOR",
    "PROMPT_BUGFIX",
    "PROMPT_UNIT_TESTS",
    "PROMPT_DOCSTRINGS",
    "PROMPT_MIGRATION",
    "PROMPT_API_ENDPOINT",
    "PROMPT_SQL_QUERY",
    "PROMPT_CLI_TOOL",
    "PROMPT_JSON_REPORT",
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
