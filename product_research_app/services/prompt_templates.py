from __future__ import annotations

from typing import Iterable, Tuple


def _ids_to_text(ids: Iterable[int]) -> str:
    return ", ".join(str(pid) for pid in ids)


def STRICT_JSONL_PROMPT(ids: Iterable[int], fields: Tuple[str, ...]) -> str:
    ids_text = _ids_to_text(ids)
    example = (
        "{\"items\":[{\"id\":101,\"desire\":0.78,\"desire_reason\":\"Alta búsqueda estacional y buenos reviews.\",\"competition\":0.41,"
        "\"competition_level\":\"medium\",\"revenue\":12450.0,\"units_sold\":380,\"price\":32.99,\"oldness\":0.18,\"rating\":4.4}]}"
    )
    return (
        "TAREA\n"
        "Eres un motor de transformación de datos. Dado un listado de productos, debes devolver SOLO un objeto JSON válido que cumpla este esquema lógico:\n\n"
        "- Objeto raíz con clave \"items\": array.\n"
        "- Cada elemento de \"items\" es un objeto con TODAS estas claves (todas obligatorias):\n"
        "  - id (integer)\n"
        "  - desire (number, 0..1, con 2 decimales)\n"
        "  - desire_reason (string, breve y específica)\n"
        "  - competition (number, 0..1, con 2 decimales)\n"
        "  - competition_level (string, uno de: \"low\" | \"medium\" | \"high\")\n"
        "  - revenue (number, >=0)\n"
        "  - units_sold (number, >=0)\n"
        "  - price (number, >=0)\n"
        "  - oldness (number, 0..1, con 2 decimales)\n"
        "  - rating (number, 0..5, con 1 o 2 decimales)\n\n"
        "REGLAS\n"
        "- La longitud de \"items\" debe coincidir EXACTAMENTE con el número de productos de entrada.\n"
        "- Conserva los valores de \"id\" que vengan en la entrada.\n"
        "- Estima valores faltantes de forma razonable usando las señales disponibles (títulos, categoría, precio, reviews, volumen de búsqueda, etc.). NO uses null.\n"
        "- Redondea: números en [0..1] a 2 decimales; rating a 1–2 decimales.\n"
        "- Mapea competition_level desde competition:\n"
        "  - <= 0.33 → \"low\"\n"
        "  - > 0.33 y <= 0.66 → \"medium\"\n"
        "  - > 0.66 → \"high\"\n"
        "- SALIDA: imprime ÚNICAMENTE el objeto JSON final. Sin texto antes/después, sin markdown, sin comentarios.\n\n"
        "EJEMPLO DE SALIDA VÁLIDA\n"
        f"{example}\n\n"
        f"IDs de referencia (mantén el orden): [{ids_text}]\n"
        "PRODUCTS:\n"
    )


def MISSING_ONLY_JSONL_PROMPT(ids: Iterable[int], fields: Tuple[str, ...]) -> str:
    ids_text = _ids_to_text(ids)
    return (
        "REINTENTO\n"
        "Debes repetir la misma estructura JSON indicada arriba, pero SOLO para los IDs pendientes en el orden indicado.\n"
        "Nada de texto adicional, markdown ni comentarios.\n"
        f"IDs pendientes (mismo orden): [{ids_text}]\n"
        "PRODUCTS:\n"
    )
