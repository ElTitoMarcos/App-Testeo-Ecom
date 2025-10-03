from __future__ import annotations

from typing import Iterable, Tuple


def _ids_to_text(ids: Iterable[int]) -> str:
    return ", ".join(str(pid) for pid in ids)


def STRICT_JSONL_PROMPT(ids: Iterable[int], fields: Tuple[str, ...]) -> str:
    ids_text = _ids_to_text(ids)
    example = (
        "[{\"id\":101,\"desire\":\"Resumen del deseo\",\"desire_label\":\"High\",\"desire_magnitude\":0.88,\"competition_level\":0.32}]"
    )
    return (
        "Devuelve únicamente un ARRAY JSON. Sin comentarios, sin markdown, sin texto antes o después.\n"
        f"IDs solicitados (orden estricto): [{ids_text}]\n"
        "Mantén el mismo orden en el array. Cada objeto debe incluir al menos id, desire, desire_label, desire_magnitude.\n"
        "Añade competition_level (0-1) y price cuando puedas inferirlos. Prohibidas las explicaciones.\n"
        "Ejemplo de salida válida:\n"
        f"{example}"
    )


def MISSING_ONLY_JSONL_PROMPT(ids: Iterable[int], fields: Tuple[str, ...]) -> str:
    ids_text = _ids_to_text(ids)
    example = (
        "[{\"id\":42,\"desire\":\"Nuevo resumen\",\"desire_label\":\"Medium\",\"desire_magnitude\":0.55,\"competition_level\":0.61}]"
    )
    return (
        "Reintento: devuelve únicamente un ARRAY JSON para los IDs faltantes indicados.\n"
        "No repitas IDs ya completados ni añadas explicaciones ni comentarios.\n"
        f"IDs pendientes (mismo orden): [{ids_text}]\n"
        "Cada objeto debe incluir al menos id, desire, desire_label, desire_magnitude (0-1). Añade competition_level y price si puedes.\n"
        "Ejemplo de salida esperada:\n"
        f"{example}"
    )
