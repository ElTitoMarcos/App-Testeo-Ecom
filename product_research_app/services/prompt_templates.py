from __future__ import annotations

from typing import Iterable, Tuple


def _ids_to_text(ids: Iterable[int]) -> str:
    return ", ".join(str(pid) for pid in ids)


def STRICT_JSONL_PROMPT(ids: Iterable[int], fields: Tuple[str, ...]) -> str:
    ids_text = _ids_to_text(ids)
    field_list = ", ".join(fields)
    example = (
        "{" +
        '"product_id": 123, "desire": "alta", "desire_magnitude": 0.82, '
        '"awareness_level": "problem", "competition_level": "mid"' +
        "}"
    )
    return (
        "Formato obligatorio: devuelve exactamente una línea JSON por producto pedido (JSONL).\n"
        "No añadas texto extra, comentarios ni encabezados.\n"
        f"IDs solicitados (orden estricto): [{ids_text}]\n"
        f"Cada línea debe contener las claves: product_id, {field_list}.\n"
        "Asegúrate de que el conjunto de product_id coincida exactamente con los IDs pedidos.\n"
        "Ejemplo de UNA línea válida (usa tus propios valores):\n"
        f"{example}"
    )


def MISSING_ONLY_JSONL_PROMPT(ids: Iterable[int], fields: Tuple[str, ...]) -> str:
    ids_text = _ids_to_text(ids)
    field_list = ", ".join(fields)
    example = (
        "{" +
        '"product_id": 456, "desire": "alta", "desire_magnitude": 0.75, '
        '"awareness_level": "solution", "competition_level": "low"' +
        "}"
    )
    return (
        "Reintento: devuelve únicamente JSONL para los IDs faltantes indicados.\n"
        "No repitas IDs ya completados ni añadas explicaciones.\n"
        f"IDs pendientes (entrega uno a uno): [{ids_text}]\n"
        f"Claves obligatorias: product_id, {field_list}.\n"
        "Ejemplo de salida esperada:\n"
        f"{example}"
    )
