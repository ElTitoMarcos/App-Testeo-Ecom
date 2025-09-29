from __future__ import annotations

from typing import Iterable, Tuple


def _ids_to_text(ids: Iterable[int]) -> str:
    return ", ".join(str(pid) for pid in ids)


def STRICT_JSONL_PROMPT(ids: Iterable[int], fields: Tuple[str, ...]) -> str:
    ids_text = _ids_to_text(ids)
    example = "[{\"aw\":\"Medium\",\"aw_m\":55,\"d\":\"Beneficio principal\",\"d_m\":72,\"c\":\"Low\",\"c_m\":18}]"
    return (
        "Formato obligatorio: devuelve un único array JSON con un objeto por producto solicitado.\n"
        "No añadas texto extra, comentarios ni encabezados.\n"
        f"IDs solicitados (orden estricto): [{ids_text}]\n"
        "Mantén el mismo orden en el array. Cada objeto debe contener únicamente las claves aw, aw_m, d, d_m, c, c_m.\n"
        "aw y c deben ser Low|Medium|High. aw_m, d_m y c_m son enteros 0-100.\n"
        "Ejemplo de salida válida:\n"
        f"{example}"
    )


def MISSING_ONLY_JSONL_PROMPT(ids: Iterable[int], fields: Tuple[str, ...]) -> str:
    ids_text = _ids_to_text(ids)
    example = "[{\"aw\":\"High\",\"aw_m\":80,\"d\":\"Nueva respuesta\",\"d_m\":64,\"c\":\"Medium\",\"c_m\":45}]"
    return (
        "Reintento: devuelve únicamente un array JSON para los IDs faltantes indicados.\n"
        "No repitas IDs ya completados ni añadas explicaciones.\n"
        f"IDs pendientes (mismo orden): [{ids_text}]\n"
        "Cada objeto debe incluir solo aw, aw_m, d, d_m, c, c_m con el mismo formato estricto.\n"
        "Ejemplo de salida esperada:\n"
        f"{example}"
    )
