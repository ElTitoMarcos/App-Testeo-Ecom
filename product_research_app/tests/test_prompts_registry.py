import pytest

from product_research_app.prompts import registry


@pytest.mark.parametrize("task", ["A", "B", "C", "D", "E", "E_auto", "DESIRE"])
def test_prompts_available(task: str) -> None:
    system_prompt = registry.get_system_prompt(task)
    assert system_prompt == registry.PROMPT_MASTER_V4_SYSTEM
    prompt = registry.get_task_prompt(task)
    assert isinstance(prompt, str)
    assert prompt.startswith("TAREA")


def test_json_only_flags() -> None:
    assert registry.is_json_only("B") is True
    assert registry.is_json_only("E_auto") is True
    assert registry.is_json_only("DESIRE") is True
    for task in ["A", "C", "D", "E"]:
        assert registry.is_json_only(task) is False


def test_json_schema_task_b() -> None:
    schema = registry.get_json_schema("B")
    assert schema is not None
    weights = schema["schema"]["properties"]["weights"]
    metrics = weights["required"]
    assert len(metrics) == 8
    for metric in metrics:
        bounds = weights["properties"][metric]
        assert bounds["minimum"] == 0
        assert bounds["maximum"] == 100
    order = schema["schema"]["properties"]["order"]
    assert order["minItems"] == 8
    assert order["uniqueItems"] is True


def test_json_schema_task_e_auto() -> None:
    schema = registry.get_json_schema("E_auto")
    assert schema is not None
    item_schema = schema["schema"]["properties"]["items"]["items"]
    required = set(item_schema["required"])
    for key in {"id", "status", "score", "confidence", "summary", "reason", "next_step", "signals"}:
        assert key in required
    status_enum = item_schema["properties"]["status"]["enum"]
    assert {"aprobado", "revisar", "descartar"} == set(status_enum)
    signals = item_schema["properties"]["signals"]
    assert signals["type"] == "array"


def test_json_schema_task_desire() -> None:
    schema = registry.get_json_schema("DESIRE")
    assert schema is not None
    magnitude = schema["schema"]["properties"]["desire_magnitude"]
    for key in ["scope", "urgency", "staying_power", "overall"]:
        bounds = magnitude["properties"][key]
        assert bounds["minimum"] == 0
        assert bounds["maximum"] == 100
    seasonality = schema["schema"]["properties"]["seasonality_hint"]
    window_enum = seasonality["properties"]["window"]["enum"]
    assert set(window_enum) == {"jan", "feb", "mar_apr", "may", "jun", "jul_aug", "sep", "oct", "nov", "dec"}


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("desire", "DESIRE"),
        ("DeSiRe", "DESIRE"),
        ("de-sire", "DESIRE"),
        ("de_sire", "DESIRE"),
    ],
)
def test_normalize_task_desire(alias: str, expected: str) -> None:
    assert registry.normalize_task(alias) == expected
