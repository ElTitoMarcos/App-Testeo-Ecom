import pytest

from product_research_app.prompts import registry


@pytest.mark.parametrize(
    "task",
    [
        "A",
        "B",
        "C",
        "D",
        "E",
        "E_auto",
        "DESIRE",
        "todo-terreno",
        "refactor",
        "bug_fix",
        "unit_tests",
        "docstrings",
        "migration",
        "api-endpoint",
        "sql_query",
        "cli-tool",
        "json_report",
    ],
)
def test_prompts_available(task: str) -> None:
    system_prompt = registry.get_system_prompt(task)
    assert system_prompt == registry.PROMPT_MASTER_V3_SYSTEM
    prompt = registry.get_task_prompt(task)
    assert isinstance(prompt, str)
    assert prompt.startswith("TAREA")


def test_json_only_flags() -> None:
    assert registry.is_json_only("B") is True
    assert registry.is_json_only("E_auto") is True
    assert registry.is_json_only("JSON_REPORT") is True
    for task in [
        "A",
        "C",
        "D",
        "E",
        "TODO_TERRENO",
        "REFACTOR",
        "BUGFIX",
        "UNIT_TESTS",
        "DOCSTRINGS",
        "MIGRATION",
        "API_ENDPOINT",
        "SQL_QUERY",
        "CLI_TOOL",
    ]:
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


def test_json_schema_task_json_report() -> None:
    schema = registry.get_json_schema("JSON_REPORT")
    assert schema is not None
    props = schema["schema"]["properties"]
    assert props["items"]["type"] == "array"
    item_schema = props["items"]["items"]
    required = set(item_schema["required"])
    assert required == {"id", "score", "reason"}
    score = item_schema["properties"]["score"]
    assert score["minimum"] == 0
    assert score["maximum"] == 1
    assert score["multipleOf"] == 0.01
