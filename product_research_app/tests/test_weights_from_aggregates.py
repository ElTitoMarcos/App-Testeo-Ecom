from product_research_app.services import winner_score
from product_research_app.services.winner_weights_from_aggregates import (
    calculate_weights_from_aggregates,
    PROMPT_VERSION,
)


def test_weights_from_aggregates_missing_block_returns_zeros():
    result = calculate_weights_from_aggregates(None)
    assert result["prompt_version"] == PROMPT_VERSION
    assert result["order"] == []
    assert result["notes"] == ["sin agregados: no se pueden fijar pesos sin inventar"]
    expected = {k: 0 for k in winner_score.ALLOWED_FIELDS}
    assert result["weights"] == expected


def test_weights_from_aggregates_limits_low_coverage_and_missing_metrics():
    aggregates = {
        "metrics": {
            "revenue": {
                "min": 100.0,
                "max": 1000.0,
                "p50": 400.0,
                "coverage": 0.2,
                "std": 120.0,
            }
        }
    }
    result = calculate_weights_from_aggregates(aggregates)
    weights = result["weights"]
    assert weights["revenue"] <= 15
    assert any("revenue" in note for note in result["notes"])
    missing_metrics = [k for k in winner_score.ALLOWED_FIELDS if k != "revenue"]
    for metric in missing_metrics:
        assert weights[metric] == 0
        assert any(metric in note for note in result["notes"])


def test_weights_from_aggregates_generates_prioritized_order():
    aggregates = {
        "metrics": {
            "price": {
                "min": 10,
                "max": 60,
                "p25": 20,
                "p50": 35,
                "p75": 48,
                "coverage": 0.9,
            },
            "rating": {
                "min": 3.0,
                "max": 5.0,
                "p50": 4.6,
                "std": 0.4,
                "coverage": 0.8,
            },
            "units_sold": {
                "min": 50,
                "max": 5000,
                "p50": 1500,
                "p75": 3000,
                "coverage": 0.95,
            },
            "revenue": {
                "min": 500,
                "max": 75000,
                "p50": 12000,
                "p75": 25000,
                "coverage": 0.92,
            },
            "desire": {
                "min": 1,
                "max": 5,
                "p50": 4.2,
                "coverage": 0.85,
            },
            "competition": {
                "min": 0.1,
                "max": 1.0,
                "p50": 0.85,
                "coverage": 0.9,
            },
            "oldness": {
                "min": 5,
                "max": 240,
                "p50": 40,
                "coverage": 0.88,
            },
            "awareness": {
                "min": 0,
                "max": 100,
                "p50": 55,
                "coverage": 0.7,
            },
        }
    }

    result = calculate_weights_from_aggregates(aggregates)
    weights = result["weights"]
    order = result["order"]

    # Traction metrics should dominate.
    assert weights["revenue"] > weights["price"]
    assert weights["units_sold"] >= weights["desire"]

    # High competition mean should moderate its weight below desire.
    assert weights["competition"] < weights["desire"]

    # Order must include only positive-weight metrics sorted by weight.
    assert order
    assert set(order) == {k for k, v in weights.items() if v > 0}
    sorted_weights = [weights[k] for k in order]
    assert sorted_weights == sorted(sorted_weights, reverse=True)

