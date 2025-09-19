from __future__ import annotations

from pathlib import Path

from product_research_app import student_model


def test_build_feature_sample_basic():
    raw = {
        "title": "Amazing Widget",
        "description": "A top tier gadget",
        "price": "19.99",
        "rating": "4.7",
        "units_sold": "1500",
        "launch_date": "2023-01-15",
    }
    sample = student_model.build_feature_sample(raw)
    assert "Amazing Widget" in sample["text"]
    assert sample["price"] == 19.99
    assert sample["rating"] == 4.7
    assert sample["units_sold"] == 1500.0
    assert sample["oldness"] >= 0


def test_student_manager_without_models(tmp_path: Path):
    manager = student_model.StudentModelManager(enabled=True, model_dir=tmp_path)
    assert not manager.is_ready
    result = manager.predict(
        {
            "text": "Sample product",
            "price": 12.0,
            "rating": 4.0,
            "units_sold": 120.0,
            "oldness": 30.0,
        }
    )
    assert result is None
