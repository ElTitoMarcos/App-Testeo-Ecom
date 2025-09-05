import pytest
from product_research_app.title_analyzer import analyze_titles


def test_strong_signals_and_score():
    items = [{"title": "Waterproof Magnetic Case for iPhone 15, 2 Pack, 64oz"}]
    results = analyze_titles(items)
    assert len(results) == 1
    r = results[0]
    # signals
    assert "2 pack" in r["signals"]["value"]
    assert "64oz" in r["signals"]["value"]
    assert any(c.lower() == "for iphone 15" for c in r["signals"]["compat"])
    assert "waterproof" in r["signals"]["claims"]
    # scoring should be positive despite ip_risk penalty
    assert r["titleScore"] > 0


def test_generic_product_flags_genericity():
    items = [{"title": "Premium Best Kitchen Set"}]
    r = analyze_titles(items)[0]
    assert r["signals"] == {"value": [], "claims": [], "materials": [], "compat": []}
    assert r["flags"]["genericity"] is True
    assert r["titleScore"] < 0


def test_long_title_triggers_seo_bloat():
    long_title = "Amazing " + ("quality " * 30) + "kitchen tool"
    r = analyze_titles([{"title": long_title}])[0]
    assert r["flags"]["seo_bloat"] is True


def test_ip_risk_without_license():
    r = analyze_titles([{"title": "Case for iPhone 15"}])[0]
    assert r["flags"]["ip_risk"] is True


def test_price_bucket_assignment():
    items = [
        {"title": "T1", "price": 10},
        {"title": "T2", "price": 20},
        {"title": "T3", "price": 30},
    ]
    results = analyze_titles(items)
    buckets = {r["title"]: r["summary"]["price_bucket"] for r in results}
    assert buckets["T1"] == "low"
    assert buckets["T2"] == "mid"
    assert buckets["T3"] == "high"
