import numpy as np
import pandas as pd
import pytest

from analyze import (
    UNMAPPED_TIER,
    analyze_school_premium,
    attach_school,
    load_school_map,
)


def _synthetic(premium: float, n_per_cell: int = 20, seed: int = 0) -> pd.DataFrame:
    """两个片区 × 三档学区，单价 = 片区基准 × (1+premium)^名校 × 房龄折旧 × 噪声。"""
    rng = np.random.default_rng(seed)
    rows = []
    for area, base in (("A", 6.0), ("B", 9.0)):
        for tier, mult in (("名校", 1 + premium), ("普通", 1.0), ("无学位", 0.8)):
            for i in range(n_per_cell):
                year = int(rng.integers(1985, 2020))
                sqm = float(rng.integers(30, 150))
                up = base * mult * (0.99 ** (2026 - year)) * float(np.exp(rng.normal(0, 0.03)))
                rows.append({
                    "period": "2026-01上" if i % 2 else "2026-01下",
                    "area": area,
                    "community": f"{area}-{tier}-{i % 4}",
                    "year_built": year,
                    "room_type": f"{1 + i % 4}室",
                    "area_sqm": sqm,
                    "unit_price_wan_sqm": up,
                    "negotiation_rate": -5.0,
                    "deal_date": pd.Timestamp("2026-01-10"),
                    "school_tier": tier,
                })
    return pd.DataFrame(rows)


def test_regression_recovers_known_premium():
    res = analyze_school_premium(_synthetic(premium=0.25))
    terms = {t["tier"]: t for t in res["regression"]["terms"]}
    assert terms["名校"]["premium_pct"] == pytest.approx(25, abs=3)
    assert terms["名校"]["ci_low_pct"] < 25 < terms["名校"]["ci_high_pct"]
    assert terms["无学位"]["premium_pct"] == pytest.approx(-20, abs=3)
    assert res["regression"]["age_pct_per_year"] == pytest.approx(-1, abs=0.3)


def test_same_area_comparison():
    res = analyze_school_premium(_synthetic(premium=0.25))
    assert {r["area"] for r in res["same_area"]} == {"A", "B"}
    assert res["same_area_median_pct"] > 10


def test_no_mapping_returns_empty_result():
    df = _synthetic(premium=0.25).assign(school_tier=UNMAPPED_TIER)
    res = analyze_school_premium(df)
    assert res["n_mapped"] == 0
    assert res["regression"] is None
    assert res["by_tier"] == []


def test_load_and_attach(tmp_path):
    p = tmp_path / "map.csv"
    p.write_text(
        "area,community,n_deals,primary_school,middle_school,tier,source,note\n"
        "百花,长安花园,10,百花小学,红岭中学,名校,src,\n"
        "梅林,未知小区,3,,,,,\n",
        encoding="utf-8",
    )
    m = load_school_map(p)
    assert list(m["community"]) == ["长安花园"]
    df = pd.DataFrame({"community": ["长安花园", "未知小区"]})
    out = attach_school(df, m)
    assert list(out["school_tier"]) == ["名校", UNMAPPED_TIER]
    assert out.loc[0, "school_primary"] == "百花小学"


def test_load_rejects_unknown_tier(tmp_path):
    p = tmp_path / "map.csv"
    p.write_text(
        "area,community,n_deals,primary_school,middle_school,tier,source,note\n"
        "百花,长安花园,10,百花小学,,顶级,src,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown tier"):
        load_school_map(p)
