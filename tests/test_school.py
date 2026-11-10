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


def test_analyze_by_school_thresholds():
    from analyze import analyze_by_school

    def deal(c, school, rt, sqm, total, period, neg=-10.0):
        return {"community": c, "school_primary": school, "school_note": "",
                "room_type": rt, "area_sqm": sqm, "total_price_wan": total,
                "unit_price_wan_sqm": total / sqm, "period": period,
                "negotiation_rate": neg, "deal_date": pd.Timestamp("2026-01-10")}

    df = pd.DataFrame([
        deal("甲", "百花小学", "1室", 30, 200, "2026-01上"),
        deal("甲", "百花小学", "-", 60, 250, "2026-01上"),
        deal("乙", "百花小学", "2室", 60, 400, "2026-02下", neg=-4.0),
        deal("丙", "福民小学/福田小学", "3室", 90, 500, "2026-02下"),
        deal("丁", "", "2室", 60, 100, "2026-02下"),
    ])
    by = {s["school"]: s for s in analyze_by_school(df)}
    assert set(by) == {"百花小学", "福民小学", "福田小学"}
    assert by["百花小学"]["is_top"] and not by["福民小学"]["is_top"]
    assert by["百花小学"]["cheapest"]["total"] == 200
    assert by["百花小学"]["cheapest_2r"]["community"] == "乙"
    assert by["福田小学"]["n"] == 1
    assert list(by)[0] == "百花小学"
    # 流通性：2 期 = 1 个月
    assert by["百花小学"]["per_month"] == pytest.approx(3.0)
    assert by["百花小学"]["n_active_periods"] == 2 and by["百花小学"]["active_ratio"] == 1.0
    assert by["百花小学"]["avg_neg_rate"] == pytest.approx(-8.0)
    assert by["福田小学"]["n_active_periods"] == 1 and by["福田小学"]["active_ratio"] == 0.5


def test_analyze_low_price_groups_and_threshold():
    from analyze import analyze_low_price

    def deal(c, area, total, tier, school, period="2026-01上", rt="1室"):
        return {"community": c, "area": area, "total_price_wan": total, "area_sqm": 30.0,
                "unit_price_wan_sqm": total / 30, "negotiation_rate": -10.0, "room_type": rt,
                "period": period, "school_tier": tier, "school_primary": school}

    rows = [deal("甲", "A", 100, "名校", "百花小学", p) for p in ("2026-01上", "2026-01下")] * 3  # 甲 6 笔
    rows += [deal("乙", "A", 150, "普通", "福民小学"), deal("丙", "B", 190, "未标注", "")]
    rows += [deal("丁", "B", 450, "普通", "福民小学")]  # ≥200 万，不计入
    r = analyze_low_price(pd.DataFrame(rows), max_total=200, min_deals=5)
    assert r["n"] == 8 and r["n_all"] == 9
    areas = {a["area"]: a for a in r["areas"]}
    assert areas["A"]["n"] == 7 and areas["A"]["n_top"] == 6 and areas["B"]["n"] == 1
    assert areas["A"]["per_month"] == 7.0  # 2 期 = 1 个月
    assert [c["community"] for c in r["communities"]] == ["甲"]
    assert r["communities"][0]["n_active_periods"] == 2
    assert r["n_hidden_communities"] == 2 and r["n_hidden_deals"] == 2
