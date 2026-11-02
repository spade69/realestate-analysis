import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from build_school_map import find_hits, looks_no_degree, refresh_rows, tokens  # noqa: E402


def _index(lists):
    idx = {}
    for level, school, raw in lists:
        for t in tokens(raw):
            idx.setdefault(t, []).append((level, school, "x"))
    return idx


def _schools(hits):
    return sorted({s for _, s, _ in hits})


def test_paren_alias_is_a_token():
    assert "南光名仕苑" in tokens("嘉福花园、南名仕苑(南光名仕苑)、银泰苑")
    assert "南名仕苑" in tokens("嘉福花园、南名仕苑(南光名仕苑)、银泰苑")


def test_brand_prefix_fallback_and_alias():
    idx = _index([("小学", "南华小学", "赤尾村、御景华城、汇港名苑"),
                  ("小学", "下沙小学", "金域蓝湾、下沙花好园、滨河时代")])
    assert _schools(find_hits("京基御景华城", idx)) == ["南华小学"]
    assert _schools(find_hits("万科金域蓝湾三期", idx)) == ["下沙小学"]
    assert _schools(find_hits("花好园", idx)) == ["下沙小学"]


def test_brand_prefix_only_used_when_no_primary_hit():
    idx = _index([("小学", "外国语学校福保小学部", "南光紫荆苑、福保花园"),
                  ("小学", "景鹏小学", "紫荆苑、香蜜三村")])
    assert _schools(find_hits("南光紫荆苑", idx)) == ["外国语学校福保小学部"]


def test_no_degree_rule():
    small = pd.DataFrame({"room_type": ["1室"] * 5, "area_sqm": [35.0] * 5})
    family = pd.DataFrame({"room_type": ["3室"] * 5, "area_sqm": [90.0] * 5})
    assert looks_no_degree("赛格公寓", small)
    assert not looks_no_degree("赛格公寓", family)
    assert not looks_no_degree("梅林一村", small)  # 名字不像商住
    assert not looks_no_degree("某某公寓", small.head(1))  # 样本太少


def test_refresh_rows_recounts_and_keeps_manual():
    market = pd.DataFrame({"community": ["A", "A", "B"], "area": ["x", "x", "y"]})
    rows = [
        {"area": "x", "community": "A", "n_deals": "9", "primary_school": "S",
         "middle_school": "", "tier": "普通", "source": "auto", "note": ""},
        {"area": "z", "community": "Gone", "n_deals": "1", "primary_school": "",
         "middle_school": "", "tier": "", "source": "", "note": ""},
        {"area": "z", "community": "Manual", "n_deals": "1", "primary_school": "T",
         "middle_school": "", "tier": "名校", "source": "人工确认", "note": ""},
    ]
    out = {r["community"]: r for r in refresh_rows(rows, market)}
    assert out["A"]["n_deals"] == "2" and out["A"]["primary_school"] == "S"
    assert out["B"]["n_deals"] == "1" and out["B"]["area"] == "y"
    assert "Gone" not in out and "Manual" in out
