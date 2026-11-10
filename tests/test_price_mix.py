import pandas as pd

from render_site import build_price_mix, price_mix_counts


def _df():
    return pd.DataFrame({
        "period": ["2026-01上", "2026-01上", "2026-01上", "2026-01下", "2026-01下"],
        "total_price_wan": [150, 250, 1200, 180, 450],
    })


def test_price_mix_counts_by_bucket_and_period():
    counts = price_mix_counts(_df(), ["2026-01上", "2026-01下"])
    # 行：<200万, 200-300万, 300-400万, 400-600万, ≥600万
    assert counts == [[1, 1], [1, 0], [0, 0], [0, 1], [1, 0]]
    assert sum(map(sum, counts)) == 5


def test_price_mix_caption_low_share_trend():
    table, caption = build_price_mix(_df(), ["2026-01上", "2026-01下"])
    # 300 万以下占比：第 1 期 2/3=67% → 第 2 期 1/2=50%
    assert "67%" in caption and "50%" in caption and "下降" in caption
    assert table.count("<tr>") == 1 + 2 + 1  # 表头 + 2 期 + 合计
