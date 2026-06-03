"""导出 ECharts 用的 JSON 数据。"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from analyze import load_data, analyze_same_community_room_delta


def main() -> None:
    here = Path(__file__).parent
    csvs = sorted((here / "data").glob("*.csv"))
    df = load_data(csvs)

    periods = sorted(df["period"].unique())

    n_pivot = (df.groupby(["area", "period"]).size()
               .unstack("period").fillna(0).astype(int))
    core_mask = (n_pivot >= 5).sum(axis=1) >= 3
    core_areas = n_pivot[core_mask].index.tolist()

    avg_pivot = (df.groupby(["area", "period"])["unit_price_wan_sqm"]
                 .mean().unstack("period"))
    med_pivot = (df.groupby(["area", "period"])["unit_price_wan_sqm"]
                 .median().unstack("period"))

    def series(pivot):
        return [
            {
                "name": area,
                "data": [None if pd.isna(pivot.loc[area, p]) else round(float(pivot.loc[area, p]), 2)
                         for p in periods],
            }
            for area in core_areas
        ]

    # 成交量 + 大跌占比
    per_period = df.groupby("period").agg(
        n_total=("period", "size"),
        n_with_nr=("negotiation_rate", lambda s: s.notna().sum()),
        n_big_drop=("negotiation_rate", lambda s: (s <= -20).sum()),
    )
    per_period["big_drop_rate"] = (
        per_period["n_big_drop"] / per_period["n_with_nr"] * 100
    ).round(2)

    # 同小区同户型相邻期变化（高可信子集）
    same_unit_bars = []
    for e, l in zip(periods[:-1], periods[1:]):
        pairs = analyze_same_community_room_delta(df, e, l)
        thick = [p for p in pairs if p["n_earlier"] >= 2 and p["n_later"] >= 2]
        if thick:
            avg = sum(p["change_pct"] for p in thick) / len(thick)
            same_unit_bars.append({
                "label": f"{l} vs {e}",
                "value": round(avg, 2),
                "n": len(thick),
            })
        else:
            same_unit_bars.append({"label": f"{l} vs {e}", "value": 0, "n": 0})

    out = {
        "periods": periods,
        "core_areas": core_areas,
        "avg_series": series(avg_pivot),
        "median_series": series(med_pivot),
        "volume": [int(per_period.loc[p, "n_total"]) for p in periods],
        "big_drop_rate": [float(per_period.loc[p, "big_drop_rate"]) for p in periods],
        "same_unit": same_unit_bars,
    }

    target = here / "docs" / "chart_data.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {target} ({target.stat().st_size} bytes)")
    print(json.dumps({k: v if not isinstance(v, list) else f"[{len(v)} items]" for k, v in out.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
