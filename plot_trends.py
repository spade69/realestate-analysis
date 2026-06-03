"""5 期片区均价折线图（核心片区聚焦）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

from analyze import load_data


# 设置中文字体（macOS 上的 Songti SC）
for fname in ["Songti SC", "Hiragino Sans GB", "PingFang SC", "Heiti SC", "Arial Unicode MS"]:
    if any(f.name == fname for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = [fname]
        break
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    here = Path(__file__).parent
    csvs = sorted((here / "data").glob("*.csv"))
    df = load_data(csvs)

    periods = sorted(df["period"].unique())
    print(f"Periods: {periods}")

    # 每片区在每期的样本量
    n_pivot = (df.groupby(["area", "period"]).size()
               .unstack("period").fillna(0).astype(int))
    # 核心片区：至少在 3 个期里有 ≥5 笔成交
    core_mask = (n_pivot >= 5).sum(axis=1) >= 3
    core_areas = n_pivot[core_mask].index.tolist()
    print(f"Core areas ({len(core_areas)}): {core_areas}")

    # 各核心片区的每期平均单价
    avg_pivot = (df.groupby(["area", "period"])["unit_price_wan_sqm"]
                 .mean().unstack("period"))
    med_pivot = (df.groupby(["area", "period"])["unit_price_wan_sqm"]
                 .median().unstack("period"))

    # ----- 图 1：核心片区平均单价 5 期走势 -----
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10.colors

    for i, area in enumerate(core_areas):
        ys = [avg_pivot.loc[area, p] if not pd.isna(avg_pivot.loc[area, p]) else None
              for p in periods]
        # 用细线连接非空点
        xs_valid = [p for p, y in zip(periods, ys) if y is not None]
        ys_valid = [y for y in ys if y is not None]
        if not ys_valid:
            continue
        ax.plot(xs_valid, ys_valid, marker="o", linewidth=2,
                markersize=7, color=colors[i % len(colors)], label=area)
        # 在最后一点旁标注片区名
        if xs_valid:
            ax.annotate(area, xy=(xs_valid[-1], ys_valid[-1]),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=9, color=colors[i % len(colors)],
                        verticalalignment="center")

    ax.set_title("福田核心片区 5 期平均成交单价走势（万/m²）",
                 fontsize=14, pad=15)
    ax.set_xlabel("期次", fontsize=11)
    ax.set_ylabel("平均单价 (万/m²)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-0.3, len(periods) - 0.3)
    fig.tight_layout()
    out1 = here / "reports" / "trend_core_areas_avg.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ----- 图 2：核心片区中位单价 5 期走势 -----
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, area in enumerate(core_areas):
        ys = [med_pivot.loc[area, p] if not pd.isna(med_pivot.loc[area, p]) else None
              for p in periods]
        xs_valid = [p for p, y in zip(periods, ys) if y is not None]
        ys_valid = [y for y in ys if y is not None]
        if not ys_valid:
            continue
        ax.plot(xs_valid, ys_valid, marker="s", linewidth=2,
                markersize=7, color=colors[i % len(colors)], label=area)
        if xs_valid:
            ax.annotate(area, xy=(xs_valid[-1], ys_valid[-1]),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=9, color=colors[i % len(colors)],
                        verticalalignment="center")

    ax.set_title("福田核心片区 5 期中位成交单价走势（万/m²）",
                 fontsize=14, pad=15)
    ax.set_xlabel("期次", fontsize=11)
    ax.set_ylabel("中位单价 (万/m²)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-0.3, len(periods) - 0.3)
    fig.tight_layout()
    out2 = here / "reports" / "trend_core_areas_median.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    # ----- 图 3：每期成交量 + 大跌占比 双轴 -----
    fig, ax1 = plt.subplots(figsize=(11, 6))
    per_period = df.groupby("period").agg(
        n_total=("period", "size"),
        n_with_nr=("negotiation_rate", lambda s: s.notna().sum()),
        n_big_drop=("negotiation_rate", lambda s: (s <= -20).sum()),
    )
    per_period["big_drop_rate"] = (
        per_period["n_big_drop"] / per_period["n_with_nr"] * 100
    )

    color1 = "#2E86AB"
    color2 = "#E63946"
    ax1.bar(periods, per_period["n_total"], color=color1, alpha=0.6,
            label="成交笔数")
    ax1.set_xlabel("期次", fontsize=11)
    ax1.set_ylabel("成交笔数", fontsize=11, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(periods, per_period["big_drop_rate"], color=color2,
             marker="o", linewidth=2.5, markersize=9,
             label="大跌占比 (≤-20%)")
    ax2.set_ylabel("大跌占比 % (谈价率 ≤ -20%)", fontsize=11, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, max(per_period["big_drop_rate"]) * 1.3)

    for i, (p, row) in enumerate(per_period.iterrows()):
        ax2.annotate(f"{row['big_drop_rate']:.1f}%",
                     xy=(i, row["big_drop_rate"]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, color=color2)
        ax1.annotate(f"{int(row['n_total'])}",
                     xy=(i, row["n_total"]),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=9, color=color1)

    ax1.set_title("福田 5 期成交量 + 大跌占比", fontsize=14, pad=15)
    fig.tight_layout()
    out3 = here / "reports" / "trend_volume_bigdrop.png"
    fig.savefig(out3, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out3}")

    # ----- 图 4：同小区同户型 高可信子集 趋势（4 期相邻对比） -----
    # 取每对相邻期的 高可信子集 平均变化，画成柱状图
    from analyze import analyze_same_community_room_delta
    bars = []
    labels = []
    for e, l in zip(periods[:-1], periods[1:]):
        pairs = analyze_same_community_room_delta(df, e, l)
        thick = [p for p in pairs if p["n_earlier"] >= 2 and p["n_later"] >= 2]
        if thick:
            avg = sum(p["change_pct"] for p in thick) / len(thick)
            bars.append((avg, len(thick)))
            labels.append(f"{l}\nvs {e}")
        else:
            bars.append((0, 0))
            labels.append(f"{l}\nvs {e}")

    fig, ax = plt.subplots(figsize=(11, 6))
    avgs = [b[0] for b in bars]
    ns = [b[1] for b in bars]
    colors_bar = ["#43AA8B" if v >= 0 else "#E63946" for v in avgs]
    bars_obj = ax.bar(labels, avgs, color=colors_bar, alpha=0.8)
    for i, (v, n) in enumerate(zip(avgs, ns)):
        ax.annotate(f"{v:+.2f}%\n({n} 对)",
                    xy=(i, v),
                    xytext=(0, 5 if v >= 0 else -25),
                    textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("同小区同户型相邻期变化（高可信子集 ≥2 笔/期）",
                 fontsize=14, pad=15)
    ax.set_ylabel("均价变化 %", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out4 = here / "reports" / "trend_same_unit_change.png"
    fig.savefig(out4, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out4}")


if __name__ == "__main__":
    main()
