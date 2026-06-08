"""全自动渲染 docs/index.html 和 docs/chart_data.json。

工作流：CSV → 计算 → 自动生成 KPI/表格/解读 → 模板替换 → 写出。
"""

from __future__ import annotations

import html as _html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def esc(s: object) -> str:
    """HTML escape for CSV-derived strings (community/area names)."""
    return _html.escape(str(s), quote=True)

from analyze import (
    load_data,
    filter_outliers,
    analyze_by_total_price,
    analyze_by_area,
    analyze_same_community_room_delta,
    classify_area_delta,
    predict_next_period,
)


def next_period_label(periods: list[str]) -> str:
    """从最后一期推下一期 label，例如 '2026-05上' → '2026-05下'。"""
    m = re.match(r"(\d{4})-(\d{2})([上下])", periods[-1])
    if not m:
        return "下期"
    year, mon, half = int(m.group(1)), int(m.group(2)), m.group(3)
    if half == "上":
        return f"{year}-{mon:02d}下"
    nm = mon + 1
    if nm > 12:
        return f"{year + 1}-01上"
    return f"{year}-{nm:02d}上"


# ── 工具 ──────────────────────────────────────────────────────────


def period_short(p: str) -> str:
    """'2026-03上' → '3 月上'。"""
    m = re.match(r"\d{4}-(\d{2})([上下])", p)
    if not m:
        return p
    return f"{int(m.group(1))} 月{m.group(2)}"


def period_compact(p: str) -> str:
    """'2026-03上' → '3上'。"""
    m = re.match(r"\d{4}-(\d{2})([上下])", p)
    if not m:
        return p
    return f"{int(m.group(1))}{m.group(2)}"


def signed(v: float, decimals: int = 2) -> str:
    return f"{v:+.{decimals}f}"


def fmt_pct(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}%"


# ── KPI 卡片 ──────────────────────────────────────────────────────


def render_kpi(label: str, value: str, period: str, status: str = "") -> str:
    cls = f"kpi-value {status}".strip()
    return (
        f'    <div class="kpi">\n'
        f'      <div class="kpi-label">{label}</div>\n'
        f'      <div class="{cls}">{value}</div>\n'
        f'      <div class="kpi-period">{period}</div>\n'
        f'    </div>'
    )


def build_kpi_grid(df: pd.DataFrame, periods: list[str]) -> str:
    """5 张 KPI 卡。"""
    total = len(df)
    per_period_n = df.groupby("period").size()

    peak_period = per_period_n.idxmax()
    peak_n = int(per_period_n.max())
    latest_period = periods[-1]
    latest_n = int(per_period_n[latest_period])
    change_pct = (latest_n - peak_n) / peak_n * 100

    # 历期最大跌幅
    nr = df.dropna(subset=["negotiation_rate"]).sort_values("negotiation_rate")
    if len(nr):
        worst = nr.iloc[0]
        max_drop_val = float(worst["negotiation_rate"])
        max_drop_label = f'{esc(worst["community"])} {period_short(worst["period"])}'
    else:
        max_drop_val = 0
        max_drop_label = "—"

    # ≤200 万 大跌占比
    tp = analyze_by_total_price(df)
    bucket_200 = next((b for b in tp["buckets"] if b["bucket"] == "≤200万"), None)
    bucket_200_rate = bucket_200["big_drop_rate"] * 100 if bucket_200 and bucket_200["big_drop_rate"] is not None else 0
    other_rates = [
        b["big_drop_rate"] * 100
        for b in tp["buckets"]
        if b["bucket"] != "≤200万" and b["big_drop_rate"] is not None
    ]
    other_min = min(other_rates) if other_rates else 0
    other_max = max(other_rates) if other_rates else 0

    cards = [
        render_kpi("总成交", str(total), f"{len(periods)} 期合计"),
        render_kpi(
            f"{period_short(peak_period)} 笔数",
            str(peak_n),
            f"{len(periods)} 期高峰",
            status="up",
        ),
        render_kpi(
            f"{period_short(latest_period)} 笔数",
            str(latest_n),
            f"{'回落' if change_pct < 0 else '增长'} {abs(change_pct):.0f}%" if peak_period != latest_period else "最新一期",
            status="down" if change_pct < 0 else "up",
        ),
        render_kpi(
            "历期最大跌幅",
            f"{max_drop_val:.0f}%",
            max_drop_label,
            status="down",
        ),
        render_kpi(
            "≤200 万大跌占比",
            fmt_pct(bucket_200_rate, 1),
            f"vs 其他段 {other_min:.0f}-{other_max:.0f}%",
            status="warn",
        ),
    ]
    return "\n".join(cards)


# ── Volume caption ────────────────────────────────────────────────


def build_volume_caption(df: pd.DataFrame, periods: list[str]) -> str:
    """在量近高位的期数里挑大跌占比最低的那一期作为"理性期"。"""
    rows = []
    for p in periods:
        sub = df[df["period"] == p]
        nr = sub["negotiation_rate"].dropna()
        big_drop_rate = (nr <= -20).sum() / len(nr) * 100 if len(nr) else 0
        rows.append({"period": p, "n": len(sub), "big_drop_rate": big_drop_rate})
    max_n = max(r["n"] for r in rows)
    # 在量 ≥ 80% 高峰的子集里找大跌占比最低的
    high_vol = [r for r in rows if r["n"] >= max_n * 0.8]
    best = min(high_vol, key=lambda r: r["big_drop_rate"])
    min_drop = min(r["big_drop_rate"] for r in rows)
    # 必须确实是大跌最低的那一期（或接近最低）
    if best["big_drop_rate"] <= min_drop * 1.1:
        return (
            f'{period_short(best["period"])}成交活跃（{best["n"]} 笔）同时大跌占比最低'
            f'（{best["big_drop_rate"]:.1f}%），是这 {len(periods)} 期最"理性"的一期。'
        )
    peak = max(rows, key=lambda r: r["n"])
    return f'{period_short(peak["period"])}成交量最高（{peak["n"]} 笔），共 {len(periods)} 期对比。'


# ── 关键观察 (按片区) ────────────────────────────────────────────────


def categorize_area_trend(
    values: list,
    periods: list[str],
    counts: list[int] | None = None,
    min_n: int = 5,
) -> tuple[str, str | None, str]:
    """返回 (走势字符串, 警告 tag (或 None), 解读句子)。

    counts: 各期对应的样本量。低于 min_n 的视为 None（不可信）。
    """
    if counts is None:
        counts = [min_n] * len(values)
    paired = [
        (p, v) for p, v, c in zip(periods, values, counts)
        if v is not None and not pd.isna(v) and c >= min_n
    ]
    if len(paired) < 2:
        return ("—", None, "样本不足（≥5 笔的期数 < 2）")

    valid_vals = [v for _, v in paired]
    first_v, last_v = valid_vals[0], valid_vals[-1]
    max_v, min_v = max(valid_vals), min(valid_vals)
    mean = sum(valid_vals) / len(valid_vals)
    delta = (last_v - first_v) / first_v * 100 if first_v else 0
    range_pct = (max_v - min_v) / mean * 100 if mean else 0

    # 走势字符串：用 → 连接显著变化的点（变化超过 4%）
    arrows = [valid_vals[0]]
    for v in valid_vals[1:]:
        if abs(v - arrows[-1]) / arrows[-1] * 100 >= 4:
            arrows.append(v)
    trend_str = " → ".join(f"{v:.2f}" for v in arrows[-4:])  # 最多显示最近 4 个拐点

    # 规则 1：品种偏移（末期 vs 早期上涨 ≥ 25% + 高端价位 ≥ 8万 + 末期是峰值）
    # 仅在 absolute price >= 8万 (中高端价位) 时才提"豪宅扎堆"
    if last_v / first_v >= 1.25 and last_v == max_v and last_v >= 8:
        return (
            trend_str,
            "品种偏移",
            f"末期 {last_v:.2f} 比早期 {first_v:.2f} 高 {(last_v-first_v)/first_v*100:.0f}%，疑似豪宅扎堆推高均价",
        )

    # 规则 2：低位稳定
    if mean < 4.5 and range_pct < 18:
        return (trend_str, None, f"低位稳定 {min_v:.1f}-{max_v:.1f}，全程便宜也稳")

    # 规则 3：窄幅震荡
    if range_pct < 12:
        return (trend_str, None, f"窄幅震荡 {min_v:.1f}-{max_v:.1f}（稳）")

    # 规则 4：中段峰/谷
    peak_idx = valid_vals.index(max_v)
    trough_idx = valid_vals.index(min_v)
    if 0 < peak_idx < len(valid_vals) - 1:
        peak_period = paired[peak_idx][0]
        return (trend_str, None, f"{period_short(peak_period)}高点后回调")
    if 0 < trough_idx < len(valid_vals) - 1:
        trough_period = paired[trough_idx][0]
        return (trend_str, None, f"{period_short(trough_period)}小坑后回升")

    # 规则 5：持续上行/下行
    if delta >= 5:
        return (trend_str, None, f"持续上行 {signed(delta, 1)}%")
    if delta <= -5:
        return (trend_str, None, f"持续下行 {signed(delta, 1)}%")

    return (trend_str, None, "横盘震荡")


def build_area_observations(df: pd.DataFrame, periods: list[str]) -> str:
    """13 个核心片区每个一行。按警告/变化幅度排序。"""
    n_pivot = df.groupby(["area", "period"]).size().unstack("period").fillna(0).astype(int)
    core_mask = (n_pivot >= 5).sum(axis=1) >= 3
    core_areas = n_pivot[core_mask].index.tolist()

    avg_pivot = df.groupby(["area", "period"])["unit_price_wan_sqm"].mean().unstack("period")

    rows = []
    for area in core_areas:
        values = [
            None if pd.isna(avg_pivot.loc[area, p]) else float(avg_pivot.loc[area, p])
            for p in periods
        ]
        counts = [int(n_pivot.loc[area].get(p, 0)) for p in periods]
        trend, warn_tag, interp = categorize_area_trend(values, periods, counts)
        valid = [v for v in values if v is not None]
        if not valid:
            continue
        delta = (valid[-1] - valid[0]) / valid[0] * 100 if valid[0] else 0
        rows.append({
            "area": area,
            "trend": trend,
            "warn": warn_tag,
            "interp": interp,
            "abs_delta": abs(delta),
            "is_warn": warn_tag is not None,
        })

    # 排序：警告优先，再按绝对变化降序
    rows.sort(key=lambda r: (not r["is_warn"], -r["abs_delta"]))

    html = []
    for r in rows:
        cls = "num up" if "品种偏移" in (r["warn"] or "") else ("num dim" if "稳" in r["interp"] else "num")
        area_esc = esc(r["area"])
        if r["is_warn"]:
            area_cell = f'<td><strong>{area_esc}</strong></td>'
            tag_html = f'<span class="tag tag-warn">{esc(r["warn"])}</span>'
            interp_cell = f'<td>{tag_html}{r["interp"]}，<strong>不是真涨</strong></td>'
        else:
            area_cell = f'<td>{area_esc}</td>'
            interp_cell = f'<td>{r["interp"]}</td>'
        html.append(
            f'        <tr>\n'
            f'          {area_cell}\n'
            f'          <td class="{cls}">{r["trend"]}</td>\n'
            f'          {interp_cell}\n'
            f'        </tr>'
        )
    return "\n".join(html)


# ── 同小区同户型 ──────────────────────────────────────────────────


def build_same_unit_rows(df: pd.DataFrame, periods: list[str]) -> tuple[str, str]:
    """返回 (table rows HTML, conclusion p HTML)。"""
    pair_stats = []
    for e, l in zip(periods[:-1], periods[1:]):
        pairs = analyze_same_community_room_delta(df, e, l)
        thick = [p for p in pairs if p["n_earlier"] >= 2 and p["n_later"] >= 2]
        if thick:
            avg = sum(p["change_pct"] for p in thick) / len(thick)
            pair_stats.append({
                "label": f"{period_compact(l)} vs {period_compact(e)}",
                "n": len(thick),
                "avg": avg,
            })
        elif pairs:
            # 退一步：用所有 pair 的平均
            avg = sum(p["change_pct"] for p in pairs) / len(pairs)
            pair_stats.append({
                "label": f"{period_compact(l)} vs {period_compact(e)}",
                "n": len(pairs),
                "avg": avg,
            })

    # 空数据保护：少于 2 期 或 所有相邻对都无样本时
    if not pair_stats:
        return (
            '        <tr><td colspan="4" class="dim">暂无相邻期对比数据。</td></tr>',
            "样本不足，暂不展开真实变化分析。",
        )

    most_reliable_idx = max(range(len(pair_stats)), key=lambda i: pair_stats[i]["n"])

    html_rows = []
    for i, s in enumerate(pair_stats):
        avg = s["avg"]
        cls_avg = "num down" if avg < -0.5 else ("num up" if avg > 0.5 else "num")
        # 解读
        if s["n"] < 2:
            note = '<td class="dim">样本太少，噪音</td>'
        elif avg < -3:
            note = '<td>明显回调</td>'
        elif avg < -1:
            note = '<td>小幅回调</td>'
        elif avg > 3:
            note = '<td>明显反弹</td>'
        elif avg > 1:
            note = '<td>反弹微涨</td>'
        else:
            note = '<td>基本持平</td>'

        if i == most_reliable_idx and s["n"] >= 3:
            note = '<td><strong>样本最厚，最可信</strong></td>'
            row_cls = ' class="row-highlight"'
            avg_str = f'<strong>{signed(avg)}%</strong>'
            n_str = f'<strong>{s["n"]}</strong>'
            label_html = f'<strong>{s["label"]}</strong>'
        else:
            row_cls = ""
            avg_str = f'{signed(avg)}%'
            n_str = str(s["n"])
            label_html = s["label"]

        html_rows.append(
            f'        <tr{row_cls}><td>{label_html}</td><td class="num">{n_str}</td>'
            f'<td class="{cls_avg}">{avg_str}</td>{note}</tr>'
        )

    # Conclusion: 最厚一对的 |avg|
    max_abs = max(abs(s["avg"]) for s in pair_stats) if pair_stats else 0
    if max_abs < 4:
        conclusion = f'真实市场节奏：<strong>在 ±{max_abs:.0f}% 范围内震荡</strong>，没有持续单边方向。'
    else:
        # 看最近一对
        latest = pair_stats[-1]
        direction = "回调" if latest["avg"] < 0 else "反弹"
        conclusion = (
            f'真实市场节奏：相邻期变化范围 ±{max_abs:.1f}%，'
            f'最近一期（{latest["label"]}）{direction} <strong>{signed(latest["avg"])}%</strong>。'
        )
    return "\n".join(html_rows), conclusion


# ── 假设检验 ─────────────────────────────────────────────────────


def build_hypothesis(df: pd.DataFrame) -> tuple[str, str]:
    tp = analyze_by_total_price(df)
    buckets = tp["buckets"]

    rows = []
    for b in buckets:
        is_top = b["bucket"] == "≤200万"
        n = b["n"]
        avg_nr = b["avg_neg_rate"]
        rate = b["big_drop_rate"] * 100 if b["big_drop_rate"] is not None else 0
        avg_nr_str = f"{avg_nr:.2f}%" if avg_nr is not None else "—"
        rate_str = f"{rate:.1f}%"

        if is_top:
            rows.append(
                f'        <tr class="row-highlight">\n'
                f'          <td><strong>{b["bucket"]}</strong></td>\n'
                f'          <td class="num">{n}</td>\n'
                f'          <td class="num down">{avg_nr_str}</td>\n'
                f'          <td class="num warn"><strong>{rate_str}</strong></td>\n'
                f'        </tr>'
            )
        else:
            rows.append(
                f'        <tr><td>{b["bucket"]}</td><td class="num">{n}</td>'
                f'<td class="num">{avg_nr_str}</td><td class="num">{rate_str}</td></tr>'
            )

    cor = tp["correlation"]
    pr, pp = cor["pearson_r"], cor["pearson_p"]
    sr, sp = cor["spearman_r"], cor["spearman_p"]

    pearson_sig = pr is not None and pp is not None and pp < 0.05
    spearman_sig = sr is not None and sp is not None and sp < 0.05

    # 桶 200 万 vs 其他段范围
    other_rates = [b["big_drop_rate"] * 100 for b in buckets if b["bucket"] != "≤200万" and b["big_drop_rate"] is not None]
    bucket_200 = next((b for b in buckets if b["bucket"] == "≤200万"), None)
    bucket_200_rate = bucket_200["big_drop_rate"] * 100 if bucket_200 and bucket_200["big_drop_rate"] is not None else 0

    other_str = f"{min(other_rates):.0f}-{max(other_rates):.0f}%" if other_rates else "—"

    conclusion_lines = [
        f'<p>',
        f'  Pearson r = <strong>{pr:.3f}</strong> (p={pp:.2g}) <span class="{"up" if pearson_sig else "dim"}">线性相关{"显著" if pearson_sig else "不显著"}</span>；',
        f'  Spearman ρ = <strong class="{"up" if spearman_sig else "dim"}">{sr:.3f}</strong> (p={sp:.2g}) <span class="{"up" if spearman_sig else "dim"}">秩相关{"显著" if spearman_sig else "不显著"}</span>。',
        f'</p>',
    ]

    if pearson_sig and pr > 0:
        meaning = f'<strong>支持假设</strong>：总价越低，砍价确实越深 (Pearson r={pr:.3f}, p={pp:.2g})。'
    elif spearman_sig and bucket_200_rate > max(other_rates or [0]):
        meaning = (
            f'<strong>≤200 万 段砍价幅度依然最狠（{bucket_200_rate:.1f}% vs 其他段 {other_str}）</strong>，'
            '但桶间不是严格线性递增——大概率因为高总价段也出现零星大砍价。'
        )
    elif spearman_sig:
        meaning = f'秩相关显著但 ≤200 万 段并非全段最高，结论需谨慎。'
    else:
        meaning = '<strong>未发现显著相关</strong>，假设暂不成立（样本可能仍不足）。'

    conclusion_lines.append(f'<p>意思是：{meaning}</p>')

    return "\n".join(rows), "\n    ".join(conclusion_lines)


# ── 真信号摘要 ───────────────────────────────────────────────────


def build_signal_blocks(df: pd.DataFrame, periods: list[str]) -> str:
    by_area = analyze_by_area(df)
    n_by_ap = {(r["area"], r["period"]): r["n"] for r in by_area["per_period"]}

    blocks = []
    for (earlier, later), delta_rows in by_area["deltas"].items():
        if earlier not in periods or later not in periods:
            continue
        signals_down = []
        signals_up = []
        for r in delta_rows:
            n_e = n_by_ap.get((r["area"], earlier), 0)
            n_l = n_by_ap.get((r["area"], later), 0)
            label, reason = classify_area_delta(
                r["avg_unit_price_change_pct"],
                r["median_unit_price_change_pct"],
                n_e, n_l,
            )
            avg = r["avg_unit_price_change_pct"]
            if label.startswith("📉 跌"):
                strength = "强" if "强" in label else ""
                signals_down.append({
                    "area": r["area"], "avg": avg, "strength": strength,
                })
            elif label.startswith("📈 涨"):
                signals_up.append({
                    "area": r["area"], "avg": avg,
                })

        signals_down.sort(key=lambda x: x["avg"])  # 跌幅大的在前
        signals_up.sort(key=lambda x: -x["avg"])  # 涨幅大的在前

        if not signals_down and not signals_up:
            continue

        tags = []
        for s in signals_down:
            suffix = f" {s['strength']}" if s['strength'] else ""
            tags.append(f'      <span class="tag tag-down">{esc(s["area"])} {signed(s["avg"])}%{suffix}</span>')
        for s in signals_up:
            tags.append(f'      <span class="tag tag-up">{esc(s["area"])} {signed(s["avg"])}%</span>')

        block = (
            f'  <div class="signal-block">\n'
            f'    <h3>{period_short(later)} vs {period_short(earlier)}</h3>\n'
            f'    <p>\n'
            + "\n".join(tags) + "\n"
            f'    </p>\n'
            f'  </div>'
        )
        blocks.append(block)

    return "\n\n".join(blocks) if blocks else '  <p class="dim">暂无真信号。</p>'


# ── 期次详情 ─────────────────────────────────────────────────────


def build_period_details(df: pd.DataFrame, periods: list[str]) -> str:
    rows = []
    for p in periods:
        sub = df[df["period"] == p]
        nr = sub.dropna(subset=["negotiation_rate"]).sort_values("negotiation_rate")
        n_total = len(sub)
        n_nr = len(nr)
        n_big_drop = (nr["negotiation_rate"] <= -20).sum()
        if len(nr):
            worst = nr.iloc[0]
            max_drop_str = f'{worst["negotiation_rate"]:.2f}% {esc(worst["community"])}'
        else:
            max_drop_str = "—"
        rows.append(
            f'      <tr><td>{period_short(p)}</td>'
            f'<td class="num">{n_total}</td><td class="num">{n_nr}</td>'
            f'<td class="num">{n_big_drop}</td>'
            f'<td class="num down">{max_drop_str}</td></tr>'
        )
    return "\n".join(rows)


# ── TLDR 自动生成 ────────────────────────────────────────────────


def build_tldr(df: pd.DataFrame, periods: list[str]) -> str:
    n_periods = len(periods)
    per_period_n = df.groupby("period").size()

    peak_period = per_period_n.idxmax()
    peak_n = int(per_period_n.max())
    sorted_n = per_period_n.sort_values(ascending=False)
    second_period = sorted_n.index[1] if len(sorted_n) > 1 else None
    second_n = int(sorted_n.iloc[1]) if len(sorted_n) > 1 else 0
    show_combo = second_period and second_n >= peak_n * 0.9

    latest_period = periods[-1]
    latest_n = int(per_period_n[latest_period])

    # Same unit thick: 只用 ≥2 对子的相邻对（避免 n=1 噪音）
    thick_avgs = []
    for e, l in zip(periods[:-1], periods[1:]):
        pairs = analyze_same_community_room_delta(df, e, l)
        thick = [p for p in pairs if p["n_earlier"] >= 2 and p["n_later"] >= 2]
        if len(thick) >= 2:
            avg = sum(p["change_pct"] for p in thick) / len(thick)
            thick_avgs.append(avg)
    # 用中位数定方向（抗极端单期），上下震荡范围用 max
    if thick_avgs:
        sorted_avgs = sorted(thick_avgs, key=abs)
        median_abs = abs(sorted_avgs[len(sorted_avgs) // 2])
    else:
        median_abs = 0
    max_thick_abs = max((abs(a) for a in thick_avgs), default=0)

    if median_abs < 3:
        market_status = "总体平稳"
    elif median_abs < 6:
        market_status = "小幅震荡"
    else:
        market_status = "明显波动"

    # 假设检验
    tp = analyze_by_total_price(df)
    bucket_200 = next((b for b in tp["buckets"] if b["bucket"] == "≤200万"), None)
    bucket_200_rate = bucket_200["big_drop_rate"] * 100 if bucket_200 and bucket_200["big_drop_rate"] is not None else 0
    other_rates = [b["big_drop_rate"] * 100 for b in tp["buckets"] if b["bucket"] != "≤200万" and b["big_drop_rate"] is not None]
    other_str = f"{min(other_rates):.0f}-{max(other_rates):.0f}%" if other_rates else "—"

    # 找有没有片区"急涨"（品种偏移）
    n_pivot = df.groupby(["area", "period"]).size().unstack("period").fillna(0).astype(int)
    core_mask = (n_pivot >= 5).sum(axis=1) >= 3
    core_areas = n_pivot[core_mask].index.tolist()
    avg_pivot = df.groupby(["area", "period"])["unit_price_wan_sqm"].mean().unstack("period")
    mix_shift_areas = []
    for area in core_areas:
        # 同样：忽略样本 < 5 的期
        valid_vals = [
            float(avg_pivot.loc[area, p])
            for p in periods
            if not pd.isna(avg_pivot.loc[area, p]) and int(n_pivot.loc[area].get(p, 0)) >= 5
        ]
        # 仅在高端价位 (≥8万/m²) + 涨幅 ≥ 25% 时算品种偏移
        if len(valid_vals) >= 2 and valid_vals[-1] / valid_vals[0] >= 1.25 and valid_vals[-1] >= 8:
            mix_shift_areas.append(area)

    parts = [
        f'<strong>福田过去 {n_periods} 期市场{market_status}。</strong>',
    ]

    if show_combo:
        parts.append(
            f'成交量 {period_short(peak_period)} 达高峰（{peak_n} + {second_n} 笔）后 '
            f'{period_short(latest_period)} 回落到 {latest_n} 笔；'
        )
    elif peak_period != latest_period:
        parts.append(
            f'成交量 {period_short(peak_period)} 达高峰（{peak_n} 笔）后 '
            f'{period_short(latest_period)} 回落到 {latest_n} 笔；'
        )
    else:
        parts.append(f'最新一期 {period_short(latest_period)} 成交 {latest_n} 笔；')

    if max_thick_abs > 0.5:
        parts.append(f'同小区同户型在 ±{max_thick_abs:.0f}% 内震荡，没有持续单边方向。')
    else:
        parts.append('同小区同户型对比样本不足，无法判定真实方向。')

    parts.append(
        f'<strong>低总价段（≤200 万）的砍价深度依然显著高于其他段</strong>'
        f'（大跌占比 {bucket_200_rate:.1f}% vs {other_str}）。'
    )

    if mix_shift_areas:
        # 列出全部品种偏移片区（通常 1-2 个）
        names = "、".join(esc(a) for a in mix_shift_areas[:3])
        parts.append(
            f'<strong>{names}均价"急涨"</strong>是后期豪宅扎堆造成的品种偏移，'
            f'<strong>不是真实房价信号</strong>。'
        )

    return "".join(parts)


# ── 小区检索数据 ─────────────────────────────────────────────────


def build_search_data(df: pd.DataFrame) -> str:
    """生成 JSON：所有成交的精简列，供前端 JS 搜索。

    输出嵌入 <script type="application/json"> 块，必须转义 </ 防止
    数据中如出现 '</script>' 字串提前关闭脚本标签（latent XSS）。
    """
    records = []
    for _, r in df.iterrows():
        records.append({
            "p": r["period"],
            "a": str(r["area"]),
            "c": str(r["community"]),
            "rt": str(r["room_type"]),
            "sq": float(r["area_sqm"]),
            "tp": float(r["total_price_wan"]),
            "up": float(r["unit_price_wan_sqm"]),
            "nr": None if pd.isna(r["negotiation_rate"]) else float(r["negotiation_rate"]),
            "d": str(r["deal_date"].date()),
        })
    raw = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return raw.replace("</", "<\\/")


# ── 下期预测 ─────────────────────────────────────────────────────


def build_prediction_rows(df: pd.DataFrame, periods: list[str]) -> tuple[str, str]:
    """返回 (table rows HTML, caption HTML)。"""
    preds = predict_next_period(df, periods)
    if not preds:
        return ('        <tr><td colspan="6" class="dim">样本不足，无法预测。</td></tr>', "")

    next_p = next_period_label(periods)

    latest_period = periods[-1]
    html_rows = []
    for r in preds:
        area = esc(r["area"])
        cur = f'{r["current"]:.2f}'
        # 若 current 不是最末期，给 "当期均价" 加期次脚注 + 颜色提示
        cur_p = r["current_period"]
        if cur_p != latest_period:
            cur_html = (
                f'{cur} <span class="dim" '
                f'title="最末期 {period_short(latest_period)} 样本不足 5 笔，'
                f'当期均价基于 {period_short(cur_p)}">'
                f'({period_compact(cur_p)})</span>'
            )
        else:
            cur_html = cur
        pred = f'{r["predicted"]:.2f}'
        ci = f'[{r["ci_low"]:.2f}, {r["ci_high"]:.2f}]'
        chg = r["change_pct"]
        if chg > 1.5:
            chg_cls = "num up"
        elif chg < -1.5:
            chg_cls = "num down"
        else:
            chg_cls = "num dim"
        chg_str = f'{signed(chg, 1)}%'

        # R² 颜色：< 0.3 不可信
        r2 = r["r2"]
        r2_cls = "num up" if r2 >= 0.7 else ("num" if r2 >= 0.3 else "num dim")
        r2_str = f'{r2:.2f}'
        if r2 < 0.3:
            r2_str += ' <span class="dim">(噪)</span>'

        html_rows.append(
            f'        <tr>'
            f'<td>{area}</td>'
            f'<td class="num">{cur_html}</td>'
            f'<td class="num"><strong>{pred}</strong></td>'
            f'<td class="num dim">{ci}</td>'
            f'<td class="{chg_cls}">{chg_str}</td>'
            f'<td class="{r2_cls}">{r2_str}</td>'
            f'</tr>'
        )

    # caption: 选最强信号
    strong = [r for r in preds if r["r2"] >= 0.5 and abs(r["change_pct"]) >= 3]
    n_up = sum(1 for r in preds if r["change_pct"] > 1.5)
    n_down = sum(1 for r in preds if r["change_pct"] < -1.5)

    if strong:
        names = "、".join(esc(r["area"]) for r in strong[:3])
        caption = (
            f'<strong>{next_p} 预测信号：</strong>'
            f'{n_up} 个片区上行 / {n_down} 个下行（{len(preds) - n_up - n_down} 个持平）。'
            f'R² ≥ 0.5 且变化 ≥ 3% 的可信信号：<strong>{names}</strong>。'
        )
    else:
        caption = (
            f'<strong>{next_p} 预测信号：</strong>'
            f'{n_up} 个片区上行 / {n_down} 个下行（{len(preds) - n_up - n_down} 个持平），'
            f'但 R² 普遍偏低，5 期样本难以支撑高置信预测，仅供方向性参考。'
        )

    return "\n".join(html_rows), caption


# ── Outlier 通告 ─────────────────────────────────────────────────


def build_outlier_notice(outliers: pd.DataFrame) -> str:
    """生成"已剔除 N 条非市场价"通告 + 可折叠明细。"""
    n = len(outliers)
    if n == 0:
        return ""
    rows = []
    for _, r in outliers.iterrows():
        rows.append(
            f'        <tr>'
            f'<td>{period_short(r["period"])}</td>'
            f'<td>{esc(r["area"])}</td>'
            f'<td>{esc(r["community"])}</td>'
            f'<td class="num">{r["area_sqm"]:g}㎡</td>'
            f'<td class="num">{r["total_price_wan"]:g}万</td>'
            f'<td class="num down"><strong>{r["unit_price_wan_sqm"]:.2f}</strong>万/㎡</td>'
            f'</tr>'
        )
    return (
        f'<details class="outlier-notice">\n'
        f'  <summary>'
        f'已自动剔除 <strong>{n}</strong> 条非市场价（单价 &lt; 片区中位数 × 0.5），'
        f'点击展开明细'
        f'</summary>\n'
        f'  <table class="outlier-table">\n'
        f'    <thead><tr>'
        f'<th>期</th><th>片区</th><th>小区</th><th>面积</th>'
        f'<th>总价</th><th>单价</th>'
        f'</tr></thead>\n'
        f'    <tbody>\n'
        + "\n".join(rows) + "\n"
        f'    </tbody>\n'
        f'  </table>\n'
        f'</details>'
    )


# ── chart_data.json 生成 ───────────────────────────────────────────


def build_chart_data(df: pd.DataFrame, periods: list[str]) -> dict[str, Any]:
    n_pivot = df.groupby(["area", "period"]).size().unstack("period").fillna(0).astype(int)
    core_mask = (n_pivot >= 5).sum(axis=1) >= 3
    core_areas = n_pivot[core_mask].index.tolist()

    avg_pivot = df.groupby(["area", "period"])["unit_price_wan_sqm"].mean().unstack("period")
    med_pivot = df.groupby(["area", "period"])["unit_price_wan_sqm"].median().unstack("period")

    def series(pivot):
        return [
            {
                "name": area,
                "data": [
                    None if pd.isna(pivot.loc[area, p]) else round(float(pivot.loc[area, p]), 2)
                    for p in periods
                ],
            }
            for area in core_areas
        ]

    per_period = df.groupby("period").agg(
        n_total=("period", "size"),
        n_with_nr=("negotiation_rate", lambda s: s.notna().sum()),
        n_big_drop=("negotiation_rate", lambda s: (s <= -20).sum()),
    )
    per_period["big_drop_rate"] = (
        per_period["n_big_drop"] / per_period["n_with_nr"] * 100
    ).round(2)

    same_unit_bars = []
    for e, l in zip(periods[:-1], periods[1:]):
        pairs = analyze_same_community_room_delta(df, e, l)
        thick = [p for p in pairs if p["n_earlier"] >= 2 and p["n_later"] >= 2]
        if thick:
            avg = sum(p["change_pct"] for p in thick) / len(thick)
            same_unit_bars.append({"label": f"{l} vs {e}", "value": round(avg, 2), "n": len(thick)})
        else:
            same_unit_bars.append({"label": f"{l} vs {e}", "value": 0, "n": 0})

    return {
        "periods": periods,
        "core_areas": core_areas,
        "avg_series": series(avg_pivot),
        "median_series": series(med_pivot),
        "volume": [int(per_period.loc[p, "n_total"]) for p in periods],
        "big_drop_rate": [float(per_period.loc[p, "big_drop_rate"]) for p in periods],
        "same_unit": same_unit_bars,
    }


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    here = Path(__file__).parent
    csvs = sorted((here / "data").glob("*.csv"))
    df_raw = load_data(csvs)
    df, outliers = filter_outliers(df_raw)
    periods = sorted(df["period"].unique())

    placeholders = {
        "PERIOD_RANGE": f"{period_short(periods[0])} ~ {period_short(periods[-1])}",
        "N_PERIODS": str(len(periods)),
        "TOTAL_TXNS": str(len(df)),
        "TOTAL_RAW": str(len(df_raw)),
        "OUTLIER_COUNT": str(len(outliers)),
        "OUTLIER_NOTICE": build_outlier_notice(outliers),
        "TOTAL_WITH_NR": str(int(df["negotiation_rate"].notna().sum())),
        "UPDATED_DATE": datetime.now().strftime("%Y-%m-%d"),
        "TLDR_HTML": build_tldr(df, periods),
        "VOLUME_CAPTION": build_volume_caption(df, periods),
        "KPI_GRID_HTML": build_kpi_grid(df, periods),
        "AREA_OBSERVATIONS_ROWS": build_area_observations(df, periods),
        "SAME_UNIT_CONCLUSION_HTML": "",  # filled below
        "SAME_UNIT_ROWS": "",  # filled below
        "HYPOTHESIS_ROWS": "",  # filled below
        "HYPOTHESIS_CONCLUSION_HTML": "",  # filled below
        "SIGNAL_BLOCKS_HTML": build_signal_blocks(df, periods),
        "PERIOD_DETAILS_ROWS": build_period_details(df, periods),
        "NEXT_PERIOD_LABEL": next_period_label(periods),
        "PREDICTION_ROWS": "",  # filled below
        "PREDICTION_CAPTION": "",  # filled below
    }

    pred_rows, pred_caption = build_prediction_rows(df, periods)
    placeholders["PREDICTION_ROWS"] = pred_rows
    placeholders["PREDICTION_CAPTION"] = pred_caption

    placeholders["SEARCH_DATA_JSON"] = build_search_data(df)

    same_rows, same_conclusion = build_same_unit_rows(df, periods)
    placeholders["SAME_UNIT_ROWS"] = same_rows
    placeholders["SAME_UNIT_CONCLUSION_HTML"] = same_conclusion

    hyp_rows, hyp_conclusion = build_hypothesis(df)
    placeholders["HYPOTHESIS_ROWS"] = hyp_rows
    placeholders["HYPOTHESIS_CONCLUSION_HTML"] = hyp_conclusion

    template_path = here / "template" / "index.html"
    html = template_path.read_text(encoding="utf-8")

    for key, val in placeholders.items():
        html = html.replace("{{ " + key + " }}", val)

    # 校验：没有未替换的占位符
    remaining = re.findall(r"\{\{ [A-Z_]+ \}\}", html)
    if remaining:
        raise SystemExit(f"未替换的占位符: {set(remaining)}")

    out_html = here / "docs" / "index.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {out_html} ({out_html.stat().st_size} bytes)")

    # chart_data.json
    chart_data = build_chart_data(df, periods)
    out_json = here / "docs" / "chart_data.json"
    out_json.write_text(json.dumps(chart_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_json} ({out_json.stat().st_size} bytes)")

    print(f"\nPeriods: {periods}")
    print(f"Total transactions: {len(df)} ({df['negotiation_rate'].notna().sum()} with neg rate)")


if __name__ == "__main__":
    main()
