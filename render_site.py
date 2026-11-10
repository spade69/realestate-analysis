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
    split_auctions,
    analyze_school_premium,
    analyze_by_school,
    analyze_low_price,
    LOW_PRICE_MAX,
    LOW_PRICE_MIN_AREA_DEALS,
    SCHOOL_LIST_MIN_DEALS,
    SCHOOL_RECENT_PERIODS,
    attach_school,
    load_school_map,
    SCHOOL_TIERS,
    UNMAPPED_TIER,
    analyze_by_total_price,
    total_price_bucket,
    TOTAL_PRICE_LABELS,
    PRICE_ANALYSIS_MAX,
    analyze_same_community_room_delta,
)


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

    # 200 万以下 大跌占比
    tp = analyze_by_total_price(df)
    bucket_200 = next((b for b in tp["buckets"] if b["bucket"] == "<200万"), None)
    bucket_200_rate = bucket_200["big_drop_rate"] * 100 if bucket_200 and bucket_200["big_drop_rate"] is not None else 0
    other_rates = [
        b["big_drop_rate"] * 100
        for b in tp["buckets"]
        if b["bucket"] != "<200万" and b["big_drop_rate"] is not None
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
            "200 万以下大跌占比",
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


# 走势分类 → (显示名, tag 样式)。用于「多数板块共识」汇总。
CONSENSUS_LABELS = {
    "pullback": ("高点后回调", "tag-down"),
    "downtrend": ("持续下行", "tag-down"),
    "uptrend": ("持续上行", "tag-up"),
    "rebound": ("低点后回升", "tag-up"),
    "stable": ("窄幅/横盘稳定", "tag-flat"),
}


def classify_interp_category(warn_tag: str | None, interp: str) -> str:
    """把 categorize_area_trend 已生成的解读文本归类成粗粒度走势标签。"""
    if warn_tag == "品种偏移":
        return "mix_shift"
    if interp.startswith("样本不足"):
        return "insufficient"
    if interp.startswith("持续上行"):
        return "uptrend"
    if interp.startswith("持续下行"):
        return "downtrend"
    if "高点后回调" in interp:
        return "pullback"
    if "小坑后回升" in interp:
        return "rebound"
    return "stable"  # 低位稳定 / 窄幅震荡 / 横盘震荡


def build_area_consensus(df: pd.DataFrame, periods: list[str]) -> str:
    """核心片区走势分类汇总：是否存在多数共识（回调/上行/下行/稳定）。"""
    n_pivot = df.groupby(["area", "period"]).size().unstack("period").fillna(0).astype(int)
    core_mask = (n_pivot >= 5).sum(axis=1) >= 3
    core_areas = n_pivot[core_mask].index.tolist()
    avg_pivot = df.groupby(["area", "period"])["unit_price_wan_sqm"].mean().unstack("period")

    counts: dict[str, int] = {}
    excluded = 0
    for area in core_areas:
        values = [
            None if pd.isna(avg_pivot.loc[area, p]) else float(avg_pivot.loc[area, p])
            for p in periods
        ]
        area_counts = [int(n_pivot.loc[area].get(p, 0)) for p in periods]
        _, warn_tag, interp = categorize_area_trend(values, periods, area_counts)
        cat = classify_interp_category(warn_tag, interp)
        if cat in ("mix_shift", "insufficient"):
            excluded += 1
            continue
        counts[cat] = counts.get(cat, 0) + 1

    total_valid = sum(counts.values())
    if total_valid == 0:
        return '<p class="dim">核心片区样本不足，暂无法判断整体走势。</p>'

    dom_cat, dom_n = max(counts.items(), key=lambda kv: kv[1])
    dom_label, _ = CONSENSUS_LABELS[dom_cat]
    dom_pct = dom_n / total_valid * 100

    tags = "".join(
        f'<span class="tag {CONSENSUS_LABELS[cat][1]}">{CONSENSUS_LABELS[cat][0]} {n}</span>'
        for cat, n in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    excluded_note = f'，另有 {excluded} 个片区因样本不足/品种偏移不计入判断' if excluded else ''

    if dom_pct >= 50:
        headline = (
            f'<strong>{total_valid} 个核心片区中，{dom_n} 个（{dom_pct:.0f}%）呈「{dom_label}」，'
            f'是当前主流态势{excluded_note}。</strong>'
        )
    elif dom_pct >= 40:
        headline = (
            f'<strong>{total_valid} 个核心片区中，{dom_n} 个（{dom_pct:.0f}%）呈「{dom_label}」占优，'
            f'但未过半{excluded_note}。</strong>'
        )
    else:
        headline = f'<strong>{total_valid} 个核心片区走势分化，没有单一主流{excluded_note}。</strong>'

    return f'<p>{headline}</p>\n    <p>{tags}</p>'


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


# ── 假设检验 ─────────────────────────────────────────────────────


def build_hypothesis(df: pd.DataFrame) -> tuple[str, str]:
    tp = analyze_by_total_price(df)
    buckets = tp["buckets"]

    rows = []
    for b in buckets:
        is_top = b["bucket"] == "<200万"
        n = b["n"]
        avg_nr = b["avg_neg_rate"]
        rate = b["big_drop_rate"] * 100 if b["big_drop_rate"] is not None else 0
        avg_nr_str = f"{avg_nr:.2f}%" if avg_nr is not None else "—"
        rate_str = f"{rate:.1f}%"

        if is_top:
            rows.append(
                f'        <tr class="row-highlight">\n'
                f'          <td><strong>{esc(b["bucket"])}</strong></td>\n'
                f'          <td class="num">{n}</td>\n'
                f'          <td class="num down">{avg_nr_str}</td>\n'
                f'          <td class="num warn"><strong>{rate_str}</strong></td>\n'
                f'        </tr>'
            )
        else:
            rows.append(
                f'        <tr><td>{esc(b["bucket"])}</td><td class="num">{n}</td>'
                f'<td class="num">{avg_nr_str}</td><td class="num">{rate_str}</td></tr>'
            )

    cor = tp["correlation"]
    pr, pp = cor["pearson_r"], cor["pearson_p"]
    sr, sp = cor["spearman_r"], cor["spearman_p"]

    pearson_sig = pr is not None and pp is not None and pp < 0.05
    spearman_sig = sr is not None and sp is not None and sp < 0.05

    # 桶 200 万 vs 其他段范围
    other_rates = [b["big_drop_rate"] * 100 for b in buckets if b["bucket"] != "<200万" and b["big_drop_rate"] is not None]
    bucket_200 = next((b for b in buckets if b["bucket"] == "<200万"), None)
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
            f'<strong>200 万以下砍价幅度依然最狠（{bucket_200_rate:.1f}% vs 其他段 {other_str}）</strong>，'
            '但桶间不是严格线性递增——大概率因为高总价段也出现零星大砍价。'
        )
    elif spearman_sig:
        meaning = f'秩相关显著但 200 万以下并非全段最高，结论需谨慎。'
    else:
        meaning = '<strong>未发现显著相关</strong>，假设暂不成立（样本可能仍不足）。'

    conclusion_lines.append(f'<p>意思是：{meaning}</p>')

    return "\n".join(rows), "\n    ".join(conclusion_lines)


# ── 总价段成交结构 ─────────────────────────────────────────────────


PRICE_MIX_LOW = ("<200万", "200-300万")  # 「低总价」= 300 万以下
PRICE_MIX_WINDOW = 4  # 说明文字：最早 N 期 vs 最近 N 期


def price_mix_counts(df: pd.DataFrame, periods: list[str]) -> list[list[int]]:
    """counts[i][j] = 第 i 个总价段在第 j 期的成交笔数。"""
    b = df["total_price_wan"].map(total_price_bucket)
    pivot = (df.assign(_b=b).groupby(["_b", "period"]).size()
               .unstack("period").reindex(index=TOTAL_PRICE_LABELS, columns=periods).fillna(0).astype(int))
    return pivot.values.tolist()


def build_price_mix(df: pd.DataFrame, periods: list[str]) -> tuple[str, str]:
    counts = price_mix_counts(df, periods)
    totals = [sum(row[j] for row in counts) for j in range(len(periods))]

    head = "".join(f'<th class="num">{esc(l)}</th>' for l in TOTAL_PRICE_LABELS)
    rows = []
    for j, p in enumerate(periods):
        cells = "".join(
            f'<td class="num">{counts[i][j]}<span class="dim"> · {counts[i][j] / totals[j] * 100:.0f}%</span></td>'
            if totals[j] else '<td class="num">0</td>'
            for i in range(len(TOTAL_PRICE_LABELS)))
        rows.append(f'<tr><td>{esc(period_short(p))}</td>{cells}<td class="num">{totals[j]}</td></tr>')
    grand = sum(totals)
    foot = "".join(
        f'<td class="num"><strong>{sum(row)}</strong><span class="dim"> · {sum(row) / grand * 100:.0f}%</span></td>'
        for row in counts)
    table = ('<table><thead><tr><th>期次</th>' + head + '<th class="num">合计</th></tr></thead><tbody>'
             + "".join(rows)
             + f'<tr><td><strong>全部</strong></td>{foot}<td class="num"><strong>{grand}</strong></td></tr>'
             + '</tbody></table>')

    low_idx = [TOTAL_PRICE_LABELS.index(l) for l in PRICE_MIX_LOW]
    def low_share(js):
        n = sum(totals[j] for j in js)
        return sum(counts[i][j] for i in low_idx for j in js) / n * 100 if n else 0.0
    n_excl = int((df["total_price_wan"] >= PRICE_ANALYSIS_MAX).sum())
    excl_note = (f'只统计 {PRICE_ANALYSIS_MAX} 万以下，另有 {n_excl} 笔 ≥{PRICE_ANALYSIS_MAX} 万成交未计入（占比也按 {PRICE_ANALYSIS_MAX} 万以下计算）。'
                 if n_excl else '')
    w = min(PRICE_MIX_WINDOW, len(periods) // 2)
    if w >= 1:
        early, late = low_share(range(w)), low_share(range(len(periods) - w, len(periods)))
        diff = round(late - early)
        trend = "上升" if diff > 2 else ("下降" if diff < -2 else "基本持平")
        caption = (f'每期成交按总价分段堆叠（已剔除法拍与非市场价），可切换笔数 / 占比。'
                   f'{excl_note}300 万以下成交占比：最早 {w} 期 <strong>{early:.0f}%</strong> → '
                   f'最近 {w} 期 <strong>{late:.0f}%</strong>（{trend}，{f'{diff:+d}' if diff else '0'} 个百分点）。')
    else:
        caption = f'每期成交按总价分段堆叠（已剔除法拍与非市场价），可切换笔数 / 占比。{excl_note}'
    return table, caption


# ── 低总价段（200 万以下）─────────────────────────────────────────


def _low_area_row(a: dict[str, Any]) -> str:
    pm = a["per_month"]
    return (f'<tr data-area="{esc(a["area"])}">'
            f'<td><button type="button" class="row-toggle" aria-expanded="false">'
            f'<span class="caret" aria-hidden="true"></span>{esc(a["area"])}</button></td>'
            + _sort_td(str(a["n"]), a["n"])
            + _sort_td(str(a["n_communities"]), a["n_communities"])
            + _sort_td(f'{a["top_share"] * 100:.0f}%', round(a["top_share"], 4))
            + _sort_td(f'{a["median_total"]:g}万', a["median_total"])
            + _sort_td(f'{a["median_unit"]:.2f}', a["median_unit"])
            + _sort_td(f'{a["median_sqm"]:g}㎡', a["median_sqm"])
            + _sort_td("—" if pm is None else f'{pm:.1f}', None if pm is None else round(pm, 3))
            + _sort_td(_neg(a["avg_neg_rate"]), a["avg_neg_rate"])
            + '</tr>')


def build_low_price_section(df: pd.DataFrame) -> str:
    """200 万以下成交：片区表 + 小区表（图表由前端按 chart_data.low_price 渲染）。"""
    r = analyze_low_price(df)
    if not r["n"]:
        return '<p class="dim">暂无 {} 万以下成交。</p>'.format(LOW_PRICE_MAX)
    n, n_all = r["n"], r["n_all"]
    n_top = sum(a["n_top"] for a in r["areas"])
    small = [a for a in r["areas"] if a["n"] < LOW_PRICE_MIN_AREA_DEALS]
    r["areas"] = [a for a in r["areas"] if a["n"] >= LOW_PRICE_MIN_AREA_DEALS]

    area_rows = "".join(_low_area_row(a) for a in r["areas"])
    area_table = (
        '<div class="table-wrap"><table id="low-area-table" class="js-sortable"><thead><tr>'
        + _sort_th("片区", cls="") + _sort_th("成交", "desc") + _sort_th("小区", "desc")
        + _sort_th("名校占比", "desc") + _sort_th("总价中位", "asc") + _sort_th("单价中位", "asc")
        + _sort_th("面积中位", "desc") + _sort_th("月均成交", "desc") + _sort_th("平均谈价", "desc")
        + '</tr></thead><tbody>' + area_rows + '</tbody></table></div>')

    small_note = (f'<p class="caption-note">累计不足 {LOW_PRICE_MIN_AREA_DEALS} 笔的片区未列出：'
                  + "、".join(f'{esc(a["area"])}({a["n"]})' for a in small) + '。</p>') if small else ""
    one_room = int((df.loc[df["total_price_wan"] < LOW_PRICE_MAX, "room_type"] == "1室").sum())
    intro = (f'<p class="caption-note">共 <strong>{n}</strong> 笔（占全部 {n / n_all * 100:.0f}%），'
             f'分布在 {len(r["areas"]) + len(small)} 个片区；其中 1 室 {one_room} 笔，基本是小户型 / 公寓。'
             f'对口名校学位的 {n_top} 笔（{n_top / n * 100:.0f}%）。已剔除法拍与非市场价。</p>')
    figure = ('<figure><div id="chart-low-price" class="chart" role="img" '
              f'aria-label="{LOW_PRICE_MAX} 万以下成交按片区分布，区分名校与非名校学位">'
              '<div class="chart-loading">图表加载中…</div></div>'
              '<figcaption>各片区 200 万以下成交笔数，按是否对口名校学位分色</figcaption></figure>')
    periods = sorted(df["period"].unique())
    opts = '<option value="all">全部累计</option>' + "".join(
        f'<option value="{esc(p)}">{esc(p)}</option>' for p in reversed(periods))
    scope = ('<div class="scope-bar"><label for="low-scope">片区图表期次</label>'
             f'<select id="low-scope">{opts}</select>'
             '<span class="dim" id="low-scope-note">全部累计</span></div>')
    return (intro + scope + figure
            + '<h3>片区<span class="dim" style="font-weight:400;font-size:0.85rem;">'
              ' · 随期次切换，无成交的片区不显示；点片区名展开该片区的小区</span></h3>'
            + area_table + small_note)


def _round(v: Any) -> Any:
    return round(v, 4) if isinstance(v, float) else v


def low_price_chart_data(df: pd.DataFrame) -> dict[str, Any]:
    """scopes: all + 各单期；每个 scope 给片区聚合（只含有成交的片区，各带其小区），供前端切换。

    累计不足 LOW_PRICE_MIN_AREA_DEALS 笔的片区在所有 scope 里都不列。
    """
    keep = {a["area"] for a in analyze_low_price(df)["areas"] if a["n"] >= LOW_PRICE_MIN_AREA_DEALS}
    keys = ["all"] + sorted(df["period"].unique())
    scopes = {}
    for k in keys:
        sub = df if k == "all" else df[df["period"] == k]
        areas = []
        for a in analyze_low_price(sub)["areas"]:
            if a["area"] not in keep:
                continue
            row = {key: _round(v) for key, v in a.items() if key != "communities"}
            row["communities"] = [{key: _round(v) for key, v in c.items() if key != "area"}
                                  for c in a["communities"]]
            areas.append(row)
        scopes[k] = {"n": sum(a["n"] for a in areas), "areas": areas}
    return {"max": LOW_PRICE_MAX, "scopes": scopes}


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
    bucket_200 = next((b for b in tp["buckets"] if b["bucket"] == "<200万"), None)
    bucket_200_rate = bucket_200["big_drop_rate"] * 100 if bucket_200 and bucket_200["big_drop_rate"] is not None else 0
    other_rates = [b["big_drop_rate"] * 100 for b in tp["buckets"] if b["bucket"] != "<200万" and b["big_drop_rate"] is not None]
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
        f'<strong>低总价段（200 万以下）的砍价深度依然显著高于其他段</strong>'
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


# ── 学区溢价 ─────────────────────────────────────────────────────


def _pct(v: float) -> str:
    return f"{v:+.1f}%"


def build_school_section(df: pd.DataFrame) -> str:
    res = analyze_school_premium(df)
    n_total, n_mapped = res["n_total"], res["n_mapped"]
    cov = n_mapped / n_total * 100 if n_total else 0
    parts = [
        f'<p class="caption-note">已标注学区的成交 <strong>{n_mapped}</strong> / {n_total} 笔'
        f'（{cov:.0f}%）· 映射表 <code>ref/school_map.csv</code> · 分档：'
        + " / ".join(SCHOOL_TIERS) + '</p>'
    ]
    if n_mapped == 0:
        parts.append('<div class="conclusion"><p>学区映射表尚未填写，暂无分析。</p></div>')
        return "\n  ".join(parts)

    rows = []
    for t in res["by_tier"]:
        nr = f'{t["avg_neg_rate"]:.2f}%' if t["avg_neg_rate"] is not None else "—"
        rows.append(
            f'<tr><td>{esc(t["tier"])}</td><td class="num">{t["n_communities"]}</td>'
            f'<td class="num">{t["n"]}</td><td class="num">{t["median_unit_price"]:.2f}</td>'
            f'<td class="num">{nr}</td></tr>'
        )
    parts.append(
        '<div class="table-wrap"><table><thead><tr><th>学区档</th><th class="num">小区数</th>'
        '<th class="num">成交</th><th class="num">单价中位数 (万/㎡)</th><th class="num">平均谈价率</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
    )

    if res["same_area"]:
        rows = [
            f'<tr><td>{esc(r["area"])}</td><td class="num">{r["n_top"]} / {r["n_base"]}</td>'
            f'<td class="num">{r["median_top"]:.2f}</td><td class="num">{r["median_base"]:.2f}</td>'
            f'<td class="num {"up" if r["premium_pct"] > 0 else "down"}">{_pct(r["premium_pct"])}</td></tr>'
            for r in sorted(res["same_area"], key=lambda r: -r["premium_pct"])
        ]
        parts.append(
            '<h3>同片区对比：名校 vs 普通</h3>'
            '<div class="table-wrap"><table><thead><tr><th>片区</th><th class="num">笔数 名校/普通</th>'
            '<th class="num">名校中位</th><th class="num">普通中位</th><th class="num">溢价</th>'
            '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
        )

    lines = []
    if res["same_area_median_pct"] is not None:
        lines.append(f'同片区对比：{len(res["same_area"])} 个片区的名校溢价中位数为 '
                     f'<strong>{_pct(res["same_area_median_pct"])}</strong>（未控制房龄/面积）。')
    reg = res["regression"]
    if reg:
        for t in reg["terms"]:
            sig = t["p_value"] < 0.05
            lines.append(
                f'回归（控制片区、期次、房龄、面积、户型，n={reg["n"]}，R²={reg["r2"]:.2f}）：'
                f'{esc(t["tier"])}相对普通学位 <strong class="{"up" if sig else "dim"}">{_pct(t["premium_pct"])}</strong>'
                f'（95% CI {_pct(t["ci_low_pct"])} ~ {_pct(t["ci_high_pct"])}，p={t["p_value"]:.2g}，'
                f'{"显著" if sig else "不显著"}）。'
            )
        lines.append(f'房龄每增加 1 年，单价约变化 {_pct(reg["age_pct_per_year"])}。')
    else:
        lines.append('已标注样本不足，暂不做回归（需普通档及名校或无学位档各 ≥3 笔、合计 ≥30 笔）。')
    parts.append('<div class="conclusion">' + "".join(f"<p>{l}</p>" for l in lines) + '</div>')
    parts.append(build_school_cards(df))
    return "\n  ".join(parts)


def _deal_line(d: dict[str, Any]) -> str:
    warn = f' <span class="warn" title="{esc(d["note"])}">⚠ 部分楼栋</span>' if d.get("note") and "栋" in d["note"] else ""
    return (f'{esc(d["community"])} · {d["sqm"]:g}㎡ {esc(d["room_type"])} · '
            f'{d["unit"]:.2f}万/㎡ · {period_short(d["period"])}{warn}')


def _neg(v: float | None) -> str:
    return f"{v:.1f}%" if v is not None else "—"


def _wan(d: dict[str, Any] | None) -> str:
    return f'{d["total"]:g}万' if d else "—"


def _liquidity(s: dict[str, Any]) -> str:
    nr = f'{s["avg_neg_rate"]:.1f}%' if s["avg_neg_rate"] is not None else "—"
    return (f'月均 {s["per_month"]:.1f} 笔 · {s["n_active_periods"]}/{s["n_periods"]} 期有成交'
            f' · 谈价 {nr}')


def _sort_td(text: str, v: float | None, cls: str = "num") -> str:
    """可排序表格的单元格：data-v 存排序用的原始数值，空值排最后。"""
    dv = "" if v is None else f"{v:g}"
    return f'<td class="{cls}" data-v="{dv}">{text}</td>'


def _sort_th(label: str, first: str | None = None, cls: str = "num") -> str:
    """first: 首次点击的排序方向（desc / asc）；None = 不可排序。"""
    if first is None:
        return f'<th class="{cls}">{label}</th>'
    return (f'<th class="{cls} sortable" data-first="{first}" aria-sort="none">'
            f'<button type="button">{label}<span class="sort-ind" aria-hidden="true"></span></button></th>')


def _filter_bar(table_id: str, label: str, counts: tuple[int, int]) -> str:
    """全部 / 名校 / 非名校 筛选按钮；行上用 data-group="top|other"。"""
    total = counts[0] + counts[1]
    return (f'<div class="school-filter" role="group" aria-label="{esc(label)}" data-for="{table_id}">'
            f'<button type="button" data-filter="all" aria-pressed="true">全部 {total}</button>'
            f'<button type="button" data-filter="top" aria-pressed="false">名校 ★ {counts[0]}</button>'
            f'<button type="button" data-filter="other" aria-pressed="false">非名校 {counts[1]}</button>'
            '<span class="dim school-filter-hint">点表头排序，再点切换升 / 降序</span>'
            '</div>')


def build_school_cards(df: pd.DataFrame) -> str:
    """名校逐校卡片（上车门槛 + 流通性）+ 全部学校汇总表。成交过少的学校不展示。"""
    all_schools = analyze_by_school(df)
    schools = [s for s in all_schools if s["n"] >= SCHOOL_LIST_MIN_DEALS]
    hidden = [s for s in all_schools if s["n"] < SCHOOL_LIST_MIN_DEALS]
    if not schools:
        return ""
    periods = sorted(df["period"].unique())
    recent_label = f"近 {SCHOOL_RECENT_PERIODS} 期（{period_short(periods[-SCHOOL_RECENT_PERIODS])} 起）"

    cards = []
    for s in (x for x in schools if x["is_top"]):
        c, c2, cr = s["cheapest"], s["cheapest_2r"], s["cheapest_recent"]
        comms = "、".join(
            f'{esc(x["community"])}<span class="dim">({x["n"]})</span>'
            for x in s["communities"][:5]
        ) + ("…" if len(s["communities"]) > 5 else "")
        lines = [
            f'<div class="school-head"><h3>{esc(s["school"])}</h3>'
            f'<span class="dim">{s["n"]} 笔 · {s["n_communities"]} 个小区</span></div>',
            f'<div class="school-label">最低成交总价</div>'
            f'<div class="kpi-value">{c["total"]:g}<span class="unit">万</span></div>'
            f'<div class="school-sub">{_deal_line(c)}</div>',
            '<dl class="school-dl">',
            f'<dt>2室及以上最低</dt><dd>' + (f'<strong>{c2["total"]:g}万</strong> {_deal_line(c2)}' if c2 else '<span class="dim">无成交</span>') + '</dd>',
            f'<dt>{recent_label}最低</dt><dd>' + (f'<strong>{cr["total"]:g}万</strong> {_deal_line(cr)}' if cr else '<span class="dim">无成交</span>') + '</dd>',
            f'<dt>总价 / 单价中位</dt><dd>{s["median_total"]:g}万 · {s["median_unit"]:.2f}万/㎡</dd>',
            f'<dt>流通性</dt><dd>{_liquidity(s)}</dd>',
            f'<dt>对口小区</dt><dd>{comms}</dd>',
            '</dl>',
        ]
        cards.append('<div class="school-card">' + "".join(lines) + '</div>')

    rows = []
    for s in schools:
        c, c2, cr = s["cheapest"], s["cheapest_2r"], s["cheapest_recent"]
        rows.append(
            f'<tr data-group="{"top" if s["is_top"] else "other"}">'
            f'<td>{esc(s["school"])}{" ★" if s["is_top"] else ""}</td>'
            + _sort_td(str(s["n_communities"]), s["n_communities"])
            + _sort_td(str(s["n"]), s["n"])
            + _sort_td(f'{c["total"]:g}万', c["total"])
            + f'<td>{esc(c["community"])}</td>'
            + _sort_td(_wan(c2), c2["total"] if c2 else None)
            + _sort_td(_wan(cr), cr["total"] if cr else None)
            + _sort_td(f'{s["median_total"]:g}万', s["median_total"])
            + _sort_td(f'{s["median_unit"]:.2f}', s["median_unit"])
            + _sort_td(f'{s["per_month"]:.1f}', round(s["per_month"], 3))
            + _sort_td(f'{s["n_active_periods"]}/{s["n_periods"]}', s["n_active_periods"])
            + _sort_td(_neg(s["avg_neg_rate"]), s["avg_neg_rate"])
            + '</tr>'
        )
    n_top = sum(1 for s in schools if s["is_top"])

    hidden_note = (
        f'<p class="caption-note">另有 {len(hidden)} 所学校累计成交不足 {SCHOOL_LIST_MIN_DEALS} 笔，'
        '流通性差、价格参考意义有限，未列出：'
        + "、".join(f'{esc(h["school"])}({h["n"]})' for h in sorted(hidden, key=lambda h: -h["n"]))
        + '。</p>'
    ) if hidden else ""
    return (
        '<h3>重点学校：上车门槛</h3>'
        '<p class="caption-note">按对口小学汇总全部期次成交（已剔除法拍和非市场价）。'
        '共享学区/大学区的小区同时计入每所对口学校；括号内为该小区成交笔数。'
        '最低价可能是商住公寓或极小户型，看「2室及以上最低」更接近家庭上车门槛。'
        f'只展示累计成交 ≥{SCHOOL_LIST_MIN_DEALS} 笔的学校。'
        '流通性：月均成交（一期 = 半个月）、有成交的期数、平均谈价率（越接近 0 越好卖）。</p>'
        '<div class="school-grid">' + "".join(cards) + '</div>'
        '<details><summary>全部学校（' + str(len(schools)) + ' 所，★ 为名校）</summary>'
        + _filter_bar("school-table", "筛选学校", (n_top, len(schools) - n_top)) +
        '<div class="table-wrap" style="margin:0;border:none;box-shadow:none;">'
        '<table id="school-table" class="js-sortable"><thead><tr>'
        + _sort_th("学校", cls="")
        + _sort_th("小区", "desc") + _sort_th("成交", "desc") + _sort_th("最低总价", "asc") + _sort_th("最低价小区", cls="")
        + _sort_th("2室+最低", "asc") + _sort_th(f"近{SCHOOL_RECENT_PERIODS}期最低", "asc")
        + _sort_th("总价中位", "asc") + _sort_th("单价中位", "asc")
        + _sort_th("月均成交", "desc") + _sort_th("有成交期数", "desc") + _sort_th("平均谈价", "desc")
        + '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>' + hidden_note + '</details>'
    )


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
            "s": " / ".join(x for x in (r["school_primary"], r["school_middle"]) if x),
            "st": "" if r["school_tier"] == UNMAPPED_TIER else str(r["school_tier"]),
        })
    raw = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return raw.replace("</", "<\\/")


# ── Outlier 通告 ─────────────────────────────────────────────────


def _excluded_notice(rows_df: pd.DataFrame, summary: str) -> str:
    """生成"已剔除 N 条 xxx"通告 + 可折叠明细。"""
    if len(rows_df) == 0:
        return ""
    rows = []
    for _, r in rows_df.iterrows():
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
        f'  <summary>{summary}，点击展开明细</summary>\n'
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


def build_outlier_notice(outliers: pd.DataFrame) -> str:
    return _excluded_notice(
        outliers,
        f'已自动剔除 <strong>{len(outliers)}</strong> 条非市场价（单价 &lt; 片区中位数 × 0.5）',
    )


def build_auction_notice(auctions: pd.DataFrame) -> str:
    return _excluded_notice(
        auctions,
        f'另有 <strong>{len(auctions)}</strong> 条法拍成交，不计入价格与谈价分析',
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

    return {
        "periods": periods,
        "core_areas": core_areas,
        "avg_series": series(avg_pivot),
        "median_series": series(med_pivot),
        "volume": [int(per_period.loc[p, "n_total"]) for p in periods],
        "big_drop_rate": [float(per_period.loc[p, "big_drop_rate"]) for p in periods],
        "low_price": low_price_chart_data(df),
        "price_mix": {"labels": TOTAL_PRICE_LABELS, "counts": price_mix_counts(df, periods)},
    }


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    here = Path(__file__).parent
    csvs = sorted((here / "data").glob("*.csv"))
    df_raw = load_data(csvs)
    df_market, auctions = split_auctions(df_raw)
    df, outliers = filter_outliers(df_market)
    school_map_path = here / "ref" / "school_map.csv"
    school_map = (load_school_map(school_map_path) if school_map_path.exists()
                  else pd.DataFrame(columns=["community", "primary_school", "middle_school", "tier", "note"]))
    df = attach_school(df, school_map)
    periods = sorted(df["period"].unique())

    placeholders = {
        "PERIOD_RANGE": f"{period_short(periods[0])} ~ {period_short(periods[-1])}",
        "N_PERIODS": str(len(periods)),
        "TOTAL_TXNS": str(len(df)),
        "TOTAL_RAW": str(len(df_raw)),
        "OUTLIER_COUNT": str(len(outliers)),
        "OUTLIER_NOTICE": build_outlier_notice(outliers) + build_auction_notice(auctions),
        "TOTAL_WITH_NR": str(int(df["negotiation_rate"].notna().sum())),
        "HYPOTHESIS_N": str(int(df.loc[df["total_price_wan"] < PRICE_ANALYSIS_MAX, "negotiation_rate"].notna().sum())),
        "PRICE_ANALYSIS_MAX": str(PRICE_ANALYSIS_MAX),
        "UPDATED_DATE": datetime.now().strftime("%Y-%m-%d"),
        "TLDR_HTML": build_tldr(df, periods),
        "VOLUME_CAPTION": build_volume_caption(df, periods),
        "KPI_GRID_HTML": build_kpi_grid(df, periods),
        "AREA_CONSENSUS_HTML": build_area_consensus(df, periods),
        "AREA_OBSERVATIONS_ROWS": build_area_observations(df, periods),
        "HYPOTHESIS_ROWS": "",  # filled below
        "HYPOTHESIS_CONCLUSION_HTML": "",  # filled below
        "PERIOD_DETAILS_ROWS": build_period_details(df, periods),
    }

    mix_table, mix_caption = build_price_mix(df, periods)
    placeholders["PRICE_MIX_TABLE"] = mix_table
    placeholders["LOW_PRICE_SECTION_HTML"] = build_low_price_section(df)
    placeholders["PRICE_MIX_CAPTION"] = mix_caption

    placeholders["SEARCH_DATA_JSON"] = build_search_data(df)
    placeholders["SCHOOL_SECTION_HTML"] = build_school_section(df)

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
