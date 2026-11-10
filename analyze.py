"""福田二手成交分析（spec §4.1–§4.6）。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats as _stats


COLUMNS = [
    "period", "region", "area", "community", "year_built",
    "room_type", "area_sqm", "total_price_wan", "unit_price_wan_sqm",
    "negotiation_rate", "deal_date",
]
# 可选尾列：法拍标记（1 = 法拍）。旧 CSV 无此列，读入时补 False。
AUCTION_COLUMN = "is_auction"

THRESHOLDS = (-20, -30, -50)

# Outlier: 单价低于片区中位数的 50% → 视为非市场价（底商/老破小/关联交易）
OUTLIER_RATIO = 0.5

DIST_BUCKETS = [
    ("≥-5%",        lambda r: r >= -5),
    ("-5% ~ -10%",  lambda r: -10 <= r < -5),
    ("-10% ~ -20%", lambda r: -20 <= r < -10),
    ("-20% ~ -30%", lambda r: -30 <= r < -20),
    ("-30% ~ -50%", lambda r: -50 <= r < -30),
    ("≤-50%",       lambda r: r < -50),
]

# 总价段分析（结构图 / 假设检验）只看 1000 万以下；豪宅成交另计，不影响其他板块
PRICE_ANALYSIS_MAX = 1000

TOTAL_PRICE_BUCKETS = [
    ("<200万",    lambda p: p < 200),
    ("200-300万", lambda p: 200 <= p < 300),
    ("300-400万", lambda p: 300 <= p < 400),
    ("400-600万", lambda p: 400 <= p < 600),
    ("600-1000万", lambda p: 600 <= p < PRICE_ANALYSIS_MAX),
]
TOTAL_PRICE_LABELS = [b[0] for b in TOTAL_PRICE_BUCKETS]

# 小区名 OCR 误字 / 异写 → 标准名。读入时统一，原始 CSV 保持截图原样。
# 只收能确认的（同片区、一字之差、另一写法与学校名单或楼盘正式名一致）。
COMMUNITY_ALIASES = {
    "航天工业华庭": "航天立业华庭",
    "恒邦壹峰": "恒邦壹峯",
    "阁网网苑": "阁林网苑",
    "绿景红湾": "绿景虹湾",
    "绿景中城天悦花园": "绿景中城天邑花园",
    "嘉葆阳光雅居": "嘉鑫阳光雅居",
    "景源年庭": "景源华庭",
    "富通九城公馆": "富通九曜公馆",
    "深山上城": "深业上城",
    "香茘绿洲": "香荔绿洲",
    "香蜜湖水樾花都": "香蜜湖水榭花都",
}


def load_data(paths: Iterable[Path | str]) -> pd.DataFrame:
    """读多个 CSV，拼成单一 DataFrame，校验列、转换类型。"""
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        if list(df.columns) == COLUMNS:
            df[AUCTION_COLUMN] = False
        elif list(df.columns) == COLUMNS + [AUCTION_COLUMN]:
            df[AUCTION_COLUMN] = df[AUCTION_COLUMN].eq("1")
        else:
            raise ValueError(f"{p}: columns mismatch, got {list(df.columns)}")
        for col in ["area_sqm", "total_price_wan", "unit_price_wan_sqm",
                    "negotiation_rate", "year_built"]:
            df[col] = pd.to_numeric(df[col].replace("", pd.NA), errors="raise")
        df["deal_date"] = pd.to_datetime(df["deal_date"], format="%Y-%m-%d")
        df["community"] = df["community"].replace(COMMUNITY_ALIASES)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def split_auctions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (market_df, auctions_df)。法拍价非市场成交、户型与谈价率缺失，不进价格分析。"""
    mask = df[AUCTION_COLUMN]
    return df[~mask].copy().reset_index(drop=True), df[mask].copy().reset_index(drop=True)


# ── 学区 ──────────────────────────────────────────────────────────

SCHOOL_MAP_COLUMNS = ["area", "community", "n_deals", "primary_school",
                      "middle_school", "tier", "source", "note"]
# 名校 = 公认优质学位；普通 = 有公办学位；无学位 = 公寓/商住/产权不足等无法入学
SCHOOL_TIERS = ("名校", "普通", "无学位")
UNMAPPED_TIER = "未标注"
SCHOOL_MIN_N = 3  # 同片区对比时每档最少样本
SCHOOL_RECENT_PERIODS = 4  # “近期最低价”看最近几期
SCHOOL_LIST_MIN_DEALS = 10  # 学校卡片 / 全部学校表只展示累计成交 ≥ 此值的学校

# 名校：社会公认的头部公办小学及集团校区（tools/build_school_map.py 据此给小区定档）
TOP_SCHOOLS = (
    "深圳实验学校小学部", "深圳外国语小学", "百花小学", "园岭小学",
    "荔园小学(通新岭)", "荔园小学(百花)", "荔园小学(玮鹏)", "荔园小学(八卦岭)",
    "荔园外国语小学", "荔园外国语小学(天骄)", "荔园外国语小学(狮岭)",
    "福田区外国语小学", "红岭实验小学", "莲花小学",
)


def load_school_map(path: Path | str) -> pd.DataFrame:
    """读 小区→学校 映射表。tier 为空的行视为未标注，不参与匹配。"""
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(m.columns) != SCHOOL_MAP_COLUMNS:
        raise ValueError(f"{path}: columns mismatch, got {list(m.columns)}")
    bad = set(m["tier"]) - set(SCHOOL_TIERS) - {""}
    if bad:
        raise ValueError(f"{path}: unknown tier {sorted(bad)}, expected {SCHOOL_TIERS}")
    m = m[m["tier"] != ""]
    dup = m["community"][m["community"].duplicated()]
    if len(dup):
        raise ValueError(f"{path}: duplicate community {sorted(set(dup))}")
    return m[["community", "primary_school", "middle_school", "tier", "note"]]


def attach_school(df: pd.DataFrame, school_map: pd.DataFrame) -> pd.DataFrame:
    """按小区名挂上 school_primary / school_middle / school_tier。"""
    m = school_map.rename(columns={"primary_school": "school_primary",
                                   "middle_school": "school_middle",
                                   "tier": "school_tier",
                                   "note": "school_note"})
    out = df.merge(m, on="community", how="left")
    out["school_tier"] = out["school_tier"].fillna(UNMAPPED_TIER)
    for col in ("school_primary", "school_middle", "school_note"):
        out[col] = out[col].fillna("")
    return out


def _deal(r: pd.Series) -> dict[str, Any]:
    return {
        "community": r["community"], "period": r["period"],
        "total": float(r["total_price_wan"]), "sqm": float(r["area_sqm"]),
        "unit": float(r["unit_price_wan_sqm"]), "room_type": r["room_type"],
        "date": str(r["deal_date"].date()),
        "note": r.get("school_note", ""),
    }


def analyze_by_school(df: pd.DataFrame) -> list[dict[str, Any]]:
    """按对口小学聚合：上车门槛（最低总价）、2室+最低总价、近期最低价、中位数，及流通性。

    流通性：月均成交笔数（一期 = 半个月）、有成交的期数占比、平均谈价率。
    共享/大学区的小区同时计入其对口的每所学校。名校在前，其余按成交笔数降序。
    """
    rows = df[df["school_primary"] != ""].copy()
    if rows.empty:
        return []
    rows["school"] = rows["school_primary"].str.split("/")
    rows = rows.explode("school")
    periods = sorted(df["period"].unique())
    recent = set(periods[-SCHOOL_RECENT_PERIODS:])

    out = []
    for school, g in rows.groupby("school"):
        g2 = g[~g["room_type"].isin(["1室", "-"])]
        gr = g[g["period"].isin(recent)]
        comm = (g.groupby("community")
                 .agg(n=("community", "size"), min_total=("total_price_wan", "min"),
                      note=("school_note", "first"))
                 .sort_values(["n", "min_total"], ascending=[False, True]))
        nr = g["negotiation_rate"].dropna()
        n_active = g["period"].nunique()
        out.append({
            "school": school,
            "is_top": school in TOP_SCHOOLS,
            "n": len(g),
            "n_communities": len(comm),
            "per_month": len(g) / (len(periods) / 2),
            "n_active_periods": n_active,
            "n_periods": len(periods),
            "active_ratio": n_active / len(periods),
            "avg_neg_rate": float(nr.mean()) if len(nr) else None,
            "cheapest": _deal(g.loc[g["total_price_wan"].idxmin()]),
            "cheapest_2r": _deal(g2.loc[g2["total_price_wan"].idxmin()]) if len(g2) else None,
            "cheapest_recent": _deal(gr.loc[gr["total_price_wan"].idxmin()]) if len(gr) else None,
            "median_total": float(g["total_price_wan"].median()),
            "median_unit": float(g["unit_price_wan_sqm"].median()),
            "communities": [{"community": c, "n": int(r.n), "min_total": float(r.min_total),
                             "note": r.note}
                            for c, r in comm.iterrows()],
        })
    out.sort(key=lambda s: (not s["is_top"], -s["n"]))
    return out


# 低总价段：单独统计 LOW_PRICE_MAX 以下的成交（按片区 / 小区）
LOW_PRICE_MAX = 200          # 万
LOW_PRICE_MIN_DEALS = 5      # 小区表只列成交 ≥ 此值的小区


def analyze_low_price(df: pd.DataFrame, max_total: float = LOW_PRICE_MAX,
                      min_deals: int = LOW_PRICE_MIN_DEALS) -> dict[str, Any]:
    """总价 < max_total 的成交，按片区、按小区聚合。

    片区：笔数、小区数、对口名校笔数、总价 / 单价 / 面积中位、平均谈价率、月均成交（一期 = 半月）。
    小区：只列 ≥ min_deals 笔的，带对口小学与学区档；其余小区计入 n_hidden_*。
    df 需已 attach_school（缺学区列时按未标注处理）。
    """
    periods = sorted(df["period"].unique())
    months = len(periods) / 2 if periods else 1
    lo = df[df["total_price_wan"] < max_total].copy()
    for col, default in (("school_tier", UNMAPPED_TIER), ("school_primary", "")):
        if col not in lo:
            lo[col] = default
    out: dict[str, Any] = {"max_total": max_total, "min_deals": min_deals, "n": len(lo),
                           "n_all": len(df), "areas": [], "communities": [],
                           "n_hidden_communities": 0, "n_hidden_deals": 0}
    if lo.empty:
        return out

    def neg(g: pd.DataFrame) -> float | None:
        nr = g["negotiation_rate"].dropna()
        return float(nr.mean()) if len(nr) else None

    for area, g in lo.groupby("area"):
        n_top = int((g["school_tier"] == "名校").sum())
        out["areas"].append({
            "area": area, "n": len(g), "n_communities": int(g["community"].nunique()),
            "n_top": n_top, "top_share": n_top / len(g),
            "median_total": float(g["total_price_wan"].median()),
            "median_unit": float(g["unit_price_wan_sqm"].median()),
            "median_sqm": float(g["area_sqm"].median()),
            "avg_neg_rate": neg(g), "per_month": len(g) / months,
        })
    out["areas"].sort(key=lambda r: (-r["n"], r["area"]))

    for (area, comm), g in lo.groupby(["area", "community"]):
        if len(g) < min_deals:
            out["n_hidden_communities"] += 1
            out["n_hidden_deals"] += len(g)
            continue
        cheapest = g.loc[g["total_price_wan"].idxmin()]
        out["communities"].append({
            "community": comm, "area": area, "n": len(g),
            "school_primary": g["school_primary"].iloc[0], "tier": g["school_tier"].iloc[0],
            "min_total": float(cheapest["total_price_wan"]), "min_sqm": float(cheapest["area_sqm"]),
            "median_total": float(g["total_price_wan"].median()),
            "median_sqm": float(g["area_sqm"].median()),
            "one_room_share": float((g["room_type"] == "1室").mean()),
            "n_active_periods": int(g["period"].nunique()), "n_periods": len(periods),
            "avg_neg_rate": neg(g), "per_month": len(g) / months,
        })
    out["communities"].sort(key=lambda r: (-r["n"], r["min_total"]))
    return out


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """返回 (coef, stderr, dof)。经典 OLS 标准误。"""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - np.linalg.matrix_rank(X)
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    return coef, np.sqrt(np.diag(cov)), dof


def analyze_school_premium(df: pd.DataFrame) -> dict[str, Any]:
    """学区溢价：分档概况 + 同片区对比 + 控制变量回归。

    回归：log(单价) ~ 学区档(基准=普通) + 片区 + 期次 + 房龄 + log(面积) + 户型。
    溢价 = exp(β) − 1。只用已标注学区档的样本。
    """
    n_total = len(df)
    mapped = df[df["school_tier"] != UNMAPPED_TIER].copy()
    result: dict[str, Any] = {
        "n_total": n_total, "n_mapped": len(mapped),
        "by_tier": [], "same_area": [], "same_area_median_pct": None,
        "regression": None,
    }
    if mapped.empty:
        return result

    for tier in SCHOOL_TIERS:
        g = mapped[mapped["school_tier"] == tier]
        if g.empty:
            continue
        nr = g["negotiation_rate"].dropna()
        result["by_tier"].append({
            "tier": tier, "n": len(g),
            "n_communities": g["community"].nunique(),
            "median_unit_price": float(g["unit_price_wan_sqm"].median()),
            "avg_neg_rate": float(nr.mean()) if len(nr) else None,
        })

    # 同片区：名校 vs 普通 单价中位数
    for area, g in mapped.groupby("area"):
        top = g[g["school_tier"] == "名校"]["unit_price_wan_sqm"]
        base = g[g["school_tier"] == "普通"]["unit_price_wan_sqm"]
        if len(top) >= SCHOOL_MIN_N and len(base) >= SCHOOL_MIN_N:
            result["same_area"].append({
                "area": area, "n_top": len(top), "n_base": len(base),
                "median_top": float(top.median()), "median_base": float(base.median()),
                "premium_pct": float((top.median() / base.median() - 1) * 100),
            })
    if result["same_area"]:
        result["same_area_median_pct"] = float(
            np.median([r["premium_pct"] for r in result["same_area"]]))

    # 回归
    reg = mapped.dropna(subset=["year_built"])
    tiers_present = [t for t in ("名校", "无学位") if (reg["school_tier"] == t).sum() >= SCHOOL_MIN_N]
    if (reg["school_tier"] == "普通").sum() < SCHOOL_MIN_N or not tiers_present or len(reg) < 30:
        return result
    ref_year = int(reg["deal_date"].dt.year.max())
    X = pd.DataFrame({"const": 1.0,
                      "age": ref_year - reg["year_built"].astype(float),
                      "log_sqm": np.log(reg["area_sqm"].astype(float))},
                     index=reg.index)
    for t in tiers_present:
        X[f"tier_{t}"] = (reg["school_tier"] == t).astype(float)
    for col, prefix in (("area", "area"), ("period", "period")):
        X = X.join(pd.get_dummies(reg[col], prefix=prefix, drop_first=True, dtype=float))
    X = X.join(pd.get_dummies(reg["room_type"].map(_room_type_group), prefix="rt",
                              drop_first=True, dtype=float))
    y = np.log(reg["unit_price_wan_sqm"].astype(float).to_numpy())
    coef, se, dof = _ols(y, X.to_numpy())
    resid = y - X.to_numpy() @ coef
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    tcrit = _stats.t.ppf(0.975, dof)
    terms = []
    for t in tiers_present:
        i = list(X.columns).index(f"tier_{t}")
        b, s = coef[i], se[i]
        terms.append({
            "tier": t,
            "premium_pct": float((np.exp(b) - 1) * 100),
            "ci_low_pct": float((np.exp(b - tcrit * s) - 1) * 100),
            "ci_high_pct": float((np.exp(b + tcrit * s) - 1) * 100),
            "p_value": float(2 * _stats.t.sf(abs(b / s), dof)),
        })
    i_age = list(X.columns).index("age")
    result["regression"] = {
        "n": len(reg), "r2": float(r2), "terms": terms,
        "age_pct_per_year": float((np.exp(coef[i_age]) - 1) * 100),
    }
    return result


def detect_outliers(df: pd.DataFrame, ratio: float = OUTLIER_RATIO) -> pd.Series:
    """标记非市场价（单价 < 片区中位数 × ratio）。

    片区中位数用全期合并数据算（少量 outlier 不会拉低 median）。
    注意：跨度 ≤ 1 年时此方法稳健；若未来做年比/多年分析且片区均价
    整体已大幅迁移（>30%），应改用 per-period median 避免误判。
    返回布尔 Series，True = outlier。
    """
    area_median = df.groupby("area")["unit_price_wan_sqm"].transform("median")
    return df["unit_price_wan_sqm"] < area_median * ratio


def filter_outliers(
    df: pd.DataFrame, ratio: float = OUTLIER_RATIO
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (clean_df, outliers_df)，两者均为新拷贝（不改原 df）。"""
    mask = detect_outliers(df, ratio)
    return df[~mask].copy().reset_index(drop=True), df[mask].copy().reset_index(drop=True)


def analyze_extremes(df: pd.DataFrame) -> dict[str, Any]:
    """spec §4.1: -20/-30/-50% 阈值笔数 + 楼盘明细 + 最大跌幅记录。"""
    sub = df.dropna(subset=["negotiation_rate"]).copy()
    sub = sub.sort_values("negotiation_rate", ascending=True)

    counts: dict[int, int] = {}
    listings: dict[int, list[dict]] = {}
    for t in THRESHOLDS:
        hits = sub[sub["negotiation_rate"] <= t]
        counts[t] = len(hits)
        listings[t] = hits.to_dict("records")

    max_drop = sub.iloc[0].to_dict() if len(sub) else {}
    return {"counts": counts, "listings": listings, "max_drop": max_drop, "n": len(sub)}


def analyze_distribution(df: pd.DataFrame) -> dict[str, int]:
    """spec §4.2: 六桶分布（仅 negotiation_rate 非空子集）。"""
    sub = df.dropna(subset=["negotiation_rate"])
    return {label: int(sub["negotiation_rate"].apply(pred).sum())
            for label, pred in DIST_BUCKETS}


def analyze_by_area(
    df: pd.DataFrame,
    period_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """spec §4.3: 片区 × 期 聚合 + 任意期对的变化%。

    period_pairs: 要计算的 (earlier, later) 对；默认按字典序生成所有相邻对。
    返回 deltas 是 dict[(earlier, later)] -> list[row]。
    """
    per_period_rows = []
    for (area, period), group in df.groupby(["area", "period"]):
        n = len(group)
        per_period_rows.append({
            "area": area,
            "period": period,
            "n": n,
            "avg_unit_price": float(group["unit_price_wan_sqm"].mean()),
            "median_unit_price": float(group["unit_price_wan_sqm"].median()),
            "avg_neg_rate": (
                float(group["negotiation_rate"].mean())
                if group["negotiation_rate"].notna().any() else None
            ),
            "sparse": n < 3,
        })

    if period_pairs is None:
        periods = sorted(df["period"].unique())
        period_pairs = list(zip(periods[:-1], periods[1:]))

    pivot = (df.groupby(["area", "period"])["unit_price_wan_sqm"]
             .agg(["mean", "median"]).unstack("period"))
    n_pivot = df.groupby(["area", "period"]).size().unstack("period").fillna(0).astype(int)

    def pct(e, l):
        if pd.isna(e) or pd.isna(l) or e == 0:
            return None
        return (l - e) / e * 100.0

    deltas: dict[tuple[str, str], list[dict]] = {}
    for earlier, later in period_pairs:
        rows = []
        for area in pivot.index:
            e_mean = pivot["mean"].loc[area].get(earlier, float("nan"))
            l_mean = pivot["mean"].loc[area].get(later, float("nan"))
            e_med = pivot["median"].loc[area].get(earlier, float("nan"))
            l_med = pivot["median"].loc[area].get(later, float("nan"))
            n_e = int(n_pivot.loc[area].get(earlier, 0))
            n_l = int(n_pivot.loc[area].get(later, 0))
            rows.append({
                "area": area,
                "earlier": earlier,
                "later": later,
                "n_earlier": n_e,
                "n_later": n_l,
                "earlier_avg": None if pd.isna(e_mean) else float(e_mean),
                "later_avg": None if pd.isna(l_mean) else float(l_mean),
                "avg_unit_price_change_pct": pct(e_mean, l_mean),
                "median_unit_price_change_pct": pct(e_med, l_med),
            })
        deltas[(earlier, later)] = rows

    return {"per_period": per_period_rows, "deltas": deltas, "period_pairs": period_pairs}


def classify_area_delta(
    avg_pct: float | None,
    med_pct: float | None,
    n_earlier: int,
    n_later: int,
    min_n: int = 5,
    strong_n: int = 10,
    flat_band: float = 3.0,
    sign_threshold: float = 2.0,
    gap_threshold: float = 20.0,
) -> tuple[str, str]:
    """给片区跨期涨跌分类一个 (label, reason)。

    规则（按优先级）：
    - 任一期 < min_n 笔 → 样本不足
    - 平均/中位方向矛盾（|x| > sign_threshold 才算有方向）→ 采样偏差
    - 平均/中位差距 > gap_threshold pp → 采样偏差（极端值/品种偏移）
    - |平均| < flat_band → 持平
    - 否则按平均符号判涨/跌；两期都 ≥ strong_n 标"强"，否则"中"
    """
    if n_earlier < min_n or n_later < min_n:
        return ("🚫 样本不足",
                f"任一期 < {min_n} 笔（{n_earlier}/{n_later}）")
    if avg_pct is None or med_pct is None:
        return ("🚫 数据不足", "无法计算变化%")

    avg_dir = 1 if avg_pct > sign_threshold else (-1 if avg_pct < -sign_threshold else 0)
    med_dir = 1 if med_pct > sign_threshold else (-1 if med_pct < -sign_threshold else 0)

    if avg_dir != 0 and med_dir != 0 and avg_dir != med_dir:
        return ("⚠️ 采样偏差", "平均/中位方向矛盾（品种偏移）")
    if abs(avg_pct - med_pct) > gap_threshold:
        return ("⚠️ 采样偏差",
                f"平均/中位差距 {abs(avg_pct - med_pct):.1f}pp，极端值或品种偏移")

    is_strong = n_earlier >= strong_n and n_later >= strong_n
    tier = "强" if is_strong else "中"

    if abs(avg_pct) < flat_band:
        return ("➡️ 持平", f"平均变化 {avg_pct:+.1f}% 在 ±{flat_band}% 内")
    if avg_pct >= flat_band:
        return (f"📈 涨（{tier}）",
                f"平均 {avg_pct:+.1f}%、中位 {med_pct:+.1f}% 同向")
    return (f"📉 跌（{tier}）",
            f"平均 {avg_pct:+.1f}%、中位 {med_pct:+.1f}% 同向")


def _room_type_group(rt: str) -> str:
    if rt in ("1室", "2室", "3室"):
        return rt
    return "4室及以上"


def analyze_by_room_type(df: pd.DataFrame) -> list[dict[str, Any]]:
    """spec §4.4: 户型 × 期 聚合（4+ 合并）。"""
    df2 = df.assign(room_type_group=df["room_type"].map(_room_type_group))
    rows = []
    for (rt, period), group in df2.groupby(["room_type_group", "period"]):
        rows.append({
            "room_type_group": rt,
            "period": period,
            "n": len(group),
            "avg_unit_price": float(group["unit_price_wan_sqm"].mean()),
            "avg_area_sqm": float(group["area_sqm"].mean()),
            "avg_neg_rate": (
                float(group["negotiation_rate"].mean())
                if group["negotiation_rate"].notna().any() else None
            ),
        })
    return rows


def total_price_bucket(p: float) -> str | None:
    """总价段标签；≥ PRICE_ANALYSIS_MAX 不归档，返回 None。"""
    for label, pred in TOTAL_PRICE_BUCKETS:
        if pred(p):
            return label
    if p >= PRICE_ANALYSIS_MAX:
        return None
    raise ValueError(f"unbucketed: {p}")


def analyze_by_total_price(df: pd.DataFrame) -> dict[str, Any]:
    """spec §4.5: 总价段聚合 + 总价 vs 谈价率相关性假设检验（只看 PRICE_ANALYSIS_MAX 以下）。"""
    n_excluded = int((df["total_price_wan"] >= PRICE_ANALYSIS_MAX).sum())
    df = df[df["total_price_wan"] < PRICE_ANALYSIS_MAX]
    df2 = df.assign(_bucket=df["total_price_wan"].map(total_price_bucket))

    bucket_rows = []
    for label in TOTAL_PRICE_LABELS:
        group = df2[df2["_bucket"] == label]
        nr = group["negotiation_rate"].dropna()
        n_total = len(group)
        bucket_rows.append({
            "bucket": label,
            "n": n_total,
            "n_with_nr": len(nr),
            "avg_unit_price": float(group["unit_price_wan_sqm"].mean()) if n_total else None,
            "avg_total_price": float(group["total_price_wan"].mean()) if n_total else None,
            "avg_neg_rate": float(nr.mean()) if len(nr) else None,
            "big_drop_count": int((nr <= -20).sum()),
            "big_drop_rate": (float((nr <= -20).sum()) / len(nr)) if len(nr) else None,
        })

    sub = df.dropna(subset=["negotiation_rate"])
    n = len(sub)
    if n >= 3:
        pr, pp = _stats.pearsonr(sub["total_price_wan"], sub["negotiation_rate"])
        sr, sp = _stats.spearmanr(sub["total_price_wan"], sub["negotiation_rate"])
        if pr > 0 and pp < 0.05:
            conclusion = (f"**支持假设**：总价越低、折让越深 "
                          f"(Pearson r={pr:.3f}, p={pp:.4g}, n={n})")
        else:
            conclusion = (f"**未发现显著相关**：Pearson r={pr:.3f}, p={pp:.4g}, n={n}")
    else:
        pr = pp = sr = sp = None
        conclusion = f"样本不足 (n={n})"

    return {
        "n_excluded": n_excluded,
        "buckets": bucket_rows,
        "correlation": {
            "n": n, "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp,
            "conclusion": conclusion,
        },
    }


def analyze_same_community_room_delta(
    df: pd.DataFrame,
    earlier: str | None = None,
    later: str | None = None,
) -> list[dict[str, Any]]:
    """spec §4.7: 跨期同小区同户型对比（消除品种偏移）。

    earlier / later: 默认取数据中最早/最晚两期。
    返回每个 (community, room_type) 同时在两期都有成交的对子。
    """
    if earlier is None or later is None:
        periods = sorted(df["period"].unique())
        earlier = earlier or periods[0]
        later = later or periods[-1]

    rows = []
    for (community, rt), group in df.groupby(["community", "room_type"]):
        per_period = {}
        for p, sub in group.groupby("period"):
            per_period[p] = {
                "n": len(sub),
                "avg_unit_price": float(sub["unit_price_wan_sqm"].mean()),
                "avg_total_price": float(sub["total_price_wan"].mean()),
                "avg_area": float(sub["area_sqm"].mean()),
                "avg_neg_rate": (
                    float(sub["negotiation_rate"].mean())
                    if sub["negotiation_rate"].notna().any() else None
                ),
            }
        if earlier not in per_period or later not in per_period:
            continue
        e = per_period[earlier]
        l = per_period[later]
        change_pct = (l["avg_unit_price"] - e["avg_unit_price"]) / e["avg_unit_price"] * 100
        rows.append({
            "community": community,
            "room_type": rt,
            "area": group["area"].iloc[0],
            "year_built": (int(group["year_built"].iloc[0])
                           if pd.notna(group["year_built"].iloc[0]) else None),
            "n_earlier": e["n"],
            "n_later": l["n"],
            "earlier_period": earlier,
            "later_period": later,
            "earlier_avg_unit": e["avg_unit_price"],
            "later_avg_unit": l["avg_unit_price"],
            "change_pct": change_pct,
            "earlier_avg_area": e["avg_area"],
            "later_avg_area": l["avg_area"],
        })
    rows.sort(key=lambda x: x["change_pct"])
    return rows


def predict_next_period(
    df: pd.DataFrame,
    periods: list[str],
    min_periods_with_data: int = 3,
    min_n_per_period: int = 5,
) -> list[dict[str, Any]]:
    """对每个核心片区做线性回归预测下期均价 + 95% 置信带。

    核心片区 = 至少 min_periods_with_data 个期都有 ≥ min_n_per_period 笔。
    回归基于"片区 × 期"的平均单价；x = 期索引 0..n-1，预测 x = n。

    `current` 是该片区**最后一个合格期**（≥ min_n_per_period 笔）的均价，
    可能不是 periods 的最末期（若最末期样本不足）。`current_period` 字段
    标识具体是哪一期。

    返回 list[{area, current, current_period, predicted, ci_low, ci_high,
                change_pct, slope, r2, n_periods, trend}]
    按 abs(change_pct) 降序（预测变化最剧烈的在前）。
    """
    from scipy import stats as _s

    n_pivot = df.groupby(["area", "period"]).size().unstack("period").fillna(0).astype(int)
    avg_pivot = df.groupby(["area", "period"])["unit_price_wan_sqm"].mean().unstack("period")

    out: list[dict[str, Any]] = []
    for area in n_pivot.index:
        # 取有 ≥ min_n_per_period 笔且非 NaN 的期
        xs, ys, qualifying_periods = [], [], []
        for i, p in enumerate(periods):
            count = int(n_pivot.loc[area].get(p, 0))
            v = avg_pivot.loc[area, p] if p in avg_pivot.columns else float("nan")
            if count >= min_n_per_period and not pd.isna(v):
                xs.append(i)
                ys.append(float(v))
                qualifying_periods.append(p)
        n = len(xs)
        if n < min_periods_with_data:
            continue
        slope, intercept, r, _p, _stderr = _s.linregress(xs, ys)
        x_next = len(periods)  # 预测下一期的索引
        y_pred = slope * x_next + intercept

        # 95% 预测区间
        y_fit = [slope * x + intercept for x in xs]
        residuals = [yi - yh for yi, yh in zip(ys, y_fit)]
        if n > 2:
            sse = sum(r_ * r_ for r_ in residuals)
            s_err = (sse / (n - 2)) ** 0.5
            x_mean = sum(xs) / n
            sxx = sum((x - x_mean) ** 2 for x in xs)
            se_pred = s_err * (1 + 1 / n + (x_next - x_mean) ** 2 / sxx) ** 0.5
            t_crit = float(_s.t.ppf(0.975, n - 2))
            margin = t_crit * se_pred
        else:
            margin = 0.0

        current = ys[-1]
        current_period = qualifying_periods[-1]
        change_pct = (y_pred - current) / current * 100 if current else 0
        if abs(change_pct) < 1.5:
            trend = "持平"
        elif change_pct > 0:
            trend = "上行"
        else:
            trend = "下行"

        out.append({
            "area": area,
            "current": round(current, 2),
            "current_period": current_period,
            "predicted": round(y_pred, 2),
            "ci_low": round(y_pred - margin, 2),
            "ci_high": round(y_pred + margin, 2),
            "change_pct": round(change_pct, 2),
            "slope": round(slope, 3),
            "r2": round(r * r, 3),
            "n_periods": n,
            "trend": trend,
        })

    out.sort(key=lambda r: -abs(r["change_pct"]))
    return out


def analyze_repeat_communities(df: pd.DataFrame) -> list[dict[str, Any]]:
    """spec §4.6: 两期合计 ≥2 笔的小区列表。"""
    counts = df.groupby("community").size()
    repeats = counts[counts >= 2].index.tolist()
    out = []
    for community in repeats:
        txns = df[df["community"] == community].sort_values("deal_date")
        out.append({
            "community": community,
            "n": len(txns),
            "transactions": txns[
                ["period", "area", "room_type", "area_sqm", "total_price_wan",
                 "unit_price_wan_sqm", "negotiation_rate", "deal_date"]
            ].to_dict("records"),
        })
    out.sort(key=lambda r: -r["n"])
    return out


def _fmt_drop_listing(r: dict[str, Any]) -> str:
    yb = int(r["year_built"]) if pd.notna(r["year_built"]) else "?"
    return (f"{r['negotiation_rate']:.2f}% | {r['area']} | "
            f"{r['community']}({yb}) | {r['room_type']} | "
            f"{r['area_sqm']:g}㎡ | {r['total_price_wan']:g}万 | "
            f"{r['unit_price_wan_sqm']:.2f}万/㎡ | {r['deal_date'].date()}")


def render_report(df: pd.DataFrame) -> str:
    periods = sorted(df["period"].unique())
    L: list[str] = []
    L.append(f"# 福田二手成交分析（{', '.join(periods)}）")
    L.append("")
    L.append(f"**数据规模：** 总成交 {len(df)} 笔；含谈价率 "
             f"{df['negotiation_rate'].notna().sum()} 笔。")
    for p in periods:
        sub = df[df["period"] == p]
        L.append(f"- {p}：{len(sub)} 笔（{sub['negotiation_rate'].notna().sum()} 有谈价率）")
    L.append("")

    # §4.1
    L.append("## §4.1 谈价率极值")
    L.append("")
    for p in periods:
        ext = analyze_extremes(df[df["period"] == p])
        L.append(f"### {p}（含谈价率 n={ext['n']}）")
        L.append("")
        for t in (-20, -30, -50):
            L.append(f"- 跌幅 ≤ **{t}%**：**{ext['counts'][t]} 笔**")
        if ext["max_drop"]:
            md = ext["max_drop"]
            yb = int(md["year_built"]) if pd.notna(md["year_built"]) else "?"
            L.append(f"- **最大跌幅：{md['negotiation_rate']:.2f}%** — "
                     f"{md['area']} / {md['community']}({yb}) "
                     f"{md['room_type']} {md['area_sqm']:g}㎡ "
                     f"{md['total_price_wan']:g}万 "
                     f"{md['unit_price_wan_sqm']:.2f}万/㎡ "
                     f"{md['deal_date'].date()}")
        for t in (-20, -30, -50):
            if ext["counts"][t]:
                L.append("")
                L.append(f"**跌幅 ≤ {t}% 明细：**")
                L.append("")
                L.append("| 谈价率 | 片区 | 小区 | 户型 | 面积 | 总价 | 单价 | 日期 |")
                L.append("|---|---|---|---|---|---|---|---|")
                for r in ext["listings"][t]:
                    yb = int(r["year_built"]) if pd.notna(r["year_built"]) else "?"
                    L.append(f"| {r['negotiation_rate']:.2f}% | {r['area']} | "
                             f"{r['community']}({yb}) | {r['room_type']} | "
                             f"{r['area_sqm']:g}㎡ | {r['total_price_wan']:g}万 | "
                             f"{r['unit_price_wan_sqm']:.2f} | {r['deal_date'].date()} |")
        L.append("")

    # §4.2
    L.append("## §4.2 谈价率分布")
    L.append("")
    L.append("| 桶 | " + " | ".join(periods) + " | 合计 |")
    L.append("|---|" + "---|" * (len(periods) + 1))
    bd = {p: analyze_distribution(df[df["period"] == p]) for p in periods}
    bd_total = analyze_distribution(df)
    for label, _ in DIST_BUCKETS:
        cells = " | ".join(str(bd[p][label]) for p in periods)
        L.append(f"| {label} | {cells} | {bd_total[label]} |")
    L.append("")

    # §4.3
    L.append("## §4.3 片区维度")
    L.append("")
    a = analyze_by_area(df)
    L.append("**每期聚合：**")
    L.append("")
    L.append("| 片区 | 期 | 笔数 | 平均单价 | 中位单价 | 平均谈价率 | 稀疏 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in sorted(a["per_period"], key=lambda x: (x["area"], x["period"])):
        nr = f"{r['avg_neg_rate']:.2f}%" if r["avg_neg_rate"] is not None else "—"
        L.append(f"| {r['area']} | {r['period']} | {r['n']} | "
                 f"{r['avg_unit_price']:.2f} | {r['median_unit_price']:.2f} | "
                 f"{nr} | {'⚠️' if r['sparse'] else ''} |")
    L.append("")
    label_priority = {
        "📉 跌（强）": 0, "📈 涨（强）": 1,
        "📉 跌（中）": 2, "📈 涨（中）": 3,
        "➡️ 持平": 4,
        "⚠️ 采样偏差": 5,
        "🚫 样本不足": 6, "🚫 数据不足": 7,
    }

    real_signal_by_pair: dict[tuple[str, str], dict[str, list]] = {}
    for (earlier, later), delta_rows in a["deltas"].items():
        classified = []
        for r in delta_rows:
            label, reason = classify_area_delta(
                r["avg_unit_price_change_pct"],
                r["median_unit_price_change_pct"],
                r["n_earlier"], r["n_later"],
            )
            classified.append({**r, "label": label, "reason": reason})
        classified.sort(key=lambda x: (
            label_priority.get(x["label"], 99),
            -abs(x["avg_unit_price_change_pct"] or 0),
        ))

        L.append(f"### {later} vs {earlier}")
        L.append("")
        L.append("| 类别 | 片区 | 早期笔数 | 晚期笔数 | 早期均价 | "
                 "晚期均价 | 平均变化% | 中位变化% | 依据 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in classified:
            if r["avg_unit_price_change_pct"] is None:
                continue
            ea = f"{r['earlier_avg']:.2f}" if r["earlier_avg"] is not None else "—"
            la = f"{r['later_avg']:.2f}" if r["later_avg"] is not None else "—"
            avg = f"{r['avg_unit_price_change_pct']:+.2f}%"
            med = (f"{r['median_unit_price_change_pct']:+.2f}%"
                   if r["median_unit_price_change_pct"] is not None else "—")
            L.append(f"| {r['label']} | {r['area']} | {r['n_earlier']} | "
                     f"{r['n_later']} | {ea} | {la} | {avg} | {med} | "
                     f"{r['reason']} |")
        L.append("")

        only_one_period = [r for r in delta_rows
                           if r["avg_unit_price_change_pct"] is None]
        if only_one_period:
            L.append(f"_仅一期有数据：_ "
                     + ", ".join(sorted(r["area"] for r in only_one_period)))
            L.append("")

        real_signal_by_pair[(earlier, later)] = {
            "down": [r for r in classified if r["label"].startswith("📉 跌")],
            "up": [r for r in classified if r["label"].startswith("📈 涨")],
        }

    # 真信号摘要总表（横跨所有期对）
    L.append("### 真信号摘要（剔除采样偏差/小样本后）")
    L.append("")
    for (earlier, later), sig in real_signal_by_pair.items():
        L.append(f"**{later} vs {earlier}：**")
        if sig["down"]:
            for r in sig["down"]:
                L.append(f"- 📉 **{r['area']}** {r['avg_unit_price_change_pct']:+.2f}% "
                         f"(平均) / {r['median_unit_price_change_pct']:+.2f}% (中位)，"
                         f"{r['n_earlier']}→{r['n_later']} 笔 — {r['label']}")
        if sig["up"]:
            for r in sig["up"]:
                L.append(f"- 📈 **{r['area']}** {r['avg_unit_price_change_pct']:+.2f}% "
                         f"(平均) / {r['median_unit_price_change_pct']:+.2f}% (中位)，"
                         f"{r['n_earlier']}→{r['n_later']} 笔 — {r['label']}")
        if not sig["down"] and not sig["up"]:
            L.append("- _无真信号（全部持平、偏差或样本不足）_")
        L.append("")

    # §4.4
    L.append("## §4.4 户型维度")
    L.append("")
    L.append("| 户型 | 期 | 笔数 | 平均单价 | 平均面积 | 平均谈价率 |")
    L.append("|---|---|---|---|---|---|")
    rt_rows = analyze_by_room_type(df)
    order = {"1室": 0, "2室": 1, "3室": 2, "4室及以上": 3}
    for r in sorted(rt_rows, key=lambda x: (order[x["room_type_group"]], x["period"])):
        nr = f"{r['avg_neg_rate']:.2f}%" if r["avg_neg_rate"] is not None else "—"
        L.append(f"| {r['room_type_group']} | {r['period']} | {r['n']} | "
                 f"{r['avg_unit_price']:.2f} | {r['avg_area_sqm']:.1f}㎡ | {nr} |")
    L.append("")

    # §4.5
    L.append("## §4.5 总价段维度（含假设检验）")
    L.append("")
    tp = analyze_by_total_price(df)
    L.append("**两期合并：**")
    L.append("")
    L.append("| 总价桶 | 笔数 | 含谈价率 | 平均单价 | 平均总价 | 平均谈价率 | 大跌(≤-20%)笔数 | 大跌占比 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in tp["buckets"]:
        ap = f"{r['avg_unit_price']:.2f}" if r["avg_unit_price"] else "—"
        atp = f"{r['avg_total_price']:.0f}" if r["avg_total_price"] else "—"
        nr = f"{r['avg_neg_rate']:.2f}%" if r["avg_neg_rate"] is not None else "—"
        bdr = f"{r['big_drop_rate']*100:.1f}%" if r["big_drop_rate"] is not None else "—"
        L.append(f"| {r['bucket']} | {r['n']} | {r['n_with_nr']} | {ap} | "
                 f"{atp} | {nr} | {r['big_drop_count']} | {bdr} |")
    L.append("")
    cor = tp["correlation"]
    L.append("**假设检验**（用户假设：总价越低 → 折让越深；折让用负数 → "
             "若假设成立，总价 vs 谈价率应为**正相关** r > 0）")
    L.append("")
    if cor["pearson_r"] is not None:
        L.append(f"- Pearson r = **{cor['pearson_r']:.4f}**, "
                 f"p = {cor['pearson_p']:.4g}, n = {cor['n']}")
        L.append(f"- Spearman ρ = **{cor['spearman_r']:.4f}**, "
                 f"p = {cor['spearman_p']:.4g}")
    L.append(f"- **结论：{cor['conclusion']}**")
    L.append("")

    # §4.7 同小区同户型跨期对比（对每个相邻期对都做一次，再加首末对比）
    L.append("## §4.7 同小区同户型跨期对比（消除品种偏移）")
    L.append("")
    L.append("> 同一小区+户型在两期都有成交才会出现在表里。"
             "笔数 ≥ 2 的更可信；1+1 比较仍受面积差异影响。")
    L.append("")

    pair_jobs: list[tuple[str, str, str]] = []  # (title, earlier, later)
    for e, l in zip(periods[:-1], periods[1:]):
        pair_jobs.append((f"{l} vs {e}（相邻期）", e, l))
    if len(periods) >= 3:
        pair_jobs.append((f"{periods[-1]} vs {periods[0]}（首末对比）",
                          periods[0], periods[-1]))

    for title, earlier, later in pair_jobs:
        L.append(f"### {title}")
        L.append("")
        pairs = analyze_same_community_room_delta(df, earlier, later)
        if not pairs:
            L.append("_无数据。_")
            L.append("")
            continue
        avg_change = sum(p["change_pct"] for p in pairs) / len(pairs)
        n_down = sum(1 for p in pairs if p["change_pct"] < -1)
        n_up = sum(1 for p in pairs if p["change_pct"] > 1)
        n_flat = len(pairs) - n_down - n_up
        # 仅笔数 ≥2 的高可信子集
        thick = [p for p in pairs if p["n_earlier"] >= 2 and p["n_later"] >= 2]
        thick_str = ""
        if thick:
            thick_avg = sum(p["change_pct"] for p in thick) / len(thick)
            thick_str = (f"；其中两期都 ≥2 笔的高可信子集 **{len(thick)} 对**，"
                         f"平均 **{thick_avg:+.2f}%**")
        L.append(f"**对子数：{len(pairs)}**；跌 {n_down} / 涨 {n_up} / 持平 {n_flat}；"
                 f"整体平均 **{avg_change:+.2f}%**{thick_str}。")
        L.append("")
        L.append(f"| 小区(年份) | 片区 | 户型 | {earlier} 笔数 | {later} 笔数 | "
                 f"{earlier} 均价 | {later} 均价 | **变化%** | "
                 f"{earlier} 均面积 | {later} 均面积 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in pairs:
            yb = f"({r['year_built']})" if r["year_built"] else ""
            mark = "✅" if r["n_earlier"] >= 2 and r["n_later"] >= 2 else ""
            change_str = f"**{r['change_pct']:+.2f}%** {mark}".strip()
            L.append(f"| {r['community']}{yb} | {r['area']} | {r['room_type']} | "
                     f"{r['n_earlier']} | {r['n_later']} | "
                     f"{r['earlier_avg_unit']:.2f} | {r['later_avg_unit']:.2f} | "
                     f"{change_str} | {r['earlier_avg_area']:.0f}㎡ | "
                     f"{r['later_avg_area']:.0f}㎡ |")
        L.append("")

    # §4.6
    L.append("## §4.6 重点楼盘（两期合计 ≥2 笔，按笔数降序）")
    L.append("")
    repeats = analyze_repeat_communities(df)
    L.append(f"共 **{len(repeats)}** 个小区两期合计成交 ≥2 笔。")
    L.append("")
    for r in repeats:
        L.append(f"### {r['community']}（{r['n']} 笔）")
        L.append("")
        L.append("| 期 | 片区 | 户型 | 面积 | 总价 | 单价 | 谈价率 | 日期 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for t in r["transactions"]:
            nr = (f"{t['negotiation_rate']:.2f}%"
                  if pd.notna(t["negotiation_rate"]) else "—")
            L.append(f"| {t['period']} | {t['area']} | {t['room_type']} | "
                     f"{t['area_sqm']:g}㎡ | {t['total_price_wan']:g}万 | "
                     f"{t['unit_price_wan_sqm']:.2f} | {nr} | "
                     f"{t['deal_date'].date()} |")
        L.append("")

    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="福田二手成交分析")
    parser.add_argument(
        "--include-outliers", action="store_true",
        help="包含非市场价（单价 < 片区中位数 × 0.5 的记录），默认剔除",
    )
    args = parser.parse_args()

    here = Path(__file__).parent
    csvs = sorted((here / "data").glob("*.csv"))
    if not csvs:
        raise SystemExit("No CSVs in data/")
    df_raw, auctions = split_auctions(load_data(csvs))
    if len(auctions):
        print(f"剔除 {len(auctions)} 条法拍")

    if args.include_outliers:
        df = df_raw
        print(f"Loaded {len(df)} rows (含 outlier)")
    else:
        df, outliers = filter_outliers(df_raw)
        print(f"Loaded {len(df_raw)} rows, 剔除 {len(outliers)} 条非市场价 → {len(df)} 条用于分析")
        if len(outliers):
            print("剔除明细：")
            for _, r in outliers.iterrows():
                print(f"  {r['period']} {r['area']}/{r['community']} "
                      f"{r['area_sqm']:g}㎡ {r['total_price_wan']:g}万 "
                      f"→ {r['unit_price_wan_sqm']:.2f}万/㎡")

    print(f"Periods: {sorted(df['period'].unique())}")
    print(f"Negotiation-rate non-null: {df['negotiation_rate'].notna().sum()}")

    report = render_report(df)
    periods = sorted(df['period'].unique())
    out_name = f"福田_{periods[0]}_to_{periods[-1]}.md"
    out = here / "reports" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nReport written: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
