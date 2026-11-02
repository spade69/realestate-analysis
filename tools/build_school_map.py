"""用 ref/school_lists.json（各校对口小区原文）自动填充 ref/school_map.csv。

- 行：按 data/*.csv 重新统计小区与成交笔数（不含法拍，小区名经 analyze.COMMUNITY_ALIASES 统一），
  已有行的学校 / 档位保留，新小区追加，按笔数降序。
- 匹配：小区名与名单项相同，或一方是另一方前缀（≥3 字），或去掉“花园/小区/大厦/N期”等后缀后相同；
  小区名去掉开发商前缀（京基、中洲、万科……）后再比一次；名单项括号里的别名也算一项；
  ALIASES 处理名单写法和成交写法差太多的情况（如「下沙花好园」↔「花好园」）。
- source 列以“人工”开头的行视为人工确认，不会被覆盖。
- 学区档：命中 analyze.TOP_SCHOOLS 任一学校 → 名校；命中其他小学 → 普通；
  任何名单（含初中）都没命中、名字像商住（公寓/公馆/商业中心……）且成交以一房小户型为主 → 无学位（规则推定）；
  其余留空（未标注）。
- 名单来自本地宝转载的福田教育网 2023 年对口小区，划片逐年调整，结果需人工复核。

用法：python3 tools/build_school_map.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from analyze import TOP_SCHOOLS, load_data, split_auctions  # noqa: E402

DATA = HERE / "data"
LISTS = HERE / "ref" / "school_lists.json"
MAP = HERE / "ref" / "school_map.csv"

TOP_PRIMARY = set(TOP_SCHOOLS)

# 名单无法按名称匹配、但页面原文能确认的情况
OVERRIDES = {
    # 2025 益田村片区大学区：福强小学、全海小学、益田小学（全海仅含 44-82 栋）
    "益田村": (["福强小学", "全海小学", "益田小学"], "2025 益田村片区大学区（show977861）"),
}

# 成交小区名 → 名单里的写法（名称差异超出前缀 / 后缀规则）
ALIASES = {
    "花好园": "下沙花好园",          # 下沙小学
    "香蜜湖水榭花都": "水榭花都",    # 荔园外国语小学
}

# 开发商 / 品牌前缀：成交记录常带，名单常不带（京基御景华城 ↔ 御景华城）
BRAND = re.compile(r"^(京基|中洲|绿景|万科|深房|深物业|皇达|合正|阳基|南光|新天|皇庭|新世界|华润|招商|卓越|星河)")

# 无学位规则：名字关键词 + 一房占比 + 面积中位数，且不在任何名单里
NO_DEGREE_NAME = re.compile(r"公寓|公馆|商业中心|金融中心|SOHO|写字楼|酒店|大厦|广场")
NO_DEGREE_ONE_ROOM = 0.8
NO_DEGREE_MAX_SQM = 50
NO_DEGREE_MIN_DEALS = 2

# 只有部分楼栋对口的小区，备注提示（成交记录无楼栋信息，无法区分）
PARTIAL = {
    "国城花园": "仅 1、2 栋对口深圳实验学校小学部",
    "园岭新村": "1-62、92-97 栋对口园岭小学",
    "莲花北村": "1-36 栋对口景莲小学，莲花北村东对口新莲小学",
}

SUFFIX = re.compile(r"(一期|二期|三期|四期|\d期|[A-D]区|小区|花园|家园|大厦|公寓)$")


def tokens(raw: str) -> list[str]:
    raw = re.sub(r"(属于共享学区。?|对口小区[:：]?|业主子女)", "", raw)
    # 括号内的别名单独成项：南名仕苑(南光名仕苑)
    raw = re.sub(r"[(（]([^)）]{3,})[)）]", r"、\1、", raw)
    out = []
    for t in re.split(r"[，、,；;。\s]|\d+[:：]", raw):
        t = re.sub(r"[(（][^)）]*[)）]", "", t)
        t = re.sub(r"\d+[-—一至]?\d*(栋|幢|巷|号|座).*$", "", t)
        t = re.sub(r"(A|B|C|D|AB|BCD|ABC)座.*$", "", t)
        t = re.sub(r"(一期|二期|三期|四期|高层)$", "", t).strip()
        if len(t) >= 2:
            out.append(t)
    return out


def matches(community: str, token: str) -> bool:
    if community == token:
        return True
    if len(community) >= 3 and token.startswith(community):
        return True
    if len(token) >= 3 and community.startswith(token):
        return True
    a, b = SUFFIX.sub("", community), SUFFIX.sub("", token)
    return len(a) >= 3 and a == b


def find_hits(community: str, index: dict) -> set:
    """先用原名（及 ALIASES）匹配；没命中小学再用去品牌前缀名（≥3 字）兜底。

    去前缀名只作兜底：「南光紫荆苑」去掉前缀后会撞上景田的「紫荆苑」。
    """
    names = [community] + ([ALIASES[community]] if community in ALIASES else [])
    hits = {h for n in names for t, hs in index.items() if matches(n, t) for h in hs}
    bare = BRAND.sub("", community)
    if (not any(lv == "小学" for lv, _, _ in hits)
            and bare != community and len(SUFFIX.sub("", bare)) >= 3):
        hits |= {h for t, hs in index.items() if matches(bare, t) for h in hs}
    return hits


def looks_no_degree(community: str, deals) -> bool:
    if not NO_DEGREE_NAME.search(community) or len(deals) < NO_DEGREE_MIN_DEALS:
        return False
    return ((deals["room_type"] == "1室").mean() >= NO_DEGREE_ONE_ROOM
            and deals["area_sqm"].median() <= NO_DEGREE_MAX_SQM)


def refresh_rows(rows: list[dict], market) -> list[dict]:
    """按成交数据重算 n_deals，补新小区，去掉已无成交的非人工行。"""
    counts = market.groupby("community").size()
    areas = market.groupby("community")["area"].agg(lambda s: s.mode().iloc[0])
    old = {r["community"]: r for r in rows}
    out = []
    for c, n in counts.items():
        r = old.pop(c, None) or {"area": areas[c], "community": c, "primary_school": "",
                                 "middle_school": "", "tier": "", "source": "", "note": ""}
        r["n_deals"] = str(n)
        out.append(r)
    out += [r for r in old.values() if r["source"].startswith("人工")]
    out.sort(key=lambda r: (-int(r["n_deals"]), r["area"], r["community"]))
    return out


def main() -> None:
    lists = json.loads(LISTS.read_text(encoding="utf-8"))
    index: dict[str, list[tuple[str, str, str]]] = {}
    for s in lists:
        for t in tokens(s["raw"]):
            index.setdefault(t, []).append((s["level"], s["school"], s["id"]))

    with MAP.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    market, _ = split_auctions(load_data(sorted(DATA.glob("*.csv"))))
    rows = refresh_rows(rows, market)
    by_comm = dict(tuple(market.groupby("community")))

    for r in rows:
        if r["source"].startswith("人工"):
            continue
        c = r["community"]
        hits = find_hits(c, index)
        prim = sorted({s for lv, s, _ in hits if lv == "小学"})
        mid = sorted({s for lv, s, _ in hits if lv == "初中"})
        ids = sorted({i for _, _, i in hits})
        note = ""
        if c in OVERRIDES:
            prim, note = OVERRIDES[c]
        r["primary_school"] = "/".join(prim)
        r["middle_school"] = "/".join(mid)
        r["tier"] = ("名校" if set(prim) & TOP_PRIMARY else "普通") if prim else ""
        r["source"] = ("本地宝(福田教育网2023) " + " ".join(f"show{i}" for i in ids)) if hits else ""
        r["note"] = note or PARTIAL.get(c, "") or ("多校共享/分享学区" if len(prim) > 1 else "")
        if not hits and looks_no_degree(c, by_comm[c]):
            r["tier"] = "无学位"
            r["source"] = "规则推定"
            r["note"] = "商住名 + 一房小户型为主，未见于任何对口名单，待人工复核"

    with MAP.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    total = sum(int(r["n_deals"]) for r in rows)
    covered = sum(int(r["n_deals"]) for r in rows if r["tier"])
    print(f"tiered communities: {sum(1 for r in rows if r['tier'])}/{len(rows)}, "
          f"deals covered {covered}/{total} ({covered / total:.0%})")
    for tier in ("名校", "普通", "无学位"):
        print(f"  {tier}: {sum(int(r['n_deals']) for r in rows if r['tier'] == tier)}")


if __name__ == "__main__":
    main()
