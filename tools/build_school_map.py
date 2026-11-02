"""用 ref/school_lists.json（各校对口小区原文）自动填充 ref/school_map.csv。

- 匹配：小区名与名单项相同，或一方是另一方前缀（≥3 字），或去掉“花园/小区/大厦/N期”等后缀后相同。
- source 列以“人工”开头的行视为人工确认，不会被覆盖。
- 学区档：命中 TOP_PRIMARY 任一学校 → 名校；命中其他小学 → 普通；没命中 → 留空（未标注）。
- 名单来自本地宝转载的福田教育网 2023 年对口小区，划片逐年调整，结果需人工复核。

用法：python3 tools/build_school_map.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LISTS = HERE / "ref" / "school_lists.json"
MAP = HERE / "ref" / "school_map.csv"

# 名校（草案，待确认）：社会公认的头部公办小学及集团校区
TOP_PRIMARY = {
    "深圳实验学校小学部", "深圳外国语小学", "百花小学", "园岭小学",
    "荔园小学(通新岭)", "荔园小学(百花)", "荔园小学(玮鹏)", "荔园小学(八卦岭)",
    "荔园外国语小学", "荔园外国语小学(天骄)", "荔园外国语小学(狮岭)",
    "福田区外国语小学", "红岭实验小学", "莲花小学",
}

# 名单无法按名称匹配、但页面原文能确认的情况
OVERRIDES = {
    # 2025 益田村片区大学区：福强小学、全海小学、益田小学（全海仅含 44-82 栋）
    "益田村": (["福强小学", "全海小学", "益田小学"], "2025 益田村片区大学区（show977861）"),
}

# 只有部分楼栋对口的小区，备注提示（成交记录无楼栋信息，无法区分）
PARTIAL = {
    "国城花园": "仅 1、2 栋对口深圳实验学校小学部",
    "园岭新村": "1-62、92-97 栋对口园岭小学",
}

SUFFIX = re.compile(r"(一期|二期|三期|四期|\d期|[A-D]区|小区|花园|家园|大厦|公寓)$")


def tokens(raw: str) -> list[str]:
    raw = re.sub(r"(属于共享学区。?|对口小区[:：]?|业主子女)", "", raw)
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


def main() -> None:
    lists = json.loads(LISTS.read_text(encoding="utf-8"))
    index: dict[str, list[tuple[str, str, str]]] = {}
    for s in lists:
        for t in tokens(s["raw"]):
            index.setdefault(t, []).append((s["level"], s["school"], s["id"]))

    with MAP.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())

    for r in rows:
        if r["source"].startswith("人工"):
            continue
        c = r["community"]
        hits = {h for t, hs in index.items() if matches(c, t) for h in hs}
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

    with MAP.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    total = sum(int(r["n_deals"]) for r in rows)
    covered = sum(int(r["n_deals"]) for r in rows if r["tier"])
    print(f"tiered communities: {sum(1 for r in rows if r['tier'])}/{len(rows)}, "
          f"deals covered {covered}/{total} ({covered / total:.0%})")


if __name__ == "__main__":
    main()
