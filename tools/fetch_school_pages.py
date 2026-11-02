"""抓本地宝学校对口小区页，追加 / 更新到 ref/school_lists.json。

本地宝站内搜索是 JS 渲染抓不到，只能按文章编号抓：
    python3 tools/fetch_school_pages.py 923313 923679 ...

正文取「导语」之后、「地址 / 联系电话 / 文章信息来源」之前的段落；首行若是
「XX招生范围/对口小区/学区划分：」这类标题行则丢掉。已存在的 id 会被覆盖。
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LISTS = HERE / "ref" / "school_lists.json"

TITLE_NOISE = re.compile(
    r"(- 深圳本地宝|\d{4}年?|福田区|哪些小区(可以|能够)?(就)?读|对口小区|学区划分(详情)?|招生范围|详细)")
HEADER_LINE = re.compile(r"(招生范围|对口小区|学区划分)[^，、,]*[:：]\s*$")
STOP_LINE = re.compile(r"地址|联系电话|文章信息来源|官方网址|学校网址|交通情况")


def page_text(raw_html: str) -> list[str]:
    body = re.sub(r"<script.*?</script>|<style.*?</style>", "", raw_html, flags=re.S)
    txt = html.unescape(re.sub(r"<[^>]+>", "\n", body))
    return [l.strip(" \t　") for l in txt.split("\n") if l.strip(" \t　")]


def parse(page_id: str, raw_html: str) -> dict:
    title = re.search(r"<title>([^<]*)</title>", raw_html)
    title = title.group(1) if title else ""
    lines = page_text(raw_html)
    start = next(i for i, l in enumerate(lines) if l.startswith("导语")) + 2
    seg = []
    for l in lines[start:]:
        if STOP_LINE.search(l):
            break
        seg.append(l)
    school = TITLE_NOISE.sub("", title).strip()
    # 去掉与校名相同的行和标题行
    seg = [l for l in seg if not HEADER_LINE.search(l) and school not in l]
    year = re.search(r"(20\d\d)年", "".join(lines[:start + 2]))
    return {
        "id": page_id,
        "url": f"https://m.bendibao.com/show{page_id}.html",
        "school": school,
        "level": "初中" if re.search(r"中学|初中", school) else "小学",
        "year_hint": year.group(1) if year else "",
        "raw": "".join(seg),
    }


def fetch(page_id: str) -> str:
    """本地宝连续请求会 403，退避重试。"""
    for wait in (0, 5, 15, 45):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(f"https://m.bendibao.com/show{page_id}.html", timeout=20) as h:
                return h.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            if e.code != 403:
                raise
    raise RuntimeError(f"show{page_id}: still 403 after retries")


def main(ids: list[str]) -> None:
    data = json.loads(LISTS.read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in data}
    for i in ids:
        rec = parse(i, fetch(i))
        time.sleep(1)
        print(f"{i} {rec['level']} {rec['school']} {rec['year_hint']}: {rec['raw'][:60]}…")
        by_id[i] = rec
    out = sorted(by_id.values(), key=lambda s: s["id"])
    LISTS.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
