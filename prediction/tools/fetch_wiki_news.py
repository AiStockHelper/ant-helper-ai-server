"""위키피디아 'Portal:Current events' 일자별 뉴스(영문)를 data/news_wiki.sqlite 에 모은다.

날짜별 페이지는 그날 일어난 일만 담고 있어, 월말 시점에 '그 달까지의 뉴스'만 골라 쓰면 미래 정보가 섞이지 않는다(내용 자체는 사후 편집될 수 있다).
요청 한도(429)가 엄격해서 원문(wikitext)을 40페이지씩 묶어 받는다. 이어받기를 지원한다.
저장 형식: "[분류]" 줄 다음에 "- 주제: 내용" 줄들(출처 표기·링크 제거).
사용: python -m tools.fetch_wiki_news [--start 2015-12-01] [--end 2026-11-12] [--batch 40]    필요: pip install requests
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PREDICTION_DATA_DIR", ROOT / "data"))
DST = DATA / "news_wiki.sqlite"
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ant-helper-research/1.0 (historical news backtest)"}
PAUSE = 1.5        # 요청 사이 쉬는 시간(초)


def page_title(d: date) -> str:
    return f"Portal:Current events/{d.year} {d.strftime('%B')} {d.day}"


def clean(s: str) -> str:
    s = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>|<!--.*?-->", "", s, flags=re.S)
    for _ in range(3):
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"\[https?://[^\s\]]+[^\]]*\]", "", s)                       # [주소 (출처)] 는 지운다
    s = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", s)
    s = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"'{2,}", "", s)
    s = re.sub(r"<[^>]+>", "", s).replace("&nbsp;", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?)])", r"\1", s)
    return re.sub(r"(\(\s*\)\s*)+$", "", s).strip()


def parse_wikitext(content: str) -> str:
    lines = content.splitlines()
    out: List[str] = []
    topic = ""
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        bold = re.fullmatch(r"\x27\x27\x27(.+?)\x27\x27\x27\s*", line)            # 2020-12 부터는 분류 제목이 ;분류 가 아니라 굵은 글씨다
        if line.startswith(";") or bold:
            out.append(f"[{clean(line[1:] if line.startswith(chr(59)) else bold.group(1))}]")
            topic = ""
        elif line.startswith("*") and not line.startswith("**"):
            text = clean(line[1:])
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if nxt.startswith("**"):
                topic = text                       # 하위 항목이 있으면 주제 제목
            else:
                topic = ""
                if len(text) > 3:
                    out.append(f"- {text}")
        elif line.startswith("**"):
            text = clean(line.lstrip("*"))
            if len(text) > 3:
                out.append(f"- {topic}: {text}" if topic else f"- {text}")
    return "\n".join(out)


def fetch_batch(days: List[date]) -> Dict[str, str]:
    import requests

    titles = "|".join(page_title(d) for d in days)
    for attempt in range(8):
        r = requests.get(API, params={"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main", "titles": titles,
                                      "format": "json", "formatversion": 2}, headers=HEADERS, timeout=120)
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 30)) + 2 + attempt * 3)
            continue
        r.raise_for_status()
        pages = {p["title"]: p for p in r.json().get("query", {}).get("pages", [])}
        res = {}
        for d in days:
            p = pages.get(page_title(d))
            text = ""
            if p and not p.get("missing") and p.get("revisions"):
                text = parse_wikitext(p["revisions"][0]["slots"]["main"]["content"])
            res[d.isoformat()] = text
        return res
    raise RuntimeError("요청 한도에서 벗어나지 못했다")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-12-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--redo", action="store_true", help="이미 받은 날도 다시 받는다(형식을 통일할 때)")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DST)
    con.execute("create table if not exists news (date text primary key, text text)")
    done = set() if a.redo else {r[0] for r in con.execute("select date from news")}
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    todo = [d for d in days if d.isoformat() not in done]
    print(f"{len(days):,}일 중 받을 것 {len(todo):,}", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        for k, v in fetch_batch(chunk).items():
            con.execute("insert or replace into news values (?,?)", (k, v))
        con.commit()
        done_n = min(i + a.batch, len(todo))
        print(f"  {done_n:,}/{len(todo):,} {(time.time() - t0) / 60:.1f}분 경과", flush=True)
        time.sleep(PAUSE)
    tot = con.execute("select count(*), sum(length(text)), sum(length(text) = 0) from news").fetchone()
    print(f"완료: {tot[0]:,}일, 글자 수 {tot[1]:,}, 빈 날 {tot[2]:,} → {DST}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
