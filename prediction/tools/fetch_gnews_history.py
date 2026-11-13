"""구글뉴스 RSS 의 날짜 지정 검색으로 과거 경제 뉴스 헤드라인을 모은다(뉴스 AI 스킬 백테스트용).

주의
- 구글뉴스는 기사 시각을 '날짜 단위'(UTC 08:00 같은 자리 표시)로만 준다. 그래서 하루를 통째로 하나의 날짜로 취급하고,
  판단일 D 에는 D 보다 이전 날짜의 기사만 쓴다(같은 날짜 기사에는 장 마감 뒤 소식이 섞여 있을 수 있다).
- after/before 경계 해석이 모호해서 [N-1, N+1] 로 넓게 검색한 뒤 날짜가 정확히 N 인 기사만 남긴다.
- 질의당 몇 건 안 나오므로 주제별 질의를 여러 개 쓴다. 이어받기를 지원한다.

사용: python -m tools.fetch_gnews_history [--start 2026-01-01] [--end 2026-11-12] [--workers 3]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PREDICTION_DATA_DIR", ROOT / "data"))
DST = DATA / "news_study" / "gnews_hist.sqlite"

QUERIES = {
    "kr": ["코스피 마감", "코스닥", "삼성전자 SK하이닉스", "환율 금리", "반도체 수출", "실적 발표", "2차전지 바이오 자동차", "외국인 순매수"],
    "us": ["Wall Street stocks close", "Nasdaq S&P 500", "Nvidia Tesla Apple", "Fed interest rates", "oil prices", "earnings report shares", "tariffs trade", "Treasury yields dollar"],
}
LOCALE = {"kr": ("ko", "KR", "KR:ko"), "us": ("en-US", "US", "US:en")}
_db_lock = threading.Lock()


def fetch(lang: str, q: str, n: date):
    hl, gl, ceid = LOCALE[lang]
    rng = f"after:{(n - timedelta(days=1)).isoformat()} before:{(n + timedelta(days=1)).isoformat()}"
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": f"{q} {rng}", "hl": hl, "gl": gl, "ceid": ceid})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AntHelper research)"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                root = ET.fromstring(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(20 * (attempt + 1))
                continue
            raise
        except Exception:  # noqa: BLE001
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError("재시도 초과")
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        d = it.findtext("pubDate")
        if not title or not d:
            continue
        stamp = parsedate_to_datetime(d)
        if stamp.date() != n:
            continue                                    # 날짜가 정확히 N 인 기사만
        src = ""
        if " - " in title:
            title, src = title.rsplit(" - ", 1)
        out.append((stamp.isoformat(), title.strip(), src.strip()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-11-12")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    DST.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DST, check_same_thread=False)
    con.execute("create table if not exists hl (lang text, day text, q text, stamp text, title text, source text)")
    con.execute("create table if not exists done (lang text, day text, q text, n integer, primary key (lang, day, q))")
    done = {(r[0], r[1], r[2]) for r in con.execute("select lang, day, q from done")}
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    todo = [(lang, d, q) for d in days for lang in ("kr", "us") for q in QUERIES[lang] if (lang, d.isoformat(), q) not in done]
    print(f"받을 요청 {len(todo):,}건 (이미 받음 {len(done):,})", flush=True)
    t0 = time.time()

    def work(job):
        lang, d, q = job
        time.sleep(0.5)
        return job, fetch(lang, q, d)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, j) for j in todo]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                (lang, d, q), items = f.result()
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"  실패 {exc}", flush=True)
                continue
            with _db_lock:
                con.executemany("insert into hl values (?,?,?,?,?,?)", [(lang, d.isoformat(), q, s, t, src) for s, t, src in items])
                con.execute("insert or replace into done values (?,?,?,?)", (lang, d.isoformat(), q, len(items)))
                if i % 100 == 0:
                    con.commit()
            ok += 1
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i:,}/{len(todo):,} (실패 {fail}) {rate:.1f}건/초, 남은 시간 약 {(len(todo) - i) / max(rate, 0.01) / 60:.0f}분", flush=True)
    con.commit()
    n = con.execute("select lang, count(distinct day), count(*) from hl group by lang").fetchall()
    print(f"완료: 성공 {ok:,} 실패 {fail} | {n} → {DST}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
