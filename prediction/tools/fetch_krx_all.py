"""생존자 편향 없는 한국 주식 일봉 수집: 현재 상장 종목 + 상장폐지된 보통주를 모두 data/candles_krx_all.sqlite 에 모은다.

- 현재 상장: FinanceDataReader StockListing('KRX') 중 코스피·코스닥(6자리 코드).
- 상장폐지: StockListing('KRX-DELISTING') 중 --delisted-since 이후 폐지된 코스피·코스닥 '주권'(6자리 코드).
- 시세: FinanceDataReader.DataReader(코드) (네이버 금융, 수정주가). 폐지 종목도 폐지 전날까지 내려온다.
이어받기를 지원한다(이미 받은 종목은 건너뜀). 다른 DB 는 건드리지 않는다.

사용: python -m tools.fetch_krx_all [--start 2014-01-01] [--delisted-since 2015-06-01] [--workers 4]
필요: pip install finance-datareader
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PREDICTION_DATA_DIR", ROOT / "data"))
DST = DATA / "candles_krx_all.sqlite"


def universe(delisted_since: str):
    import FinanceDataReader as fdr
    import pandas as pd

    listed = fdr.StockListing("KRX")
    listed = listed[listed["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"]) & listed["Code"].astype(str).str.fullmatch(r"\d{6}")]
    rows = [(r.Code, r.Name, "KOSPI" if r.Market == "KOSPI" else "KOSDAQ", "LISTED", None, None) for r in listed.itertuples()]

    dl = fdr.StockListing("KRX-DELISTING")
    dl["DelistingDate"] = pd.to_datetime(dl["DelistingDate"], errors="coerce")
    dl = dl[(dl["DelistingDate"] >= delisted_since) & dl["Market"].isin(["KOSPI", "KOSDAQ"]) & dl["Symbol"].astype(str).str.fullmatch(r"\d{6}")
            & dl["SecuGroup"].astype(str).str.contains("주권")]
    seen = {r[0] for r in rows}
    for r in dl.itertuples():
        if r.Symbol not in seen:
            rows.append((r.Symbol, r.Name, r.Market, "DELISTED", str(r.DelistingDate.date()), str(r.Reason)))
            seen.add(r.Symbol)
    return rows


def fetch_one(code: str, start: str, end: str):
    import FinanceDataReader as fdr
    last = None
    for attempt in range(4):
        try:
            df = fdr.DataReader(code, start, end)
            return df
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--delisted-since", default="2015-06-01")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 앞의 N개만")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DST)
    con.execute("create table if not exists candle (market text, symbol text, date text, open real, high real, low real, close real, volume real, "
                "primary key (market, symbol, date))")
    con.execute("create table if not exists meta (symbol text primary key, name text, market text, status text, delisting_date text, reason text, "
                "rows integer, first_date text, last_date text)")
    done = {r[0] for r in con.execute("select symbol from meta where rows is not null")}
    uni = universe(a.delisted_since)
    if a.limit:
        uni = uni[:a.limit]
    todo = [u for u in uni if u[0] not in done]
    print(f"유니버스 {len(uni):,}개 (상장 {sum(1 for u in uni if u[3] == 'LISTED'):,} + 폐지 {sum(1 for u in uni if u[3] == 'DELISTED'):,}), 이미 받음 {len(done):,}, 받을 것 {len(todo):,}", flush=True)
    end = date.today().isoformat()
    t0 = time.time()
    ok = fail = empty = 0

    def work(u):
        try:
            return u, fetch_one(u[0], a.start, end), None
        except Exception as exc:  # noqa: BLE001
            return u, None, exc

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(work, u) for u in todo]
        for i, f in enumerate(as_completed(futures), 1):
            (code, name, market, status, dldate, reason), df, err = f.result()
            if err is not None:
                fail += 1
                con.execute("insert or replace into meta values (?,?,?,?,?,?,?,?,?)", (code, name, market, status, dldate, reason, None, None, None))
            elif df is None or df.empty:
                empty += 1
                con.execute("insert or replace into meta values (?,?,?,?,?,?,?,?,?)", (code, name, market, status, dldate, reason, 0, None, None))
            else:
                df = df.dropna(subset=["Close"])
                df = df[df["Close"] > 0]
                rows = [("KRX", code, idx.strftime("%Y-%m-%d"), float(o), float(h), float(l), float(c), float(v or 0))
                        for idx, o, h, l, c, v in df[["Open", "High", "Low", "Close", "Volume"]].itertuples()]
                con.executemany("insert or replace into candle values (?,?,?,?,?,?,?,?)", rows)
                con.execute("insert or replace into meta values (?,?,?,?,?,?,?,?,?)",
                            (code, name, market, status, dldate, reason, len(rows), df.index.min().strftime("%Y-%m-%d"), df.index.max().strftime("%Y-%m-%d")))
                ok += 1
            if i % 100 == 0:
                con.commit()
                rate = i / (time.time() - t0)
                print(f"  {i:,}/{len(todo):,} (성공 {ok:,} · 빈 응답 {empty:,} · 실패 {fail:,}) {rate:.1f}개/초, 남은 시간 약 {(len(todo) - i) / max(rate, 0.01) / 60:.0f}분", flush=True)
    con.commit()
    n = con.execute("select count(*), count(distinct symbol) from candle").fetchone()
    print(f"완료: 성공 {ok:,} · 빈 응답 {empty:,} · 실패 {fail:,} | candle {n[0]:,}행 / {n[1]:,}종목 → {DST}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
