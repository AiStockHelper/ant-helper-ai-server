"""미국 S&P 500 '그날의 구성종목' 백테스트용 데이터 수집: 한 번이라도 지수에 들어갔던 종목의 일봉을 data/candles_us_sp500.sqlite 에 모은다.

- 구성 이력: github.com/fja05680/sp500 의 sp500_ticker_start_end.csv (종목별 편입일·편출일). 이 이력으로 '그날의 구성종목'을 복원한다.
- 시세: Yahoo Finance(yfinance), 분할·배당 반영 수정주가. 파산·피인수로 사라진 종목은 Yahoo 에 없는 경우가 많아 일부 빠진다(meta.rows=0).
- 지수: ^GSPC 를 market='IDX' 로 함께 저장한다.
다른 DB 는 건드리지 않는다. 사용: python -m tools.fetch_us_sp500 [--start 2012-01-01] [--batch 40]
필요: pip install yfinance pandas
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PREDICTION_DATA_DIR", ROOT / "data"))
DST = DATA / "candles_us_sp500.sqlite"
MEMBERS_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"


def main() -> None:
    import pandas as pd
    import yfinance as yf

    logging.disable(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    csv = DATA / "sp500_ticker_start_end.csv"
    urllib.request.urlretrieve(MEMBERS_URL, csv)
    mem = pd.read_csv(csv)
    mem.columns = ["ticker", "start_date", "end_date"]

    con = sqlite3.connect(DST)
    con.execute("create table if not exists candle (market text, symbol text, date text, open real, high real, low real, close real, volume real, "
                "primary key (market, symbol, date))")
    con.execute("create table if not exists meta (symbol text primary key, rows integer, first_date text, last_date text)")
    con.execute("create table if not exists member (ticker text, start_date text, end_date text)")
    con.execute("delete from member")
    con.executemany("insert into member values (?,?,?)", [(r.ticker, str(r.start_date), None if pd.isna(r.end_date) else str(r.end_date)) for r in mem.itertuples()])
    done = {r[0] for r in con.execute("select symbol from meta")}
    tickers = sorted(set(mem["ticker"].astype(str)))
    if a.limit:
        tickers = tickers[:a.limit]
    todo = [t for t in ["^GSPC"] + tickers if t not in done]
    print(f"종목 {len(tickers):,}개 + 지수, 이미 받음 {len(done):,}, 받을 것 {len(todo):,}", flush=True)
    ok = empty = 0
    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        batch = todo[i:i + a.batch]
        ymap = {t: t.replace(".", "-") for t in batch}
        try:
            df = yf.download(list(ymap.values()), start=a.start, progress=False, auto_adjust=True, group_by="ticker", threads=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  배치 실패({exc.__class__.__name__}), 개별 재시도", flush=True)
            df = None
        for t in batch:
            y = ymap[t]
            try:
                if df is not None and len(batch) > 1:
                    x = df[y]
                elif df is not None:
                    x = df if "Close" in df.columns else df[y]
                else:
                    x = yf.download(y, start=a.start, progress=False, auto_adjust=True, threads=False)
                    if isinstance(x.columns, pd.MultiIndex):
                        x.columns = x.columns.get_level_values(0)
                x = x.dropna(subset=["Close"])
                x = x[x["Close"] > 0]
            except Exception:  # noqa: BLE001
                x = None
            if x is None or len(x) == 0:
                empty += 1
                con.execute("insert or replace into meta values (?,?,?,?)", (t, 0, None, None))
                continue
            market = "IDX" if t.startswith("^") else "US"
            rows = [(market, t, idx.strftime("%Y-%m-%d"), float(r.Open), float(r.High), float(r.Low), float(r.Close), float(r.Volume or 0))
                    for idx, r in x.iterrows()]
            con.executemany("insert or replace into candle values (?,?,?,?,?,?,?,?)", rows)
            con.execute("insert or replace into meta values (?,?,?,?)", (t, len(rows), x.index.min().strftime("%Y-%m-%d"), x.index.max().strftime("%Y-%m-%d")))
            ok += 1
        con.commit()
        print(f"  {min(i + a.batch, len(todo)):,}/{len(todo):,} (성공 {ok:,} · 시세 없음 {empty:,}) {(time.time() - t0) / 60:.1f}분 경과", flush=True)
    n = con.execute("select count(*), count(distinct symbol) from candle").fetchone()
    print(f"완료: 성공 {ok:,} · 시세 없음 {empty:,} | candle {n[0]:,}행 / {n[1]:,}종목 → {DST}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
