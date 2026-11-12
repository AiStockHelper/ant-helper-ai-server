"""백테스트용으로 긴 일봉(기본 3년)을 별도 파일(data/candles_long.sqlite)에 받는다. 서비스 저장소(candles.sqlite)는 건드리지 않는다.
대상은 서비스 저장소에서 최근 거래대금이 큰 종목 N개(시장별)다. 한국은 yfinance 의 .KS/.KQ 를 쓴다.
사용: python -m tools.fetch_history KRX|US [--top 500] [--years 3]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "data" / "candles.sqlite"
DST = ROOT / "data" / "candles_long.sqlite"
CHUNK = 100


def top_symbols(market: str, n: int) -> list[str]:
    con = sqlite3.connect(SRC)
    df = pd.read_sql_query(
        "select symbol, close, volume, date from candle where market=? and date >= (select date(max(date), '-40 day') from candle where market=?)",
        con, params=(market, market))
    df["tv"] = df["close"] * df["volume"]
    df = df.sort_values("date").groupby("symbol").tail(20)
    tv = df.groupby("symbol")["tv"].mean().sort_values(ascending=False)
    if market == "KRX":
        tv = tv[tv.index.str.match(r"^[0-9A-Z]{6}$") & tv.index.str.endswith("0")]
    else:
        tv = tv[tv.index.str.match(r"^[A-Z]{1,5}$")]
    return list(tv.index[:n])


def download(tickers: list[str], start: str):
    import yfinance as yf
    for attempt in range(3):
        try:
            return yf.download(tickers, start=start, interval="1d", group_by="ticker", auto_adjust=False,
                               threads=True, progress=False)
        except Exception as exc:  # noqa: BLE001
            print("  retry", attempt + 1, exc)
            time.sleep(3 * (attempt + 1))
    return None


def rows_from(df, ticker: str, sym: str, market: str):
    if df is None or df.empty or ticker not in set(df.columns.get_level_values(0)):
        return []
    sub = df[ticker].dropna(subset=["Close"])[["Open", "High", "Low", "Close", "Volume"]]
    return [(market, sym, idx.strftime("%Y-%m-%d"), float(o), float(h), float(l), float(c), float(v or 0))
            for idx, o, h, l, c, v in sub.itertuples() if c and c > 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("market", choices=["KRX", "US"])
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--years", type=float, default=3.0)
    a = ap.parse_args()
    syms = top_symbols(a.market, a.top)
    start = (date.today() - timedelta(days=int(a.years * 365))).isoformat()
    print(f"{a.market}: {len(syms)}종목, {start} 부터")
    con = sqlite3.connect(DST)
    con.execute("create table if not exists candle (market text, symbol text, date text, open real, high real, low real, "
                "close real, volume real, primary key (market, symbol, date))")
    total = 0
    for i in range(0, len(syms), CHUNK):
        chunk = syms[i:i + CHUNK]
        got = set()
        if a.market == "KRX":
            df = download([s + ".KS" for s in chunk], start)
            rows = []
            for s in chunk:
                r = rows_from(df, s + ".KS", s, a.market)
                if r:
                    rows += r
                    got.add(s)
            missing = [s for s in chunk if s not in got]
            if missing:
                df2 = download([s + ".KQ" for s in missing], start)
                for s in missing:
                    rows += rows_from(df2, s + ".KQ", s, a.market)
        else:
            df = download(chunk, start)
            rows = [r for s in chunk for r in rows_from(df, s, s, a.market)]
        con.executemany("insert or replace into candle values (?,?,?,?,?,?,?,?)", rows)
        con.commit()
        total += len(rows)
        print(f"  {min(i + CHUNK, len(syms))}/{len(syms)} 종목, 누적 {total:,}행")
    print("done", con.execute("select market, count(distinct symbol), min(date), max(date) from candle group by market").fetchall())


if __name__ == "__main__":
    main()
