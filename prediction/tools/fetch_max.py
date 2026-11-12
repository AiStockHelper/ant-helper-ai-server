"""전략 탐색용 긴 일봉(수정주가: 분할·배당 반영)을 data/candles_max.sqlite 에 받는다. 서비스 저장소는 건드리지 않는다.

시장 구분(market 열):
  IDX  : 주가지수(^GSPC, ^IXIC, ^KS11 ...) - 상장폐지·생존자 편향이 없다. 가격지수라 배당은 빠져 있다.
  ETF  : 미국 상장 ETF(자산군·섹터·국가) - 생존자 편향이 적다.
  KETF : 한국 상장 ETF
  KRX / US : 개별 종목. '지금 거래대금이 큰 종목'만 받으므로 생존자 편향이 있다(결과가 부풀려질 수 있다).
사용: python -m tools.fetch_max [--years 20] [--stocks 400]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fetch_history import top_symbols  # noqa: E402

DST = ROOT / "data" / "candles_max.sqlite"

IDX = ["^GSPC", "^IXIC", "^RUT", "^KS11", "^KQ11", "^N225", "^STOXX50E"]
ETF = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWY", "EWJ", "VGK", "MCHI", "VNQ", "TLT", "IEF", "SHY", "AGG", "LQD", "HYG",
       "GLD", "SLV", "DBC", "USO", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "BIL"]
KETF = ["069500.KS", "229200.KS", "114800.KS", "122630.KS", "132030.KS", "148070.KS", "153130.KS", "360750.KS"]


def download(tickers: list[str], start: str):
    import yfinance as yf
    for attempt in range(3):
        try:
            return yf.download(tickers, start=start, interval="1d", group_by="ticker", auto_adjust=True,
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


def save(con, rows) -> int:
    con.executemany("insert or replace into candle values (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=20.0)
    ap.add_argument("--stocks", type=int, default=400)
    a = ap.parse_args()
    start = (date.today() - timedelta(days=int(a.years * 365))).isoformat()
    con = sqlite3.connect(DST)
    con.execute("create table if not exists candle (market text, symbol text, date text, open real, high real, low real, "
                "close real, volume real, primary key (market, symbol, date))")
    for market, tickers in (("IDX", IDX), ("ETF", ETF), ("KETF", KETF)):
        df = download(tickers, start)
        rows = [r for t in tickers for r in rows_from(df, t, t, market)]
        print(f"{market}: {save(con, rows):,}행", flush=True)
    for market in ("US", "KRX"):
        syms = top_symbols(market, a.stocks)
        total = 0
        for i in range(0, len(syms), 100):
            chunk = syms[i:i + 100]
            if market == "KRX":
                df = download([s + ".KS" for s in chunk], start)
                rows, got = [], set()
                for s in chunk:
                    r = rows_from(df, s + ".KS", s, market)
                    if r:
                        rows += r
                        got.add(s)
                missing = [s for s in chunk if s not in got]
                if missing:
                    df2 = download([s + ".KQ" for s in missing], start)
                    for s in missing:
                        rows += rows_from(df2, s + ".KQ", s, market)
            else:
                df = download(chunk, start)
                rows = [r for s in chunk for r in rows_from(df, s, s, market)]
            total += save(con, rows)
            print(f"  {market} {min(i + 100, len(syms))}/{len(syms)} 종목, 누적 {total:,}행", flush=True)
    print("done", con.execute("select market, count(distinct symbol), min(date), max(date) from candle group by market").fetchall())


if __name__ == "__main__":
    main()
