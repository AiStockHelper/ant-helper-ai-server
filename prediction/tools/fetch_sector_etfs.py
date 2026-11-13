"""뉴스→섹터 백테스트용 섹터 ETF 일봉을 data/sector_etfs.sqlite 에 모은다(수정주가).

- 미국: SPDR 섹터 ETF 11종 + SPY (Yahoo, yfinance)
- 한국: KODEX/TIGER 업종 ETF (FinanceDataReader). 상장 시점이 달라 시점마다 쓸 수 있는 ETF 수가 다르다.
사용: python -m tools.fetch_sector_etfs [--start 2015-06-01]   필요: pip install yfinance finance-datareader
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PREDICTION_DATA_DIR", ROOT / "data"))
DST = DATA / "sector_etfs.sqlite"

US = {
    "XLK": "Technology", "XLE": "Energy", "XLF": "Financials", "XLV": "Health Care", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLI": "Industrials", "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication Services", "SPY": "S&P 500 (benchmark)",
}
KR = {
    "091160": "반도체", "091180": "자동차", "091170": "은행", "117700": "건설", "117680": "철강", "102970": "증권", "117460": "에너지화학",
    "140710": "운송", "140700": "보험", "244580": "바이오", "266390": "경기소비재", "266370": "IT", "266420": "헬스케어", "266410": "필수소비재",
    "300950": "게임", "305720": "2차전지", "069500": "KODEX 200 (benchmark)",
}


def main() -> None:
    import pandas as pd
    logging.disable(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-06-01")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DST)
    con.execute("create table if not exists price (market text, symbol text, date text, open real, high real, low real, close real, volume real, primary key (market, symbol, date))")
    con.execute("create table if not exists meta (market text, symbol text, name text, first_date text, rows integer, primary key (market, symbol))")

    def save(market: str, sym: str, name: str, df) -> None:
        df = df.dropna(subset=["Close"])
        df = df[df["Close"] > 0]
        if df.empty:
            print(f"  {market} {sym} {name}: 시세 없음")
            return
        rows = [(market, sym, i.strftime("%Y-%m-%d"), float(r.Open), float(r.High), float(r.Low), float(r.Close), float(r.Volume or 0)) for i, r in df.iterrows()]
        con.executemany("insert or replace into price values (?,?,?,?,?,?,?,?)", rows)
        con.execute("insert or replace into meta values (?,?,?,?,?)", (market, sym, name, df.index.min().strftime("%Y-%m-%d"), len(rows)))
        print(f"  {market} {sym:6s} {name:24s} {df.index.min().date()} ~ {df.index.max().date()} ({len(rows):,}일)")

    import yfinance as yf
    for sym, name in US.items():
        x = yf.download(sym, start=a.start, progress=False, auto_adjust=True, threads=False)
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        save("US", sym, name, x)
    import FinanceDataReader as fdr
    for sym, name in KR.items():
        save("KR", sym, name, fdr.DataReader(sym, a.start))
    con.commit()
    print(f"저장: {DST}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
