"""'지수·ETF 추세/변동성 타깃 자동매매' 스킬의 실제 규칙을 그대로 재현한 일별 백테스트 (연구용, 서비스에는 포함되지 않는다).

스킬 규칙(backend/.../IndexEtfSkill.java 와 같아야 한다):
  - 하루 한 번, 마감 15분 전 가격으로 판단하고 같은 가격으로 체결한다(lag 0). 보수적 점검으로 하루 늦게 체결(lag 1)도 함께 본다.
  - 모드 TREND : 종가가 SMA(window) 위면 목표 비중 100%, 아니면 0%.
  - 모드 VOL   : 목표 비중 = min(1, 목표변동성 / 최근 20일 변동성). 매월 첫 거래일에 다시 계산.
  - 모드 BOTH  : 추세가 꺼져 있으면 0%, 켜져 있으면 VOL 비중.
  - 현재 비중과 목표 비중의 차이가 허용폭(기본 시드의 10%p)보다 작으면 거래하지 않는다. 전량 매도/신규 진입은 허용폭과 무관.
  - 남는 돈은 현금(미국 BIL 수익, 한국 연 2%로 가정).
사용: python -m tools.etf_skill_backtest
"""
from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore")
DB = ROOT / "data" / "candles_max.sqlite"
EVAL_START = pd.Timestamp("2007-12-03")
IS_END = pd.Timestamp("2016-12-31")
BAND = 0.10


def load():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("select symbol,date,close from candle where market in ('ETF','KETF','IDX') order by date", con)
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot_table(index="date", columns="symbol", values="close")


def simulate(close: pd.Series, cash_ret: pd.Series, mode: str, window: int, target: float, cost: float, lag: int, band: float = BAND):
    r = close.pct_change().fillna(0.0)
    sma = close.rolling(window, min_periods=int(window * 0.95)).mean()
    vol20 = r.rolling(20).std() * np.sqrt(252)
    idx = close.index
    equity, pos, cash = 1.0, 0.0, 1.0           # 단위: 시드 1.0
    pos = 0.0
    out_eq, trades, expo = [], 0, []
    vol_w, vol_month = 0.0, None
    pending = None                               # lag 1 용: 어제 내린 결정(목표 비중)
    for i, t in enumerate(idx):
        if i > 0:
            pos *= (1 + r.iloc[i])
            cash *= (1 + cash_ret.iloc[i])
        equity = pos + cash
        # 결정
        decide_now = i >= window
        if decide_now:
            on = bool(close.iloc[i] > sma.iloc[i]) if mode in ("TREND", "BOTH") else True
            month = (t.year, t.month)
            if mode in ("VOL", "BOTH"):
                if vol_month != month or pos == 0:
                    v = vol20.iloc[i]
                    vol_w = 0.0 if np.isnan(v) or v <= 0 else min(1.0, target / v)
                    vol_month = month
                w_t = vol_w if on else 0.0
            else:
                w_t = 1.0 if on else 0.0
            if lag == 1:
                decision, pending = pending, w_t
            else:
                decision = w_t
            if decision is not None:
                cur = pos / equity
                full = decision == 0.0 and cur > 0 or decision > 0 and cur == 0
                if full or abs(decision - cur) > band:
                    want = decision * equity
                    delta = want - pos
                    fee = abs(delta) * cost
                    pos += delta - 0.0
                    cash -= delta + fee
                    trades += 1
        out_eq.append(pos + cash)
        expo.append(pos / (pos + cash) if pos + cash > 0 else 0)
    eq = pd.Series(out_eq, index=idx)
    return eq, trades, pd.Series(expo, index=idx)


def stats(eq: pd.Series, cash_ret: pd.Series, trades: int, expo: pd.Series):
    def seg(e):
        r = e.pct_change().dropna()
        ex = r - cash_ret.reindex(r.index).fillna(0)
        yrs = len(r) / 252
        return {"cagr": (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(252),
                "sharpe": ex.mean() / ex.std() * np.sqrt(252), "mdd": (e / e.cummax() - 1).min()}
    e = eq[eq.index >= EVAL_START]
    full, ins, oos = seg(e), seg(e[e.index <= IS_END]), seg(e[e.index > IS_END])
    yrs = len(e) / 252
    return full, ins, oos, trades / yrs, expo[expo.index >= EVAL_START].mean()


def main() -> None:
    px = load()
    us_cash = px["BIL"].pct_change().fillna(0.0)
    kr_cash = pd.Series(0.02 / 252, index=px.index)
    assets = [("SPY", us_cash, 0.0005), ("QQQ", us_cash, 0.0005), ("069500.KS", kr_cash, 0.001), ("229200.KS", kr_cash, 0.001)]
    modes = [("보유", None, 0, 0), ("추세 SMA150", "TREND", 150, 0), ("추세 SMA200", "TREND", 200, 0), ("변동성 타깃 10%", "VOL", 20, 0.10),
             ("변동성 타깃 12%", "VOL", 20, 0.12), ("변동성 타깃 15%", "VOL", 20, 0.15), ("추세150 + 변동성 12%", "BOTH", 150, 0.12),
             ("추세150 + 변동성 15%", "BOTH", 150, 0.15), ("추세200 + 변동성 12%", "BOTH", 200, 0.12)]
    print(f"허용폭 {BAND*100:.0f}%p, 현금 수익: 미국 BIL / 한국 연 2%. lag0 = 마감 15분 전 가격으로 판단·체결, lag1 = 하루 늦게 체결")
    for sym, cash_ret, cost in assets:
        s = px[sym].dropna()
        cr = cash_ret.reindex(s.index).fillna(0.0)
        print(f"\n=== {sym}  ({s.index[0].date()} ~ {s.index[-1].date()}, 편도 비용 {cost*100:.2f}%) ===")
        print(f"{'전략':22s} {'lag':>3s} {'CAGR':>6s} {'변동성':>6s} {'샤프':>5s} {'MDD':>7s} | {'IS샤프':>6s} {'OOS샤프':>6s} {'OOS CAGR':>8s} {'OOS MDD':>8s} | {'연 거래횟수':>8s} {'평균 투자비중':>9s}")
        for name, mode, win, tv in modes:
            for lag in ((0, 1) if mode else (0,)):
                if mode is None:
                    eq = (1 + s.pct_change().fillna(0)).cumprod()
                    trades, expo = 0, pd.Series(1.0, index=s.index)
                else:
                    eq, trades, expo = simulate(s, cr, mode, win if mode != "VOL" else 150, tv, cost, lag)
                full, ins, oos, tr, ex = stats(eq, cr, trades, expo)
                print(f"{name:22s} {lag:3d} {full['cagr']*100:+5.1f}% {full['vol']*100:5.1f}% {full['sharpe']:5.2f} {full['mdd']*100:6.1f}% | {ins['sharpe']:6.2f} {oos['sharpe']:6.2f} {oos['cagr']*100:+7.1f}% {oos['mdd']*100:7.1f}% | {tr:8.1f} {ex*100:8.0f}%")


if __name__ == "__main__":
    main()
