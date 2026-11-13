"""'오르는 종목만 들고, 내리면 팔아서 현금' 추세 추종 전략의 일별 백테스트 (연구용 도구, 서비스에는 포함되지 않는다).

매일 종가 기준으로 종목별 신호(보유/비보유)를 계산하고, 신호는 하루 늦게(다음 날 종가에) 체결한다고 가정해 그 다음 날 수익을 반영한다.
 - 슬롯형(S): 후보 N종목(거래대금 상위 200)마다 1/N 비중. 신호가 없는 만큼은 현금 -> 시장이 나쁠 때 자연스럽게 현금 비중이 늘어난다.
 - 집중형(F): 신호가 켜진 종목만 동일 비중으로 항상 100% 투자(신호가 하나도 없으면 현금).
기준: 후보 전체 동일 비중을 매일 맞춰 들고 있는 것(비용 없음 = 전략에 불리하게 잡음).
거래비용: 비중 변화량 x 편도 비용(한국 0.2%, 미국 0.1%; 세금·수수료·슬리피지 포함 가정).
지표: 연환산 수익·변동성·샤프, 최대 낙폭, 평균 투자 비중, 시장 베타를 뺀 알파(연환산, t값), 앞/뒤 절반, 기준이 가장 크게 빠졌던 구간의 성과.
주의: 대상이 '지금 거래대금이 큰 종목'이라 과거에 많이 오른 종목이 더 많다(생존자 편향). 기간이 약 2년이라 약세장 표본이 적다.
사용: python -m tools.trend_follow_backtest KRX|US [--universe 200]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
DB = ROOT / "data" / "candles_long.sqlite"
SIDE_COST = {"KRX": 0.002, "US": 0.001}
START = 205   # 200일선 계산에 필요한 앞부분


def load(market: str):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("select symbol,date,close,volume from candle where market=? order by date", con, params=(market,))
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot_table(index="date", columns="symbol", values="close")
    vol = df.pivot_table(index="date", columns="symbol", values="volume")
    close = close.dropna(axis=1, thresh=int(len(close) * 0.8))
    return close.ffill(limit=3), vol.reindex_like(close).fillna(0)


def latch(enter: pd.DataFrame, exit_: pd.DataFrame) -> pd.DataFrame:
    """진입 신호에 켜지고 청산 신호에 꺼지는 상태(둘 다 아니면 이전 상태 유지)."""
    s = pd.DataFrame(np.nan, index=enter.index, columns=enter.columns)
    s[enter] = 1.0
    s[exit_ & ~enter] = 0.0
    return s.ffill().fillna(0.0)


def signals(c: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ma50, ma200 = c.rolling(50, min_periods=45).mean(), c.rolling(200, min_periods=180).mean()
    hi60_prev = c.shift(1).rolling(60, min_periods=50).max()
    lo20_prev = c.shift(1).rolling(20, min_periods=18).min()
    hi60 = c.rolling(60, min_periods=50).max()
    ret120 = c.pct_change(120)
    return {
        "50일선 위": (c > ma50).astype(float),
        "200일선 위": (c > ma200).astype(float),
        "50일선 위 & 50>200 (상승 확인)": ((c > ma50) & (ma50 > ma200)).astype(float),
        "120일 상승 & 50일선 위": ((ret120 > 0) & (c > ma50)).astype(float),
        "60일 신고가 진입 / 20일 저점 이탈 청산": latch(c >= hi60_prev, c < lo20_prev),
        "60일 신고가 진입 / 고점 대비 -10% 청산": latch(c >= hi60_prev, c < hi60 * 0.90),
    }


def stats(r: pd.Series, bench: pd.Series) -> dict:
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    x, y = bench.values, r.values
    beta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    res = y - beta * x
    alpha = res.mean() * 252
    t = res.mean() / (res.std(ddof=1) / np.sqrt(len(res)))
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol > 0 else np.nan, "mdd": dd, "beta": beta, "alpha": alpha, "t": t}


def run(market: str, universe: int):
    c, v = load(market)
    ret = c.pct_change().fillna(0.0)
    tv20 = (c * v).rolling(20, min_periods=15).mean()
    in_uni = (tv20.rank(axis=1, ascending=False) <= universe).astype(float)
    uni_prev = in_uni.shift(1).fillna(0.0)
    bench_w = uni_prev.div(uni_prev.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    bench = (bench_w * ret).sum(axis=1)
    sig = signals(c)
    cost = SIDE_COST[market]
    curves: dict[str, pd.Series] = {"(기준) 후보 전체 동일 비중": bench}
    expo: dict[str, float] = {"(기준) 후보 전체 동일 비중": 1.0}
    turn: dict[str, float] = {"(기준) 후보 전체 동일 비중": 0.0}
    for name, s in sig.items():
        # 신호는 t 종가에 알고 t+1 종가에 체결 -> t+2 수익부터 반영(보수적)
        held = s.shift(2).fillna(0.0) * uni_prev
        for kind in ("S", "F"):
            if kind == "S":
                w = held / universe
            else:
                n = held.sum(axis=1).replace(0, np.nan)
                w = held.div(n, axis=0).fillna(0.0)
            gross = (w * ret).sum(axis=1)
            tn = w.diff().abs().sum(axis=1).fillna(0.0)
            net = gross - tn * cost
            key = f"[{kind}] {name}"
            curves[key], expo[key], turn[key] = net, float(w.sum(axis=1).iloc[START:].mean()), float(tn.iloc[START:].sum() / (len(tn) - START) * 252)
    return curves, expo, turn, bench


def worst_window(bench: pd.Series):
    eq = (1 + bench).cumprod()
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    return peak, trough


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("market", choices=["KRX", "US"])
    ap.add_argument("--universe", type=int, default=200)
    a = ap.parse_args()
    curves, expo, turn, bench_full = run(a.market, a.universe)
    idx = bench_full.index[START:]
    bench = bench_full.loc[idx]
    peak, trough = worst_window(bench)
    mid = idx[len(idx) // 2]
    print(f"==================== {a.market} 일별 시뮬레이션: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일), 후보 거래대금 상위 {a.universe}종목 ====================")
    print(f"편도 비용 가정 {SIDE_COST[a.market]*100:.1f}%. S=슬롯형(신호 없는 만큼 현금), F=집중형(신호 종목만 100% 투자)")
    print(f"기준이 가장 크게 빠진 구간: {peak.date()} -> {trough.date()}  (기준 {((1+bench.loc[peak:trough]).prod()-1)*100:+.1f}%)\n")
    print(f"{'전략':46s} {'연수익':>7s} {'변동성':>6s} {'샤프':>5s} {'최대낙폭':>8s} {'투자비중':>7s} {'연회전':>6s} {'베타':>5s} {'알파(연)':>9s} {'t':>5s} {'앞절반':>7s} {'뒤절반':>7s} {'하락구간':>8s}")
    for name, r_full in curves.items():
        r = r_full.loc[idx]
        st = stats(r, bench)
        h1, h2 = (1 + r[r.index <= mid]).prod() - 1, (1 + r[r.index > mid]).prod() - 1
        dn = (1 + r.loc[peak:trough]).prod() - 1
        print(f"{name:46s} {st['ann']*100:+6.1f}% {st['vol']*100:5.1f}% {st['sharpe']:5.2f} {st['mdd']*100:7.1f}% {expo[name]*100:6.0f}% {turn[name]:5.1f}x {st['beta']:5.2f} {st['alpha']*100:+8.1f}% {st['t']:5.1f} {h1*100:+6.1f}% {h2*100:+6.1f}% {dn*100:+7.1f}%")
    pd.DataFrame(curves).to_csv(ROOT / "data" / f"trend_follow_{a.market}.csv")


if __name__ == "__main__":
    main()
