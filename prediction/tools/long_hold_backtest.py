"""긴 보유 기간(20/40/60거래일) 전략 백테스트 (연구용 도구, 서비스에는 포함되지 않는다).

H 거래일마다 한 번 종목을 다시 고르고(리밸런싱), 고른 종목을 동일 비중으로 H 거래일 들고 간 수익을
'같은 날 후보 전체(거래대금 상위 200종목) 동일 비중 평균'과 비교한다. 날짜 T 에는 T 까지의 데이터만 쓴다.
거래비용: 바뀐 종목 비중 x 왕복 비용(한국 0.4%, 미국 0.2%; 세금·수수료·슬리피지 포함 가정). 기준(후보 전체)에는 비용을 안 물려서 전략에 불리하게 잡았다.
주의: 대상이 '지금 거래대금이 큰 종목'이라 과거에 많이 오른 종목이 더 많다(생존자 편향). 모든 전략의 절대수익이 부풀려진다.
사용: python -m tools.long_hold_backtest KRX|US [--top 20] [--universe 200]
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
ROUND_TRIP = {"KRX": 0.004, "US": 0.002}
HOLDS = (20, 40, 60)
START = 252          # 1년 모멘텀 계산에 필요한 앞부분


def load(market: str):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("select symbol,date,close,volume from candle where market=? order by date", con, params=(market,))
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot_table(index="date", columns="symbol", values="close")
    vol = df.pivot_table(index="date", columns="symbol", values="volume")
    close = close.dropna(axis=1, thresh=int(len(close) * 0.8))
    return close.ffill(limit=3), vol.reindex_like(close).fillna(0)


def features(c: pd.DataFrame, v: pd.DataFrame) -> dict[str, pd.DataFrame]:
    daily = c.pct_change()
    ma = {w: c.rolling(w, min_periods=int(w * 0.9)).mean() for w in (20, 60, 120, 200)}
    f = {
        "tv20": (c * v).rolling(20, min_periods=15).mean(),
        "ret20": c.pct_change(20),
        "ret60": c.pct_change(60),
        "ret120": c.pct_change(120),
        "mom": c.shift(20) / c.shift(250) - 1,
        "vol60": daily.rolling(60, min_periods=50).std(),
        "above200": c / ma[200] - 1,
        "aligned": ((c > ma[20]) & (ma[20] > ma[60]) & (ma[60] > ma[120])).astype(float),
        "near_high": c / c.rolling(250, min_periods=200).max() - 1,
    }
    f["quality"] = f["ret120"] / (f["vol60"] * np.sqrt(120))
    return f


def low(day: pd.DataFrame, col: str, n: int) -> list:
    return day[col].nsmallest(n).index.tolist()


def high(day: pd.DataFrame, col: str, n: int) -> list:
    return day[col].nlargest(n).index.tolist()


def strategies(n: int):
    def trend_lowvol(day):
        d = day[day["above200"] > 0]
        return low(d, "vol60", n)

    def trend_lowvol_cool(day):
        d = day[(day["above200"] > 0) & (day["ret20"] < 0.15)]
        return low(d, "vol60", n)

    def aligned_lowvol(day):
        d = day[day["aligned"] == 1]
        return low(d, "vol60", n)

    def lowvol_dip(day):
        d = day.loc[day["vol60"].nsmallest(max(n * 2, 40)).index]
        return low(d, "ret20", n)

    def lowvol_mom(day):
        d = day.loc[day["vol60"].nsmallest(max(n * 3, 60)).index]
        return high(d, "mom", n)

    return {
        "모멘텀(1년, 최근1개월 제외)": lambda d: high(d, "mom", n),
        "정배열 중 저변동": aligned_lowvol,
        "저변동": lambda d: low(d, "vol60", n),
        "200일선 위 중 저변동": trend_lowvol,
        "200일선 위·과열 아님 저변동": trend_lowvol_cool,
        "변동성 대비 상승(매끈한 우상향)": lambda d: high(d, "quality", n),
        "52주 고점 근접": lambda d: high(d, "near_high", n),
        "저변동 중 단기 눌림": lowvol_dip,
        "저변동 중 1년 모멘텀 상위": lowvol_mom,
        "(시장 필터) 200일선 위 저변동": trend_lowvol,     # 시장이 약세면 현금
    }


def run(market: str, top: int, universe: int):
    c, v = load(market)
    f = features(c, v)
    dates = c.index
    # 후보 전체 동일 비중 지수(시장 필터·구간 구분용)
    idx = (1 + c.pct_change().mean(axis=1).fillna(0)).cumprod()
    idx_ma = idx.rolling(120, min_periods=100).mean()
    strats = strategies(top)
    cost = ROUND_TRIP[market]
    out: list[dict] = []
    for h in HOLDS:
        for off in range(0, h, max(1, h // 4)):
            prev: dict[str, set] = {}
            for i in range(START + off, len(dates) - h, h):
                T = dates[i]
                fwd = (c.iloc[i + h] / c.iloc[i] - 1)
                day = pd.DataFrame({k: f[k].iloc[i] for k in f}).dropna(subset=["tv20", "vol60", "mom", "above200", "ret20", "quality", "near_high"])
                day = day[day["tv20"] > 0].nlargest(universe, "tv20")
                day = day[day.index.isin(fwd.dropna().index)]
                if len(day) < max(60, top * 3):
                    continue
                bench = fwd[day.index].mean()
                risk_on = bool(idx.iloc[i] > idx_ma.iloc[i]) if not np.isnan(idx_ma.iloc[i]) else True
                for name, pick in strats.items():
                    if name.startswith("(시장 필터)") and not risk_on:
                        picks: list = []
                    else:
                        picks = pick(day)
                    if not picks:
                        ret, turn = 0.0, 0.0
                    else:
                        ret = fwd[picks].mean()
                        old = prev.get((name, off, h), set())
                        turn = len(set(picks) - old) / len(picks) if old else 1.0
                    prev[(name, off, h)] = set(picks)
                    out.append({"hold": h, "off": off, "date": T, "strategy": name, "gross": ret, "net": ret - turn * cost,
                                "bench": bench, "risk_on": risk_on})
    return pd.DataFrame(out)


def report(market: str, r: pd.DataFrame, top: int, universe: int) -> None:
    print(f"\n==================== {market} (날짜마다 거래대금 상위 {universe}종목 중 {top}종목 동일 비중) ====================")
    print(f"기간: {r['date'].min().date()} ~ {r['date'].max().date()}   왕복 비용 가정 {ROUND_TRIP[market]*100:.1f}%")
    for h in HOLDS:
        s = r[r["hold"] == h].copy()
        s["ex"] = s["net"] - s["bench"]
        mid = s["date"].min() + (s["date"].max() - s["date"].min()) / 2
        n_per = s[(s["strategy"] == s["strategy"].iloc[0])].groupby("off").size().mean()
        print(f"\n--- 보유 {h}거래일 (오프셋 {s['off'].nunique()}가지, 오프셋당 약 {n_per:.0f}회 교체) / 후보 전체 평균 {s.groupby('off')['bench'].mean().mean()*100:+.2f}% (1회 기간) ---")
        print(f"{'전략':28s} {'순수익/회':>9s} {'초과(순)':>9s} {'오프셋별 초과 최소~최대':>22s} {'앞절반':>8s} {'뒤절반':>8s} {'상승장':>8s} {'하락장':>8s} {'이긴 비율':>9s}")
        for name, g in s.groupby("strategy", sort=False):
            per_off = g.groupby("off")["ex"].mean() * 100
            first, second = g[g["date"] <= mid]["ex"].mean() * 100, g[g["date"] > mid]["ex"].mean() * 100
            up, down = g[g["bench"] > 0]["ex"].mean() * 100, g[g["bench"] <= 0]["ex"].mean() * 100
            print(f"{name:28s} {g['net'].mean()*100:+8.2f}% {g['ex'].mean()*100:+8.2f}% {per_off.min():+9.2f}~{per_off.max():+6.2f}%p {first:+8.2f} {second:+8.2f} {up:+8.2f} {down:+8.2f} {(g['ex']>0).mean()*100:8.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("market", choices=["KRX", "US"])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--universe", type=int, default=200)
    a = ap.parse_args()
    r = run(a.market, a.top, a.universe)
    report(a.market, r, a.top, a.universe)
    r.to_csv(ROOT / "data" / f"long_hold_{a.market}.csv", index=False)


if __name__ == "__main__":
    main()
