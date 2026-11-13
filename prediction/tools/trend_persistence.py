"""'오르던 종목은 계속 오른다'(추세 지속) 가정이 맞는지 기간별로 검증한다.

날짜마다 종목을 추세 지표로 줄 세우고, 이후 1·5·20·60거래일 수익률과의 순위상관(IC), 상위 20%-하위 20% 수익 차이,
'우상향 상태' 종목이 이후에 오른 비율을 본다. 앞/뒤 절반으로 나눠 일관성도 본다.
주의: 대상이 '지금 거래대금이 큰 종목'이라 과거에 많이 오른 종목이 더 많이 들어와 있다(생존자 편향). 추세 지표에 유리하게 나온다.
사용: python -m tools.trend_persistence KRX|US
"""
from __future__ import annotations

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
HORIZONS = (1, 5, 20, 60)


def load(market: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("select symbol,date,close,volume from candle where market=? order by symbol,date", con, params=(market,))
    out = []
    for sym, g in df.groupby("symbol"):
        g = g.drop_duplicates("date").reset_index(drop=True)
        if len(g) < 320:
            continue
        c = g["close"]
        f = pd.DataFrame({"symbol": sym, "date": pd.to_datetime(g["date"])})
        f["tv20"] = (c * g["volume"]).rolling(20, min_periods=15).mean()
        f["ret20"] = c.pct_change(20)
        f["ret60"] = c.pct_change(60)
        f["ret120"] = c.pct_change(120)
        f["mom12_1"] = c.shift(20) / c.shift(250) - 1                       # 최근 1개월 빼고 약 1년 수익
        f["above200"] = (c / c.rolling(200, min_periods=180).mean() - 1)
        ma20, ma60, ma120 = (c.rolling(w).mean() for w in (20, 60, 120))
        f["aligned"] = ((c > ma20) & (ma20 > ma60) & (ma60 > ma120)).astype(float)   # 완전 정배열 = '우상향 상태'
        f["trend_quality"] = f["ret120"] / (c.pct_change().rolling(120).std() * np.sqrt(120))   # 변동성 대비 상승(매끈한 우상향)
        for h in HORIZONS:
            f[f"fwd{h}"] = c.shift(-h) / c - 1
        out.append(f)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    market = sys.argv[1] if len(sys.argv) > 1 else "KRX"
    p = load(market)
    p = p.dropna(subset=["tv20", "mom12_1", "above200", "trend_quality", "ret60"])
    dates = sorted(p["date"].unique())[::5]
    factors = ["ret20", "ret60", "ret120", "mom12_1", "above200", "trend_quality"]
    ic = {(f, h): [] for f in factors for h in HORIZONS}
    spread = {(f, h): [] for f in factors for h in HORIZONS}
    aligned = {h: [] for h in HORIZONS}
    meta = []
    for d in dates:
        day = p[p["date"] == d].nlargest(200, "tv20")
        if len(day) < 100:
            continue
        meta.append(d)
        for h in HORIZONS:
            col = f"fwd{h}"
            dd = day.dropna(subset=[col])
            if len(dd) < 80:
                for f in factors:
                    ic[(f, h)].append(np.nan); spread[(f, h)].append(np.nan)
                aligned[h].append(np.nan)
                continue
            for f in factors:
                ic[(f, h)].append(dd[f].corr(dd[col], method="spearman"))
                q = dd[f].rank(pct=True)
                spread[(f, h)].append(dd.loc[q > 0.8, col].mean() - dd.loc[q <= 0.2, col].mean())
            a = dd[dd["aligned"] == 1]
            if len(a) >= 10:
                aligned[h].append((a[col] > 0).mean() - (dd[col] > 0).mean())
            else:
                aligned[h].append(np.nan)
    n = len(meta)
    print(f"{market}: 평가 날짜 {n}개 (5거래일 간격, {pd.Timestamp(meta[0]).date()} ~ {pd.Timestamp(meta[-1]).date()}), 날짜마다 거래대금 상위 200종목")
    print("※ 이후 수익률이 겹치므로 t 값은 실제보다 크게 나옵니다. 부호의 일관성(앞/뒤 절반)을 같이 보세요.\n")
    print(f"{'추세 지표':16s} {'기간':>5s} {'IC':>8s} {'t':>6s} {'앞절반':>8s} {'뒤절반':>8s} {'상위20%-하위20%':>16s}")
    for f in factors:
        for h in HORIZONS:
            v = np.array(ic[(f, h)], dtype=float); v = v[~np.isnan(v)]
            s = np.array(spread[(f, h)], dtype=float); s = s[~np.isnan(s)]
            if len(v) < 6:
                continue
            half = len(v) // 2
            t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
            print(f"{f:16s} {h:4d}일 {v.mean():+8.3f} {t:6.1f} {v[:half].mean():+8.3f} {v[half:].mean():+8.3f} {s.mean()*100:+15.2f}%p")
    print("\n'완전 정배열(20>60>120일선, 종가>20일선)' 종목이 이후 오른 비율 - 후보 전체 평균 오른 비율")
    for h in HORIZONS:
        v = np.array(aligned[h], dtype=float); v = v[~np.isnan(v)]
        if len(v) >= 6:
            half = len(v) // 2
            print(f"  {h:3d}일 뒤: {v.mean()*100:+5.1f}%p (앞절반 {v[:half].mean()*100:+5.1f}, 뒤절반 {v[half:].mean()*100:+5.1f})")


if __name__ == "__main__":
    main()
