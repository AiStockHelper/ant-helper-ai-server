"""추천 순위 방식을 과거 데이터로 비교하는 간단한 백테스트(연구용 도구, 서비스에는 포함되지 않는다).

candles.sqlite 에 쌓인 일봉으로, 날짜 T 마다 T 까지의 데이터만 써서 종목을 점수화하고
다음 거래일(1일)·5거래일 뒤 수익률이 점수 상위 종목에서 전체 평균보다 나은지 본다.
사용: python -m tools.backtest_factors KRX [--top 10] [--universe 300]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.indicators import add_indicators  # noqa: E402

DB = Path(__file__).resolve().parents[1] / "data" / "candles.sqlite"
MIN_TV = {"KRX": 1e9, "US": 2e7}
MIN_PX = {"KRX": 1000.0, "US": 5.0}


def load(market: str) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "select symbol,date,open,high,low,close,volume from candle where market=? order by symbol,date", con, params=(market,))
    out = {}
    for sym, g in df.groupby("symbol"):
        if len(g) >= 100:
            out[sym] = g.reset_index(drop=True)
    return out


def features(g: pd.DataFrame) -> pd.DataFrame:
    d = add_indicators(g.assign(date=pd.to_datetime(g["date"])))
    c = d["close"]
    f = pd.DataFrame({"date": g["date"], "close": c})
    f["tv20"] = (d["close"] * d["volume"]).rolling(20, min_periods=15).mean()
    f["ret1"] = d["change_pct"]
    f["ret5"] = d["ret_5d"]
    f["ret20"] = d["ret_20d"]
    f["mom60"] = (c.shift(5) / c.shift(65) - 1) * 100
    f["vol20"] = d["volatility"]
    f["rsi"] = d["rsi14"]
    f["px_ma20"] = (c / d["ma20"] - 1) * 100
    f["px_ma60"] = (c / d["ma60"] - 1) * 100
    f["align"] = ((d["ma5"] > d["ma20"]) & (d["ma20"] > d["ma60"]) & (c > d["ma5"])).astype(float)
    f["volratio"] = d["volume_ratio"]
    f["dist_high60"] = (c / c.rolling(60, min_periods=40).max() - 1) * 100
    f["fwd1"] = (c.shift(-1) / c - 1) * 100
    f["fwd5"] = (c.shift(-5) / c - 1) * 100
    f["idx"] = np.arange(len(g))
    return f


def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def _z(s: pd.Series) -> pd.Series:
    return (s.rank(pct=True) - 0.5) * 2


def composite(day: pd.DataFrame, variant: str = "v2") -> pd.Series:
    """후보 점수(높을수록 좋음). 과거 검증에서 다음 1~5거래일에 일관되게 나은 쪽은 '변동성이 낮고, 최근 덜 오른(또는 눌린) 종목'이었다.
    추세·모멘텀을 높게 치던 이전 방식은 이 구간에서 오히려 반대로 움직였다."""
    if variant == "v1":     # 이전 시도: 모멘텀/추세 중심 (비교용)
        mom = pct_rank(day["mom60"]); quality = pct_rank(day["ret20"] / day["vol20"].clip(lower=0.5))
        trend = (pct_rank(day["px_ma60"]) + day["align"]) / 2
        return 0.30 * mom + 0.25 * quality + 0.20 * trend + 0.10 * (1 - pct_rank(day["vol20"])) + 0.15 * pct_rank(day["dist_high60"])
    s = 0.45 * _z(-day["vol20"]) + 0.20 * _z(-day["rsi"]) + 0.15 * _z(-day["ret5"]) + 0.10 * _z(-day["px_ma20"]) + 0.10 * _z(-day["mom60"])
    if variant == "v3":     # 떨어지는 칼날 제외
        s = s - ((day["ret20"] < -25) | (day["ret5"] < -15)).astype(float) * 1.0
    if variant in ("v2", "v3"):
        s = s - ((day["ret1"] > 12) | (day["ret5"] > 20)).astype(float) * 0.5   # 이미 급등한 종목 추격 금지
    return s

def old_rule_score(sym: str, g: pd.DataFrame, t: int, cur: str) -> float | None:
    """현재 서비스의 점수(규칙 기반 부분만. 종목별 ML 은 느려서 뺐다)."""
    from app import predictor
    from app.indicators import prepare_frame
    from app.screener import score_prediction
    predictor.ml_forecast = lambda df: None
    try:
        sub = g.iloc[: t + 1]
        df = prepare_frame([type("C", (), dict(date=r.date, open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume))
                            for r in sub.itertuples()])
        pred = predictor.predict_frame(df, sym, sym, cur)
        return score_prediction(pred)[0]
    except Exception:
        return None


NAMES = ["v1", "v2", "v3", "lowvol", "rsi_rev"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("market", choices=["KRX", "US"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--universe", type=int, default=300)
    ap.add_argument("--old-dates", type=int, default=8, help="기존 점수를 비교할 날짜 수(느림)")
    a = ap.parse_args()

    data = load(a.market)
    feats = {s: features(g) for s, g in data.items()}
    allf = pd.concat([f.assign(symbol=s) for s, f in feats.items()], ignore_index=True)
    allf = allf[(allf["tv20"] >= MIN_TV[a.market]) & (allf["close"] >= MIN_PX[a.market])]
    dates = sorted(allf["date"].unique())
    # 요인 계산에 60일+ 필요, 5일 뒤 수익 필요
    usable = [d for d in dates if allf[allf["date"] == d]["fwd5"].notna().sum() > 100 and allf[allf["date"] == d]["mom60"].notna().sum() > 100]
    print(f"{a.market}: 종목 {len(data)}개, 평가 날짜 {len(usable)}개 ({usable[0]} ~ {usable[-1]})")

    rows = []
    for d in usable:
        day = allf[allf["date"] == d].dropna(subset=["mom60", "vol20", "ret20", "px_ma60", "dist_high60", "rsi", "fwd1"]).copy()
        day = day.nlargest(a.universe, "tv20")
        if len(day) < 50:
            continue
        day["v1"] = composite(day, "v1")
        day["v2"] = composite(day, "v2")
        day["v3"] = composite(day, "v3")
        day["new"] = day["v3"]
        day["lowvol"] = -day["vol20"]
        day["rsi_rev"] = -day["rsi"]
        rec = {"date": d, "n": len(day), "base1": day["fwd1"].mean(), "base5": day["fwd5"].mean()}
        for name in NAMES:
            top = day.nlargest(a.top, name)
            rec[f"{name}_1"] = top["fwd1"].mean()
            rec[f"{name}_5"] = top["fwd5"].mean()
            rec[f"{name}_ic1"] = day[name].corr(day["fwd1"], method="spearman")
            rec[f"{name}_ic5"] = day[name].corr(day["fwd5"], method="spearman")
        rows.append(rec)
    r = pd.DataFrame(rows)
    half = len(r) // 2
    def show(sub: pd.DataFrame, label: str) -> None:
        print(f"\n[{label}] 날짜 {len(sub)}개, 평균 후보 {sub['n'].mean():.0f}개 | 전체 평균 1일 {sub['base1'].mean():+.3f}% 5일 {sub['base5'].mean():+.3f}%")
        for name in NAMES:
            print(f"  {name:9s} 상위{a.top}: 1일 {sub[name+'_1'].mean():+.3f}%  5일 {sub[name+'_5'].mean():+.3f}%  | IC 1일 {sub[name+'_ic1'].mean():+.3f}  5일 {sub[name+'_ic5'].mean():+.3f}  | 1일 승률(전체평균 초과) {(sub[name+'_1'] > sub['base1']).mean()*100:.0f}%")
    show(r, "전체")
    show(r.iloc[:half], "앞 절반")
    show(r.iloc[half:], "뒤 절반(검증)")

    # 기존 점수(규칙 부분)와 비교: 날짜를 골고루 뽑아 비교
    if a.old_dates > 0:
        cur = "KRW" if a.market == "KRX" else "USD"
        pick = [usable[i] for i in np.linspace(0, len(usable) - 1, a.old_dates).astype(int)]
        res = []
        for d in pick:
            day = allf[allf["date"] == d].dropna(subset=["mom60", "vol20", "ret20", "px_ma60", "dist_high60", "rsi", "fwd1"]).copy()
            day = day.nlargest(a.universe, "tv20")
            day["new"] = composite(day, "v3")
            day["old"] = [old_rule_score(s, data[s], int(i), cur) for s, i in zip(day["symbol"], day["idx"])]
            day = day.dropna(subset=["old"])
            top_old, top_new = day.nlargest(a.top, "old"), day.nlargest(a.top, "new")
            res.append((d, day["fwd1"].mean(), top_old["fwd1"].mean(), top_new["fwd1"].mean(),
                        day["fwd5"].mean(), top_old["fwd5"].mean(), top_new["fwd5"].mean(),
                        day["old"].corr(day["fwd1"], method="spearman"), day["new"].corr(day["fwd1"], method="spearman")))
        o = pd.DataFrame(res, columns=["date", "b1", "old1", "new1", "b5", "old5", "new5", "icold", "icnew"])
        print(f"\n[기존 규칙 점수 vs 새 점수] 날짜 {len(o)}개")
        print(o.round(3).to_string(index=False))
        print(f"평균: 전체 1일 {o.b1.mean():+.3f}% | 기존 상위 {o.old1.mean():+.3f}% | 새 상위 {o.new1.mean():+.3f}% || 5일 전체 {o.b5.mean():+.3f}% 기존 {o.old5.mean():+.3f}% 새 {o.new5.mean():+.3f}% || IC 기존 {o.icold.mean():+.3f} 새 {o.icnew.mean():+.3f}")


if __name__ == "__main__":
    main()
