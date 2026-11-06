"""요인별 예측력(IC) 점검: 날짜마다 유동성 상위 종목에서 요인 순위와 다음 1·5거래일 수익률의 순위상관을 구한다.
사용: python -m tools.factor_ic KRX|US [--universe 300]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.backtest_factors import MIN_PX, MIN_TV, features, load  # noqa: E402

FACTORS = {
    "mom60 (60일 수익, 최근5일 제외)": "mom60",
    "ret20 (20일 수익)": "ret20",
    "-ret5 (5일 반전)": "-ret5",
    "-ret1 (당일 반전)": "-ret1",
    "-rsi (과매도)": "-rsi",
    "px_ma20 (20일선 위)": "px_ma20",
    "-px_ma20 (20일선 아래)": "-px_ma20",
    "px_ma60 (60일선 위)": "px_ma60",
    "align (정배열)": "align",
    "volratio (거래량 급증)": "volratio",
    "dist_high60 (60일 고점 근접)": "dist_high60",
    "-dist_high60 (고점서 멀리)": "-dist_high60",
    "-vol20 (저변동)": "-vol20",
    "quality (변동성 대비 20일 수익)": "quality",
    "tv20 (거래대금 큼)": "tv20",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("market", choices=["KRX", "US"])
    ap.add_argument("--universe", type=int, default=300)
    a = ap.parse_args()
    data = load(a.market)
    allf = pd.concat([features(g).assign(symbol=s) for s, g in data.items()], ignore_index=True)
    allf["quality"] = allf["ret20"] / allf["vol20"].clip(lower=0.5)
    allf = allf[(allf["tv20"] >= MIN_TV[a.market]) & (allf["close"] >= MIN_PX[a.market])]
    need = ["mom60", "vol20", "ret20", "px_ma60", "dist_high60", "rsi", "ret5", "ret1", "px_ma20", "volratio", "fwd1", "fwd5"]
    allf = allf.dropna(subset=need)
    dates = sorted(allf["date"].unique())
    out = {k: [] for k in FACTORS}
    meta = []
    for d in dates:
        day = allf[allf["date"] == d]
        day = day.nlargest(a.universe, "tv20")
        if len(day) < 80 or day["fwd5"].notna().sum() < 80:
            continue
        meta.append(d)
        for label, expr in FACTORS.items():
            x = -day[expr[1:]] if expr.startswith("-") else day[expr]
            out[label].append((x.corr(day["fwd1"], method="spearman"), x.corr(day["fwd5"], method="spearman")))
    n = len(meta)
    print(f"{a.market}: 평가 날짜 {n}개 ({meta[0]} ~ {meta[-1]}), 날짜당 유동성 상위 {a.universe}종목")
    print(f"{'요인':32s} {'IC 1일':>8s} {'t':>6s} {'IC 5일':>8s} {'t':>6s} | {'앞절반 5일':>10s} {'뒤절반 5일':>10s}")
    for label, vals in out.items():
        v = np.array(vals)
        h = len(v) // 2
        def t(x):
            return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        print(f"{label:32s} {v[:,0].mean():+8.3f} {t(v[:,0]):6.1f} {v[:,1].mean():+8.3f} {t(v[:,1]):6.1f} | {v[:h,1].mean():+10.3f} {v[h:,1].mean():+10.3f}")


if __name__ == "__main__":
    main()
