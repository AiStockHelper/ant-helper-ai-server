"""학습 데이터 길이(160일 vs 500일)와 모델 방식(종목별 vs 전 종목 합친 모델)을 걸어가며(walk-forward) 비교한다.

날짜 T 마다 T 까지의 데이터만 써서 '다음 거래일 수익률'을 예측하고, 실제 다음 날 수익률과 비교한다.
  S160 : 서비스의 현재 방식 — 종목별로 최근 160일만 보고 규칙+GradientBoosting 예측(predict_frame)
  S500 : 같은 방식, 종목별 최근 500일(약 2년)
  P160 / P500 : 전 종목을 합쳐 학습한 하나의 모델(최근 160/500일 구간, 날짜별 순위 정규화, 목표=그날 평균 대비 초과 수익)
지표: IC(예측과 실제 수익의 순위상관), 상위 10종목의 다음 날 수익 - 후보 전체 평균, UP 예측의 적중률(기준: 후보 중 오른 종목 비율)
사용: python -m tools.backtest_models KRX|US [--universe 150] [--dates 30] [--workers 4]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
DB = ROOT / "data" / "candles_long.sqlite"
FEATURES = None  # predictor.ML_FEATURES (지연 import)


def load(market: str) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("select symbol,date,open,high,low,close,volume from candle where market=? order by symbol,date",
                           con, params=(market,))
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.drop_duplicates("date").reset_index(drop=True)
        g["date"] = pd.to_datetime(g["date"])
        if len(g) >= 520:
            out[sym] = g
    return out


def _predict_one(args):
    sym, w160, w500 = args
    from app.predictor import predict_frame
    res = {}
    for key, w in (("S160", w160), ("S500", w500)):
        try:
            r = predict_frame(w, sym, sym, "KRW")
            res[key] = (r.predicted_change_rate, r.direction, r.confidence)
        except Exception:
            res[key] = None
    return sym, res


def build_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from app.indicators import add_indicators
    from app.predictor import ML_FEATURES, _build_features
    frames = []
    for sym, g in data.items():
        ind = add_indicators(g)
        f = _build_features(ind)
        f["date"] = g["date"].values
        f["symbol"] = sym
        f["close"] = g["close"].values
        f["tv20"] = (g["close"] * g["volume"]).rolling(20, min_periods=15).mean().values
        f["fwd1"] = (g["close"].shift(-1) / g["close"] - 1).values
        f["fwd5"] = (g["close"].shift(-5) / g["close"] - 1).values
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    # 날짜별 순위(0~1) 정규화: 시장 전체 분위기·변동성 수준 차이를 없앤다
    ranks = panel.groupby("date")[ML_FEATURES].rank(pct=True)
    ranks.columns = [c + "_r" for c in ranks.columns]
    panel = pd.concat([panel, ranks], axis=1)
    panel["excess1"] = panel["fwd1"] - panel.groupby("date")["fwd1"].transform("mean")
    return panel


def pooled_predict(panel: pd.DataFrame, T, window: int, dates: list) -> pd.Series:
    """T 이전(T-1 까지 목표가 확정된) window 일로 학습해 T 일의 초과수익을 예측한다."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from app.predictor import ML_FEATURES
    cols = [c + "_r" for c in ML_FEATURES]
    idx = dates.index(T)
    train_dates = dates[max(0, idx - window): idx]          # T 미만, 마지막 날(T-1)의 목표는 T 종가로 확정
    tr = panel[panel["date"].isin(train_dates)].dropna(subset=cols + ["excess1"])
    tr = tr[tr["excess1"].abs() < 0.25]
    model = HistGradientBoostingRegressor(max_iter=150, max_depth=4, learning_rate=0.05, l2_regularization=1.0,
                                          min_samples_leaf=200, random_state=7)
    model.fit(tr[cols].to_numpy(), tr["excess1"].to_numpy())
    day = panel[panel["date"] == T].dropna(subset=cols)
    return pd.Series(model.predict(day[cols].to_numpy()), index=day["symbol"].values)


def metrics(pred: pd.Series, day: pd.DataFrame, top: int = 10) -> dict:
    d = day.set_index("symbol").loc[pred.index].dropna(subset=["fwd1"])
    p = pred.loc[d.index]
    if len(d) < 30:
        return {}
    base = d["fwd1"].mean()
    topn = p.nlargest(top).index
    up = p > 0
    return {
        "ic1": p.corr(d["fwd1"], method="spearman"),
        "ic5": p.corr(d["fwd5"], method="spearman") if d["fwd5"].notna().sum() > 30 else np.nan,
        "top1": d.loc[topn, "fwd1"].mean() - base,
        "top5": d.loc[topn, "fwd5"].mean() - d["fwd5"].mean() if d["fwd5"].notna().sum() > 30 else np.nan,
        "hit": (d.loc[up, "fwd1"] > 0).mean() if up.sum() >= 3 else np.nan,       # 상승 예측 적중률
        "base_up": (d["fwd1"] > 0).mean(),                                         # 후보 중 실제로 오른 비율(기준)
        "n_up": int(up.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("market", choices=["KRX", "US"])
    ap.add_argument("--universe", type=int, default=150)
    ap.add_argument("--dates", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    data = load(a.market)
    print(f"{a.market}: 종목 {len(data)}개 (일봉 520개 이상)")
    panel = build_panel(data)
    all_dates = sorted(panel["date"].unique())
    # 500일 학습이 가능한 날짜(앞에서 500일 이상)이고 다음 날 수익이 있는 날짜 중 최근 구간에서 균등 추출
    cand = [d for d in all_dates[505:-6]]
    step = max(1, len(cand) // a.dates)
    test_dates = cand[::step][-a.dates:]
    print(f"테스트 날짜 {len(test_dates)}개: {pd.Timestamp(test_dates[0]).date()} ~ {pd.Timestamp(test_dates[-1]).date()}")

    results = {k: [] for k in ("S160", "S500", "P160", "P500")}
    for n, T in enumerate(test_dates, 1):
        day = panel[(panel["date"] == T)].dropna(subset=["tv20", "fwd1"])
        uni = day.nlargest(a.universe, "tv20")["symbol"].tolist()
        tasks = []
        for s in uni:
            g = data[s]
            i = int(g.index[g["date"] == T][0])
            tasks.append((s, g.iloc[i - 159:i + 1].reset_index(drop=True), g.iloc[i - 499:i + 1].reset_index(drop=True)))
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            outs = list(pool.map(_predict_one, tasks, chunksize=8))
        for key in ("S160", "S500"):
            pred = pd.Series({s: r[key][0] for s, r in outs if r.get(key)})
            m = metrics(pred, day)
            if m:
                m["acc_dir"] = np.nan
                results[key].append(m)
        dates_list = [pd.Timestamp(d) for d in all_dates]
        Tt = pd.Timestamp(T)
        for key, w in (("P160", 160), ("P500", 500)):
            pred = pooled_predict(panel, Tt, w, dates_list)
            pred = pred[pred.index.isin(uni)]
            m = metrics(pred, day)
            if m:
                results[key].append(m)
        print(f"  {n}/{len(test_dates)} {pd.Timestamp(T).date()} 완료", flush=True)

    print(f"\n=== {a.market} 결과 (날짜 {len(test_dates)}개, 날짜마다 유동성 상위 {a.universe}종목) ===")
    print(f"{'모델':6s} {'IC 1일':>14s} {'IC 5일':>9s} {'상위10 초과 1일':>16s} {'상위10 초과 5일':>16s} {'상승예측 적중':>12s} {'(기준 상승비율)':>14s} {'상승예측 수/일':>12s}")
    for key, rows in results.items():
        r = pd.DataFrame(rows)
        if r.empty:
            continue
        def ms(col, pct=False):
            x = r[col].dropna()
            mean, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
            return (mean * 100, se * 100) if pct else (mean, se)
        ic1, ic1se = ms("ic1"); ic5, _ = ms("ic5"); t1, t1se = ms("top1", True); t5, t5se = ms("top5", True)
        print(f"{key:6s} {ic1:+7.3f}±{ic1se:.3f} {ic5:+9.3f} {t1:+9.3f}%±{t1se:.2f} {t5:+9.3f}%±{t5se:.2f} "
              f"{r['hit'].mean()*100:11.1f}% {r['base_up'].mean()*100:13.1f}% {r['n_up'].mean():12.1f}")
    pd.concat({k: pd.DataFrame(v) for k, v in results.items()}).to_csv(ROOT / "data" / f"backtest_models_{a.market}.csv")


if __name__ == "__main__":
    main()
