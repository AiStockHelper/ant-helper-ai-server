"""'거래 손익'을 직접 예측하도록 학습시킨 ML 로 같은 매매 규칙을 돌려 본다(donga_strategy_backtest 의 후속).

기존 모델의 목표는 '다음 날 종가가 +2% 이상인가'였다. 그런데 실제 거래는 '다음 날 시가 매수 → 익절 +2% 에 닿으면 매도, 아니면(손절선에 닿거나) 5영업일째 종가 정리' 이다.
목표가 달라서 변동성 큰 종목만 골라 승률은 높아도 지는 거래가 훨씬 컸다. 여기서는 라벨을 그 거래의 실제 손익(비용 전)으로 바꿔 회귀 모델로 학습한다.

사용: python -m tools.donga_payoff_backtest kr|us [--q 0.15 0.03] [--stop 0.02]
  kr: data/candles_krx_all.sqlite (상장폐지 포함), us: data/candles_us_sp500.sqlite (그날의 S&P 500 구성종목)
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import tools.donga_strategy_backtest as B
from tools.donga_strategy_backtest import (Costs, build_features, load_index, load_panel, load_status, member_mask, random_picks, regime_signal,
                                           simulate, stats, yearly)


def payoff_labels(o, h, l, c, tp: float, stop: Optional[float], hold: int, seed: int = 11) -> np.ndarray:
    """(신호일 d, 종목) → d+1 시가에 사서 simulate() 와 같은 규칙으로 정리했을 때의 거래 손익률(비용 전).
    매수일(d+1)에는 청산하지 않고, d+2 ~ d+1+hold 일에 익절/손절을 보며, 못 닿으면 d+1+hold 일 종가에 판다.
    같은 날 둘 다 닿으면 갭이 있는 쪽, 아니면 동전 던지기. 라벨이 확정되지 않은 마지막 날들은 NaN."""
    D, S = c.shape
    n = D - (hold + 1)
    y = np.full((D, S), np.nan)
    if n <= 0:
        return y
    rng = np.random.default_rng(seed)
    E = o[1:n + 1]
    out = np.full((n, S), np.nan)
    done = ~(np.isfinite(E) & (E > 0))
    tpx = E * (1 + tp)
    spx = E * (1 - stop) if stop is not None else None
    for j in range(2, hold + 2):
        O, H, L, C = (a[j:j + n] for a in (o, h, l, c))
        ok = np.isfinite(O) & np.isfinite(H) & np.isfinite(L) & ~done
        hit_tp = ok & (H >= tpx)
        if stop is not None:
            hit_stop = ok & (L <= spx)
            both = hit_stop & hit_tp
            coin = rng.random((n, S)) < 0.5
            use_stop = (hit_stop & ~hit_tp) | (both & ((O <= spx) | (~(O >= tpx) & coin)))
            hit_tp = hit_tp & ~use_stop
            out = np.where(use_stop, np.minimum(O, spx) / E - 1, out)
            done = done | use_stop
        out = np.where(hit_tp, np.maximum(O, tpx) / E - 1, out)
        done = done | hit_tp
    Cf = c[hold + 1:hold + 1 + n]
    last = ~done & np.isfinite(Cf)
    out = np.where(last, Cf / E - 1, out)
    y[:n] = out
    return y


def walk_forward_scores(feat: np.ndarray, y: np.ndarray, c: np.ndarray, v: np.ndarray, test_idx: range, train_days: int, retrain_every: int, hold: int,
                        mask: Optional[np.ndarray] = None, clip: Tuple[float, float] = (-0.25, 0.05)) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """매 retrain_every 거래일마다 최근 train_days 일의 '확정된' 거래 손익 라벨로 회귀 모델을 학습하고, 날짜별 (종목 인덱스, 예상 손익)을 낸다."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    valid_feat = np.isfinite(feat).sum(axis=-1) >= feat.shape[-1] - 3
    if mask is not None:
        valid_feat &= mask
    scores: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for block_start in range(test_idx.start, test_idx.stop, retrain_every):
        block = range(block_start, min(block_start + retrain_every, test_idx.stop))
        # 라벨은 d+1+hold 일 종가까지 봐야 확정된다 → 그 전에 확정된 날만 학습에 쓴다
        tr = np.arange(max(0, block_start - train_days), block_start - (hold + 1))
        m = valid_feat[tr] & np.isfinite(y[tr]) & (v[tr] > 0)
        X = feat[tr][m]
        t = np.clip(y[tr][m], *clip)
        if len(X) < 5000:
            continue
        model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, max_depth=5, min_samples_leaf=100, l2_regularization=1.0, random_state=7)
        model.fit(X, t)
        for d in block:
            idx = np.flatnonzero(valid_feat[d] & (v[d] > 0) & np.isfinite(c[d]))
            if len(idx) >= 30:
                scores[d] = (idx, model.predict(feat[d][idx]))
    return scores


def picks_from_scores(scores: Dict[int, Tuple[np.ndarray, np.ndarray]], q: float) -> Dict[int, np.ndarray]:
    out = {}
    for d, (idx, p) in scores.items():
        k = max(1, int(round(len(idx) * q)))
        out[d] = idx[np.argsort(-p)[:k]]
    return out


def edge(scores, y, qs) -> pd.DataFrame:
    """고른 종목의 실제 거래 손익(비용 전)이 같은 날 전체 평균보다 얼마나 좋은가."""
    rows = {}
    for q in qs:
        pk, un, hit = [], [], []
        for d, (idx, p) in scores.items():
            yy = y[d, idx]
            ok = np.isfinite(yy)
            if ok.sum() < 30:
                continue
            k = max(1, int(round(len(idx) * q)))
            top = np.argsort(-p)[:k]
            top = top[ok[top]]
            if len(top) == 0:
                continue
            pk.append(yy[top].mean())
            un.append(yy[ok].mean())
            hit.append((yy[top] > 0).mean())
        rows[f"상위 {q:.0%}"] = {"선정 평균 손익%": np.mean(pk) * 100, "전체 평균 손익%": np.mean(un) * 100, "차이%p": (np.mean(pk) - np.mean(un)) * 100,
                                "선정 승률%": np.mean(hit) * 100, "일수": len(pk)}
    return pd.DataFrame(rows).T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("market", choices=["kr", "us"])
    ap.add_argument("--q", type=float, nargs="+", default=[0.15, 0.03])
    ap.add_argument("--stop", type=float, default=0.02)
    ap.add_argument("--tp", type=float, default=0.02)
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--train-days", type=int, default=250)
    ap.add_argument("--retrain-every", type=int, default=21)
    ap.add_argument("--max-positions", type=int, default=20)
    ap.add_argument("--adv-cap", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    kr = a.market == "kr"
    if kr:
        dates, syms, p = load_panel("candles_krx_all.sqlite", "KRX", "2014-01-01", 120)
        cost, capital, usd = Costs(), a.capital or 30_000_000.0, False
        status = load_status("candles_krx_all.sqlite")
        delisted = np.array([status.get(s) == "DELISTED" for s in syms])
        idx_series = load_index("^KS11", "2013-06-01")
        idx_name = "코스피"
    else:
        dates, syms, p = load_panel("candles_us_sp500.sqlite", "US", "2012-01-01", 120)
        cost, capital, usd = Costs(commission=0.0001, sell_tax=0.00003, slippage=0.0005), a.capital or 50_000.0, True
        idx_series = load_index("^GSPC", "2011-06-01", "candles_us_sp500.sqlite")
        idx_name = "S&P500"
    o, h, l, c, v = p["open"], p["high"], p["low"], p["close"], p["volume"]
    first = max(int(dates.searchsorted(pd.Timestamp("2016-01-01"))), 80)
    test = range(first, len(dates) - 1)
    train_days = min(a.train_days, first - 60)
    has = np.isfinite(c) & (v > 0)
    adv = pd.DataFrame(c * v).rolling(20, min_periods=15).mean().to_numpy()
    print(f"데이터: {a.market} | 종목 {len(syms):,} | {dates[0].date()} ~ {dates[-1].date()} | 시험 {dates[first].date()} ~ | 자본 {capital:,.0f}", flush=True)

    universes: Dict[str, Optional[np.ndarray]]
    if kr:
        universes = {"전종목(상장폐지 포함)": None}
    else:
        members = member_mask(dates, syms, "candles_us_sp500.sqlite")
        rank = pd.DataFrame(np.where(members & has, adv, np.nan)).rank(axis=1, ascending=False).to_numpy()
        universes = {"S&P500 그날 구성종목": members & has, "구성종목 중 거래대금 상위 100": members & has & (rank <= 100)}

    off = regime_signal(idx_series.reindex(dates).ffill().to_numpy(), "ma60")
    feat = build_features(o, h, l, c, v)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["총수익률%", "연환산%", "최대낙폭%", "승률%", "거래당순수익%", "평균투자비중%", "매도건수"]
    allrows: Dict[str, B.Result] = {}

    def run(pk, stop, off_=None, costs=cost):
        return simulate(dates, o, h, l, c, pk, first, capital, costs, take_profit=a.tp, stop=stop, max_hold=a.hold, skip_held=True, alloc="equal",
                        max_positions=a.max_positions, same_bar="random", adv=adv, adv_cap=a.adv_cap, risk_off=off_, delist_recovery=1.0, usd=usd)

    for uname, um in universes.items():
        for label_name, stop in (("손절 없음", None), (f"손절 -{round(a.stop * 100)}%", a.stop)):
            y = payoff_labels(o, h, l, c, a.tp, stop, a.hold)
            print(f"\n##### [{uname}] 라벨 = 거래 손익({label_name}) 회귀 학습 …", flush=True)
            scores = walk_forward_scores(feat, y, c, v, test, train_days, a.retrain_every, a.hold, mask=um)
            print(f"== 선정 종목의 실제 거래 손익(비용 전) — 같은 날 전체 평균과 비교, 모델이 우위를 가졌는가")
            print(edge(scores, y, a.q).round(3).to_string(), flush=True)
            rows: Dict[str, B.Result] = {}
            for q in a.q:
                pk = picks_from_scores(scores, q)
                rows[f"상위 {q:.0%} · 필터 없음"] = run(pk, stop)
                rows[f"상위 {q:.0%} · {idx_name} 60일선"] = run(pk, stop, off)
                rows[f"상위 {q:.0%} · 비용 없음"] = run(pk, stop, costs=Costs.none())
                if q == a.q[0]:
                    res = [run(random_picks(c, v, test, q, 900 + s, mask=um), stop) for s in range(a.seeds)]
                    mid = float(np.median([x.final_net for x in res]))
                    rows["[비교] 무작위 선정"] = min(res, key=lambda r: abs(r.final_net - mid))
                    if kr:
                        sv = {d: i[~delisted[i]] for d, i in pk.items()}
                        rows["[비교] 상장폐지 제외(생존자만)"] = run(sv, stop)
            t = pd.DataFrame({k: stats(r, first, capital) for k, r in rows.items()}).T
            print(f"\n== [{uname}] 거래 손익 모델 · 익절 +{a.tp:.0%} · {label_name} · {a.hold}영업일 정리 · 균등 {a.max_positions}종목 · 비용 반영")
            print(t[cols].round(1).to_string(), flush=True)
            for k, r in rows.items():
                allrows[f"{uname} | {label_name} | {k}"] = r
    B.OUT.mkdir(parents=True, exist_ok=True)
    out = B.OUT / f"donga_payoff_{a.market}.csv"
    pd.DataFrame({k: stats(r, first, capital) for k, r in allrows.items()}).T.round(4).to_csv(out, encoding="utf-8-sig")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    sys.path.insert(0, str(B.ROOT))
    main()
