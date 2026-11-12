"""주간동아 기사(2023.6.16 「AI매매프로그램으로 주식투자해 1년간 233% 수익 냈다」)의 매매 방식을 과거 데이터로 재현한다.

기사의 방식:
  1) 코스피·코스닥 전 종목에서 '내일 종가가 +2% 이상 오를' 종목을 ML 모델로 고른다(하루 400~500개, 전체의 약 15%).
  2) 가격이 싼 순서대로 1주씩 사고, 예수금이 남으면 다시 싼 종목부터 반복해서 산다.
  3) 매수가 대비 +2% 에 도달하면 판다. 손절선은 없고, 며칠이 걸려도 +2% 가 될 때까지 계속 보유한다.
기사의 주장: 예측 적중률 55~65%, 일평균 +0.8%, 1년 +233%(3,000만 원 → 1억 원).

여기서는 같은 구조의 모델(한 개의 ML 모델이 전 종목 일봉으로 다음 거래일 +2% 여부를 학습)을 걸음마(walk-forward)로 돌려
'모델이 장중이 아니라 전일 종가까지의 정보만 쓴다'는 조건을 지킨다. 신호는 T 일 종가 후에 나오고 체결은 T+1 시가다.

사용: python -m tools.donga_strategy_backtest long|broad [--start 2016-01-01] [--q 0.15] [--capital 30000000]
  long  : data/candles_max.sqlite (거래대금 상위 400종목, 2006~2026)  ← 여러 시장 국면. 상장폐지 종목이 없어 낙관적이다.
  broad : data/candles.sqlite    (2,535종목, 2026년 한 해)             ← 기사 유니버스에 가깝지만 기간이 짧다.

주의: 종가·시가는 수정주가(분할·배당 반영)라 실제 호가와 다를 수 있다. 수수료·세금·슬리피지·호가 단위는 아래 Costs 로 반영한다.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PREDICTION_DATA_DIR", ROOT / "data"))     # 캔들 DB 위치(읽기 전용으로만 연다)
OUT = Path(os.environ.get("BACKTEST_OUT", DATA))                       # 결과 CSV 를 쓸 곳


@dataclass(frozen=True)
class Costs:
    """한국 주식 기준. 수수료는 증권사 온라인 기준 편도 0.015%, 거래세는 매도에만 0.20%(농특세 포함 가정)."""
    commission: float = 0.00015
    sell_tax: float = 0.0020
    slippage: float = 0.0005   # 시장가 체결(매수·손절·기간만료)에만 적용. 지정가 익절은 지정가에 체결된다고 본다.

    @staticmethod
    def none() -> "Costs":
        return Costs(0.0, 0.0, 0.0)


# ─────────────────────────── 데이터 ───────────────────────────

def load_panel(db: str, market: str, start: str, min_days: int) -> Tuple[pd.DatetimeIndex, List[str], Dict[str, np.ndarray]]:
    con = sqlite3.connect(f"file:{DATA / db}?mode=ro", uri=True)
    df = pd.read_sql_query("select symbol,date,open,high,low,close,volume from candle where market=? and date>=? order by date",
                           con, params=(market, start))
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    counts = df.groupby("symbol")["date"].count()
    df = df[df["symbol"].isin(counts[counts >= min_days].index)]
    wide = {c: df.pivot(index="date", columns="symbol", values=c).sort_index() for c in ("open", "high", "low", "close", "volume")}
    dates = wide["close"].index
    syms = list(wide["close"].columns)
    return dates, syms, {k: v.reindex(index=dates, columns=syms).to_numpy(dtype=float) for k, v in wide.items()}


def load_index(symbol: str, start: str) -> Optional[pd.Series]:
    try:
        con = sqlite3.connect(f"file:{DATA / 'candles_max.sqlite'}?mode=ro", uri=True)
        df = pd.read_sql_query("select date,close from candle where market='IDX' and symbol=? and date>=? order by date",
                               con, params=(symbol, start))
        con.close()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["close"]
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────── 호가 단위 ───────────────────────────

def tick_of(p: np.ndarray) -> np.ndarray:
    return np.select([p < 2000, p < 5000, p < 20000, p < 50000, p < 200000, p < 500000], [1, 5, 10, 50, 100, 500], 1000).astype(float)


def ceil_tick(p: np.ndarray) -> np.ndarray:
    t = tick_of(p)
    return np.ceil(p / t - 1e-9) * t


def floor_tick(p: np.ndarray) -> np.ndarray:
    t = tick_of(p)
    return np.floor(p / t + 1e-9) * t


# ─────────────────────────── 특징·모델 ───────────────────────────

def build_features(o, h, l, c, v) -> np.ndarray:
    """(날짜, 종목, 특징) 배열. 모두 T 일 종가까지의 정보만 쓴다."""
    C = pd.DataFrame(c)
    O, H, L, V = pd.DataFrame(o), pd.DataFrame(h), pd.DataFrame(l), pd.DataFrame(v)
    r1 = C.pct_change(1, fill_method=None)
    feats = [r1]
    for n in (2, 3, 5, 10, 20):
        feats.append(C.pct_change(n, fill_method=None))
    feats.append(r1.rolling(20, min_periods=15).std())
    delta = C.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    feats.append(100 - 100 / (1 + up / dn.replace(0, np.nan)))
    for n in (5, 20, 60):
        feats.append(C / C.rolling(n, min_periods=int(n * 0.8)).mean() - 1)
    feats.append(V / V.rolling(20, min_periods=15).mean())
    feats.append((H - L) / C)
    feats.append(O / C.shift(1) - 1)
    feats.append((C - L) / (H - L).replace(0, np.nan))
    return np.stack([f.to_numpy(dtype=float) for f in feats], axis=-1)


def next_day_return(c: np.ndarray) -> np.ndarray:
    out = np.full_like(c, np.nan)
    out[:-1] = c[1:] / c[:-1] - 1
    return out


def walk_forward_picks(feat: np.ndarray, c: np.ndarray, v: np.ndarray, test_idx: range, q: float, train_days: int,
                       retrain_every: int, threshold: float, log: Callable[[str], None]) -> Tuple[Dict[int, np.ndarray], dict]:
    """매 retrain_every 거래일마다 최근 train_days 일로 다시 학습해 그다음 구간의 신호를 낸다. T 일 신호는 T 일 종가까지만 쓴다."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    fwd = next_day_return(c)
    D, S, _ = feat.shape
    picks: Dict[int, np.ndarray] = {}
    hits_up, hits_thr, n_pick, base_thr, base_n = 0, 0, 0, 0, 0
    valid_feat = np.isfinite(feat).sum(axis=-1) >= feat.shape[-1] - 3
    first = test_idx.start
    for block_start in range(first, test_idx.stop, retrain_every):
        block = range(block_start, min(block_start + retrain_every, test_idx.stop))
        tr = np.arange(max(0, block_start - train_days), block_start)     # 라벨(T→T+1)이 block_start 전에 확정된 날만
        mask = valid_feat[tr] & np.isfinite(fwd[tr]) & (v[tr] > 0)
        X = feat[tr][mask]
        y = (fwd[tr][mask] >= threshold).astype(int)
        if len(X) < 5000 or y.sum() < 50:
            continue
        model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=6, min_samples_leaf=60,
                                               class_weight="balanced", random_state=7)
        model.fit(X, y)
        for d in block:
            ok = valid_feat[d] & (v[d] > 0) & np.isfinite(c[d])
            idx = np.flatnonzero(ok)
            if len(idx) < 30:
                continue
            p = model.predict_proba(feat[d][idx])[:, 1]
            k = max(1, int(round(len(idx) * q)))
            chosen = idx[np.argsort(-p)[:k]]
            picks[d] = chosen
            f = fwd[d][chosen]
            f = f[np.isfinite(f)]
            hits_up += int((f > 0).sum())
            hits_thr += int((f >= threshold).sum())
            n_pick += len(f)
            allf = fwd[d][idx]
            allf = allf[np.isfinite(allf)]
            base_thr += int((allf >= threshold).sum())
            base_n += len(allf)
        log(f"    학습 {block_start - train_days}~{block_start} → 신호 {block.start}~{block.stop}  (표본 {len(X):,}, 양성 {y.mean():.1%})")
    diag = {
        "pickDays": len(picks),
        "avgPicksPerDay": float(np.mean([len(x) for x in picks.values()])) if picks else 0.0,
        "pickUpRate": hits_up / n_pick if n_pick else float("nan"),
        "pickHitThrRate": hits_thr / n_pick if n_pick else float("nan"),
        "baseHitThrRate": base_thr / base_n if base_n else float("nan"),
    }
    return picks, diag


def random_picks(c: np.ndarray, v: np.ndarray, days: range, q: float, seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: Dict[int, np.ndarray] = {}
    for d in days:
        idx = np.flatnonzero(np.isfinite(c[d]) & (v[d] > 0))
        if len(idx) >= 30:
            out[d] = rng.choice(idx, size=max(1, int(round(len(idx) * q))), replace=False)
    return out


def oracle_picks(c: np.ndarray, v: np.ndarray, days: range, q: float, accuracy: float, seed: int) -> Dict[int, np.ndarray]:
    """가상의 예측기: 고른 종목의 accuracy 비율이 실제로 다음 날 올랐다(나머지는 올랐든 말든 상관없이 오르지 않은 종목)."""
    rng = np.random.default_rng(seed)
    fwd = next_day_return(c)
    out: Dict[int, np.ndarray] = {}
    for d in days:
        ok = np.isfinite(c[d]) & (v[d] > 0) & np.isfinite(fwd[d])
        up, down = np.flatnonzero(ok & (fwd[d] > 0)), np.flatnonzero(ok & (fwd[d] <= 0))
        k = max(1, int(round(ok.sum() * q)))
        k_up = min(len(up), int(round(k * accuracy)))
        k_dn = min(len(down), k - k_up)
        if k_up + k_dn >= 10:
            out[d] = np.concatenate([rng.choice(up, k_up, replace=False), rng.choice(down, k_dn, replace=False)])
    return out


# ─────────────────────────── 매매 시뮬레이션 ───────────────────────────

@dataclass
class Result:
    equity: np.ndarray
    dates: pd.DatetimeIndex
    trades_closed: int
    win_rate: float
    avg_hold_days: float
    open_positions: int
    open_underwater_value_share: float
    open_value_share: float
    avg_invested_share: float
    final_net: float


def simulate(dates: pd.DatetimeIndex, o, h, l, c, picks: Dict[int, np.ndarray], first_day: int, capital: float, costs: Costs,
             take_profit: float = 0.02, stop: Optional[float] = None, max_hold: Optional[int] = None, same_day_exit: bool = False) -> Result:
    D, S = c.shape
    shares = np.zeros(S)
    paid = np.zeros(S)           # 매수 체결가 합(수수료 제외)
    age = np.zeros(S, dtype=int)
    last = np.where(np.isfinite(c[first_day - 1]), c[first_day - 1], 0.0)
    cash = capital
    equity = np.full(D, np.nan)
    invested_share: List[float] = []
    closed_ret: List[float] = []
    closed_days: List[int] = []
    comm, tax, slip = costs.commission, costs.sell_tax, costs.slippage

    for d in range(first_day, D):
        bought_today = np.zeros(S, dtype=bool)
        # 1) 전일 종가 신호 → 오늘 시가에 매수: 싼 종목부터 1주씩, 돈이 남으면 다시 싼 종목부터
        idx = picks.get(d - 1)
        if idx is not None and len(idx):
            px = o[d, idx]
            ok = np.isfinite(px) & (px > 0) & np.isfinite(c[d - 1, idx])
            idx, px = idx[ok], px[ok]
            order = np.argsort(c[d - 1, idx], kind="stable")
            idx, px = idx[order], px[order] * (1 + slip)
            unit = px * (1 + comm)
            for _ in range(500):
                n = int(np.searchsorted(np.cumsum(unit), cash, side="right"))
                if n == 0:
                    break
                cash -= float(unit[:n].sum())
                shares[idx[:n]] += 1
                paid[idx[:n]] += px[:n]
                bought_today[idx[:n]] = True
        # 2) 보유 종목 정리(오늘 장중): 손절·기간만료가 같은 날 익절과 겹치면 불리한 쪽(손절)을 먼저 적용
        held = shares > 0
        if held.any():
            avg = np.where(held, paid / np.where(held, shares, 1), np.nan)
            can = held & (~bought_today | same_day_exit) & np.isfinite(h[d]) & np.isfinite(l[d]) & np.isfinite(o[d])
            fill = np.full(S, np.nan)
            reason = np.zeros(S, dtype=int)
            if stop is not None:
                stop_px = floor_tick(avg * (1 - stop))
                hit_stop = can & (l[d] <= stop_px)
                fill = np.where(hit_stop, np.minimum(o[d], stop_px) * (1 - slip), fill)
                reason = np.where(hit_stop, 2, reason)
            tp_px = ceil_tick(avg * (1 + take_profit))
            hit_tp = can & (reason == 0) & (h[d] >= tp_px)
            fill = np.where(hit_tp, np.maximum(o[d], tp_px), fill)        # 갭 상승이면 시가에, 아니면 지정가에 체결
            reason = np.where(hit_tp, 1, reason)
            if max_hold is not None:
                expire = can & (reason == 0) & (age + 1 >= max_hold) & np.isfinite(c[d])
                fill = np.where(expire, c[d] * (1 - slip), fill)
                reason = np.where(expire, 3, reason)
            sold = reason > 0
            if sold.any():
                proceeds = shares[sold] * fill[sold] * (1 - comm - tax)
                cash += float(proceeds.sum())
                closed_ret.extend(((fill[sold] - avg[sold]) / avg[sold]).tolist())
                closed_days.extend((age[sold] + 1).tolist())
                shares[sold] = 0.0
                paid[sold] = 0.0
                age[sold] = 0
            age[shares > 0] += 1
        # 3) 종가 평가
        last = np.where(np.isfinite(c[d]), c[d], last)
        value = float((shares * last).sum())
        equity[d] = cash + value
        invested_share.append(value / equity[d] if equity[d] > 0 else 0.0)

    held = shares > 0
    value = shares * last
    avg = np.where(held, paid / np.where(held, shares, 1), np.nan)
    pnl = np.where(held, (last - avg) / avg, 0.0)
    total = float(value.sum())
    final_net = cash + total * (1 - comm - tax - slip)
    return Result(
        equity=equity, dates=dates, trades_closed=len(closed_ret),
        win_rate=float(np.mean(np.array(closed_ret) > 0)) if closed_ret else float("nan"),
        avg_hold_days=float(np.mean(closed_days)) if closed_days else float("nan"),
        open_positions=int(held.sum()),
        open_underwater_value_share=float(value[held & (pnl < 0)].sum() / total) if total > 0 else 0.0,
        open_value_share=total / equity[-1] if equity[-1] > 0 else 0.0,
        avg_invested_share=float(np.mean(invested_share)) if invested_share else 0.0,
        final_net=final_net,
    )


# ─────────────────────────── 보고 ───────────────────────────

def stats(r: Result, first_day: int, capital: float) -> dict:
    eq = r.equity[first_day:]
    daily = eq[1:] / eq[:-1] - 1
    years = (r.dates[-1] - r.dates[first_day]).days / 365.25
    peak = np.maximum.accumulate(eq)
    return {
        "최종자산(만원)": r.final_net / 1e4,
        "총수익률%": (r.final_net / capital - 1) * 100,
        "연환산%": ((r.final_net / capital) ** (1 / years) - 1) * 100 if r.final_net > 0 and years > 0 else float("nan"),
        "일평균%": float(np.nanmean(daily)) * 100,
        "최대낙폭%": float(np.nanmin(eq / peak - 1)) * 100,
        "매도건수": r.trades_closed,
        "승률%": r.win_rate * 100,
        "평균보유일": r.avg_hold_days,
        "미청산종목": r.open_positions,
        "미청산중 손실비중%": r.open_underwater_value_share * 100,
        "평균투자비중%": r.avg_invested_share * 100,
    }


def yearly(r: Result, first_day: int) -> pd.Series:
    s = pd.Series(r.equity[first_day:], index=r.dates[first_day:])
    ye = s.groupby(s.index.year).last()
    prev = pd.concat([pd.Series([s.iloc[0]], index=[ye.index[0] - 1]), ye]).shift(1).iloc[1:]
    return (ye / prev - 1) * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["long", "broad"])
    ap.add_argument("--start", default=None, help="시험 시작일(기본: long=2016-01-01, broad=전체의 뒤 40%)")
    ap.add_argument("--q", type=float, default=0.15, help="하루에 고르는 종목 비율(기사: 3000개 중 400~500개 ≈ 15%)")
    ap.add_argument("--capital", type=float, default=30_000_000)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--train-days", type=int, default=250)
    ap.add_argument("--retrain-every", type=int, default=21)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731

    if a.dataset == "long":
        dates, syms, p = load_panel("candles_max.sqlite", "KRX", "2012-01-01", 600)
        start = a.start or "2016-01-01"
    else:
        dates, syms, p = load_panel("candles.sqlite", "KRX", "2026-01-01", 60)
        start = a.start
    o, h, l, c, v = p["open"], p["high"], p["low"], p["close"], p["volume"]
    log(f"데이터: {a.dataset} | 종목 {len(syms):,} | {dates[0].date()} ~ {dates[-1].date()} ({len(dates):,}거래일)")
    feat = build_features(o, h, l, c, v)
    if start:
        first = int(dates.searchsorted(pd.Timestamp(start)))
    else:
        first = int(len(dates) * 0.6)
    first = max(first, 80)
    test = range(first, len(dates) - 1)
    train_days = min(a.train_days, first - 60)
    log(f"시험 구간: {dates[first].date()} ~ {dates[-1].date()} ({len(test):,}거래일), 학습 {train_days}일 · {a.retrain_every}일마다 재학습, 하루 상위 {a.q:.0%} 선정")

    log("  ML 모델 학습·신호 생성…")
    ml, diag = walk_forward_picks(feat, c, v, test, a.q, train_days, a.retrain_every, a.threshold, lambda s: None)
    log(f"  신호 {diag['pickDays']}일, 하루 평균 {diag['avgPicksPerDay']:.0f}종목 | 고른 종목의 다음날 상승률(적중) {diag['pickUpRate']:.1%} | "
        f"다음날 +{a.threshold:.0%} 이상 {diag['pickHitThrRate']:.1%} (전체 평균 {diag['baseHitThrRate']:.1%})")

    cost = Costs()
    rows: Dict[str, Result] = {}

    def run(name: str, picks: Dict[int, np.ndarray], costs: Costs = cost, **kw) -> Result:
        r = simulate(dates, o, h, l, c, picks, first, a.capital, costs, take_profit=a.threshold, **kw)
        rows[name] = r
        return r

    run("① ML 모델 · 기사 규칙(손절 없음) · 비용 반영", ml)
    run("② ML 모델 · 기사 규칙 · 비용 없음(기사 가정)", ml, Costs.none())
    run("③ ML 모델 · 기사 규칙 · 첫날 장중 익절 허용", ml, same_day_exit=True)
    run("④ ML 모델 · 손절 -5% · 최대 5일 보유", ml, stop=0.05, max_hold=5)
    for acc in (0.55, 0.65):
        res = [simulate(dates, o, h, l, c, oracle_picks(c, v, test, a.q, acc, 100 + s), first, a.capital, cost, take_profit=a.threshold)
               for s in range(a.seeds)]
        rows[f"⑤ 가상 예측기 적중 {acc:.0%} · 기사 규칙"] = min(res, key=lambda r: abs(r.final_net - float(np.median([x.final_net for x in res]))))
    res = [simulate(dates, o, h, l, c, random_picks(c, v, test, a.q, 200 + s), first, a.capital, cost, take_profit=a.threshold) for s in range(a.seeds)]
    rows["⑥ 무작위 선정 · 기사 규칙"] = min(res, key=lambda r: abs(r.final_net - float(np.median([x.final_net for x in res]))))
    res = [simulate(dates, o, h, l, c, random_picks(c, v, test, a.q, 300 + s), first, a.capital, cost, take_profit=a.threshold, stop=0.05, max_hold=5)
           for s in range(a.seeds)]
    rows["⑦ 무작위 선정 · 손절 -5% · 최대 5일"] = min(res, key=lambda r: abs(r.final_net - float(np.median([x.final_net for x in res]))))

    table = pd.DataFrame({k: stats(r, first, a.capital) for k, r in rows.items()}).T
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    print("\n== 전략별 결과 (자본 {:,.0f}원)".format(a.capital))
    print(table.round(1).to_string())

    # 기준선: 같은 구간에서 시험 종목 전체를 동일 비중으로 들고 있었다면
    rets = c[first:] / c[first - 1:-1] - 1
    ew = np.nanmean(np.where(np.isfinite(rets), rets, np.nan), axis=1)
    ew_total = float(np.prod(1 + np.nan_to_num(ew)) - 1) * 100
    log(f"\n기준선: 같은 종목 동일비중 보유(매일 재조정, 비용 제외) {ew_total:+.1f}% | 일평균 {np.nanmean(ew) * 100:+.3f}%")
    kospi = load_index("^KS11", str(dates[first].date()))
    if kospi is not None and len(kospi) > 10:
        log(f"기준선: 코스피 지수 {kospi.iloc[-1] / kospi.iloc[0] * 100 - 100:+.1f}% ({kospi.index[0].date()} ~ {kospi.index[-1].date()})")

    if a.dataset == "long":
        y = pd.DataFrame({k: yearly(rows[k], first) for k in list(rows)[:1] + [n for n in rows if n.startswith("④")]}).round(1)
        y["동일비중 보유"] = pd.Series(ew, index=dates[first + 0:first + len(ew)]).groupby(lambda x: x.year).apply(lambda s: (np.prod(1 + s.fillna(0)) - 1) * 100).round(1)
        print("\n== 연도별 수익률(%) — 시장 국면에 따라 얼마나 달라지는가")
        print(y.to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"donga_backtest_{a.dataset}.csv"
    table.round(3).to_csv(out, encoding="utf-8-sig")
    log(f"\n저장: {out}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
