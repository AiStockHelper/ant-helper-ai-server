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


def load_index(symbol: str, start: str, db: str = "candles_max.sqlite") -> Optional[pd.Series]:
    try:
        con = sqlite3.connect(f"file:{DATA / db}?mode=ro", uri=True)
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
    return np.stack([f.to_numpy(dtype=np.float32) for f in feats], axis=-1)    # 3천 종목 × 12년이라 float32 로 메모리를 아낀다


def next_day_return(c: np.ndarray) -> np.ndarray:
    out = np.full_like(c, np.nan)
    out[:-1] = c[1:] / c[:-1] - 1
    return out


def walk_forward_picks(feat: np.ndarray, c: np.ndarray, v: np.ndarray, test_idx: range, q: float, train_days: int,
                       retrain_every: int, threshold: float, log: Callable[[str], None],
                       mask: Optional[np.ndarray] = None) -> Tuple[Dict[int, np.ndarray], dict]:
    """매 retrain_every 거래일마다 최근 train_days 일로 다시 학습해 그다음 구간의 신호를 낸다. T 일 신호는 T 일 종가까지만 쓴다."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    fwd = next_day_return(c)
    D, S, _ = feat.shape
    picks: Dict[int, np.ndarray] = {}
    hits_up, hits_thr, n_pick, base_thr, base_n = 0, 0, 0, 0, 0
    valid_feat = np.isfinite(feat).sum(axis=-1) >= feat.shape[-1] - 3
    if mask is not None:       # 그날의 유니버스(예: 그날의 S&P 500 구성종목)만 학습·선정에 쓴다
        valid_feat &= mask
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


def random_picks(c: np.ndarray, v: np.ndarray, days: range, q: float, seed: int, mask: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: Dict[int, np.ndarray] = {}
    for d in days:
        idx = np.flatnonzero(np.isfinite(c[d]) & (v[d] > 0) & (mask[d] if mask is not None else True))
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
            out[d] = rng.permutation(np.concatenate([rng.choice(up, k_up, replace=False), rng.choice(down, k_dn, replace=False)]))
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
    exits_take_profit: int = 0
    exits_stop: int = 0
    exits_expire: int = 0
    exits_regime: int = 0
    exits_delist: int = 0
    avg_trade_net: float = float("nan")
    closed_pnl: float = 0.0
    closed_cost: float = 0.0
    final_cash: float = 0.0
    open_cost: float = 0.0


def simulate(dates: pd.DatetimeIndex, o, h, l, c, picks: Dict[int, np.ndarray], first_day: int, capital: float, costs: Costs,
             take_profit: float = 0.02, stop: Optional[float] = None, max_hold: Optional[int] = None, same_day_exit: bool = False,
             skip_held: bool = False, alloc: str = "shares", max_positions: int = 20, same_bar: str = "stop",
             adv: Optional[np.ndarray] = None, adv_cap: float = 0.0, risk_off: Optional[np.ndarray] = None,
             delist_recovery: Optional[float] = None, usd: bool = False) -> Result:
    """risk_off[t] 가 True 면 t 일 종가 후 '급락기'로 판단한 것이다: t+1 시가에 보유 종목을 모두 팔고(시장가, 슬리피지 반영) 신규 매수를 멈춘다.
    delist_recovery 가 있으면 데이터가 끝난(상장폐지) 종목을 들고 있을 때 마지막 종가 × 그 비율로 정리한다(None 이면 마지막 종가로 계속 평가)."""
    D, S = c.shape
    if usd:       # 미국 주식은 호가 단위가 1센트
        ceil_tick, floor_tick = (lambda p: np.ceil(p * 100 - 1e-9) / 100), (lambda p: np.floor(p * 100 + 1e-9) / 100)    # noqa: F811
    else:
        ceil_tick, floor_tick = globals()["ceil_tick"], globals()["floor_tick"]
    valid_c = np.isfinite(c)
    last_idx = np.where(valid_c.any(axis=0), D - 1 - np.argmax(valid_c[::-1], axis=0), -1)     # 종목별 마지막 시세가 있는 날
    shares = np.zeros(S)
    paid = np.zeros(S)           # 매수 체결가 합(수수료 제외)
    age = np.zeros(S, dtype=int)
    last = np.where(np.isfinite(c[first_day - 1]), c[first_day - 1], 0.0)
    cash = capital
    equity = np.full(D, np.nan)
    invested_share: List[float] = []
    closed_ret: List[float] = []
    closed_days: List[int] = []
    closed_net: List[float] = []
    pnl_sum = 0.0
    cost_sum = 0.0
    exit_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    comm, tax, slip = costs.commission, costs.sell_tax, costs.slippage
    rng = np.random.default_rng(12345)

    for d in range(first_day, D):
        bought_today = np.zeros(S, dtype=bool)
        # 1) 전일 종가 신호 → 오늘 시가에 매수: 싼 종목부터 1주씩, 돈이 남으면 다시 싼 종목부터
        idx = picks.get(d - 1)
        regime_off = risk_off is not None and bool(risk_off[d - 1])
        if regime_off:
            idx = None                           # 급락기 판단 중에는 새로 사지 않는다
        if idx is not None and skip_held:
            idx = idx[shares[idx] == 0]          # 이미 들고 있는 종목은 더 사지 않는다(매수별 보유 기간이 어긋나지 않게)
        if alloc == "equal" and idx is not None and len(idx):
            # 종목별 균등 금액: 자산의 1/max_positions 씩, 신호 순위(확률 높은 순)대로 빈 자리만큼 산다
            free = max_positions - int((shares > 0).sum())
            if free > 0:
                slot = (cash + float((shares * last).sum())) / max_positions
                for k in idx:
                    if free <= 0:
                        break
                    if not (np.isfinite(o[d, k]) and o[d, k] > 0) or shares[k] > 0:
                        continue
                    unit_k = o[d, k] * (1 + slip) * (1 + comm)
                    qty = int(min(slot, cash) // unit_k)
                    if adv is not None and adv_cap > 0:
                        room = adv[d - 1, k]
                        qty = min(qty, int(adv_cap * room // unit_k) if np.isfinite(room) else 0)
                    if qty >= 1:
                        cash -= qty * unit_k
                        shares[k] += qty
                        paid[k] += o[d, k] * (1 + slip) * qty
                        bought_today[k] = True
                        free -= 1
            idx = None
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
            tp_px = ceil_tick(avg * (1 + take_profit))
            hit_tp = can & (h[d] >= tp_px)
            if stop is not None:
                stop_px = floor_tick(avg * (1 - stop))
                hit_stop = can & (l[d] <= stop_px)
                both = hit_stop & hit_tp
                # 일봉으로는 같은 날 손절·익절 중 무엇이 먼저인지 알 수 없다. 갭(시가가 이미 한쪽을 넘음)이면 그쪽이 먼저이고,
                # 그 외에는 same_bar 가정을 따른다: stop=손절 먼저(보수적), target=익절 먼저(낙관적), random=동전 던지기(기댓값)
                gap_down = o[d] <= stop_px
                gap_up = o[d] >= tp_px
                if same_bar == "stop":
                    first_stop = np.ones(S, dtype=bool)
                elif same_bar == "target":
                    first_stop = np.zeros(S, dtype=bool)
                else:
                    first_stop = rng.random(S) < 0.5
                final_stop = (hit_stop & ~hit_tp) | (both & (gap_down | (~gap_up & first_stop)))
                hit_tp = hit_tp & ~final_stop
                fill = np.where(final_stop, np.minimum(o[d], stop_px) * (1 - slip), fill)
                reason = np.where(final_stop, 2, reason)
            fill = np.where(hit_tp, np.maximum(o[d], tp_px), fill)        # 갭 상승이면 시가에, 아니면 지정가에 체결
            reason = np.where(hit_tp & (reason == 0), 1, reason)
            if max_hold is not None:
                # age 는 '매수일 이후 지난 영업일 수'다(매수일 종가 처리 뒤 1). max_hold=5 면 매수일로부터 5영업일째 종가에 정리한다.
                expire = can & (reason == 0) & (age >= max_hold) & np.isfinite(c[d])
                fill = np.where(expire, c[d] * (1 - slip), fill)
                reason = np.where(expire, 3, reason)
            if regime_off:
                to_cash = held & np.isfinite(o[d]) & (o[d] > 0)
                fill = np.where(to_cash, o[d] * (1 - slip), fill)
                reason = np.where(to_cash, 4, reason)
            if delist_recovery is not None:
                gone = held & ~np.isfinite(c[d]) & (d > last_idx)
                fill = np.where(gone, last * delist_recovery, fill)
                reason = np.where(gone, 5, reason)
            sold = reason > 0
            if sold.any():
                proceeds = shares[sold] * fill[sold] * (1 - comm - tax)
                cash += float(proceeds.sum())
                closed_ret.extend(((fill[sold] - avg[sold]) / avg[sold]).tolist())
                # 비용을 모두 뺀 거래당 수익률: 매수 수수료는 평균가에 얹고, 매도는 수수료·세금을 뺀다
                closed_net.extend(((fill[sold] * (1 - comm - tax) - avg[sold] * (1 + comm)) / (avg[sold] * (1 + comm))).tolist())
                pnl_sum += float((shares[sold] * (fill[sold] * (1 - comm - tax) - avg[sold] * (1 + comm))).sum())
                cost_sum += float((shares[sold] * avg[sold] * (1 + comm)).sum())
                for code in reason[sold]:
                    exit_counts[int(code)] += 1
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
        exits_take_profit=exit_counts[1], exits_stop=exit_counts[2], exits_expire=exit_counts[3],
        exits_regime=exit_counts[4], exits_delist=exit_counts[5],
        avg_trade_net=float(np.mean(closed_net)) if closed_net else float("nan"),
        closed_pnl=pnl_sum, closed_cost=cost_sum, final_cash=cash, open_cost=float((paid * (1 + comm)).sum()),
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
        "거래당순수익%": r.avg_trade_net * 100,
        "금액가중수익%": (r.closed_pnl / r.closed_cost * 100) if r.closed_cost > 0 else float("nan"),
        "익절%": r.exits_take_profit / max(1, r.trades_closed) * 100,
        "손절%": r.exits_stop / max(1, r.trades_closed) * 100,
        "만료%": r.exits_expire / max(1, r.trades_closed) * 100,
        "급락정리%": r.exits_regime / max(1, r.trades_closed) * 100,
        "폐지정리건수": r.exits_delist,
    }


def yearly(r: Result, first_day: int) -> pd.Series:
    s = pd.Series(r.equity[first_day:], index=r.dates[first_day:])
    ye = s.groupby(s.index.year).last()
    prev = pd.concat([pd.Series([s.iloc[0]], index=[ye.index[0] - 1]), ye]).shift(1).iloc[1:]
    return (ye / prev - 1) * 100


def window_return(r: Result, dates: pd.DatetimeIndex, years: float) -> float:
    """마지막 날로부터 years 년 전 이후의 평가자산 변화율(%). 구간이 부족하면 nan."""
    end = len(dates) - 1
    start = int(dates.searchsorted(dates[end] - pd.Timedelta(days=int(years * 365.25))))
    if start >= end or not np.isfinite(r.equity[start]):
        return float("nan")
    return (r.equity[end] / r.equity[start] - 1) * 100


def custom_report(a, dates, o, h, l, c, v, first, test, ml_picks) -> None:
    """사용자 지정 규칙(익절·손절·N영업일 후 정리)으로 예측 매수·매도를 계속 반복했을 때의 결과.
    배분 방식 두 가지를 모두 낸다: ① 기사식(싼 종목부터 1주씩 → 가격 비중 포트폴리오), ② 종목별 균등 금액(앱 스킬과 비슷)."""
    tp, sl, hold = a.threshold, a.stop, a.max_hold
    cost = Costs()
    label = f"익절 +{tp:.0%} · 손절 -{sl:.0%} · {hold}영업일 후 정리"
    mp = a.max_positions
    adv = pd.DataFrame(c * v).rolling(20, min_periods=15).mean().to_numpy()   # 20일 평균 거래대금(원)

    def run(picks, costs=cost, alloc="equal", **kw):
        base = dict(take_profit=tp, stop=sl, max_hold=hold, skip_held=True, alloc=alloc, max_positions=mp, same_bar=a.same_bar, adv=adv, adv_cap=a.adv_cap)
        base.update(kw)
        return simulate(dates, o, h, l, c, picks, first, a.capital, costs, **base)

    def median_of(make, **kw):
        res = [run(make(s), **kw) for s in range(a.seeds)]
        mid = float(np.median([x.final_net for x in res]))
        return min(res, key=lambda r: abs(r.final_net - mid))

    rnd = lambda s: random_picks(c, v, test, a.q, 400 + s)  # noqa: E731
    rows: Dict[str, Result] = {}
    for alloc, tag, name in (("equal", "균등", f"종목별 균등 금액(최대 {mp}종목, 앱 스킬과 비슷)"), ("shares", "기사식", "싼 종목부터 1주씩(기사 방식)")):
        rows[f"[{tag}] ML 예측 · {label}"] = run(ml_picks, alloc=alloc)
        rows[f"[{tag}]  └ 비용 없음"] = run(ml_picks, Costs.none(), alloc=alloc)
        rows[f"[{tag}]  └ 손절만 뺌(기간 정리는 유지)"] = run(ml_picks, alloc=alloc, stop=None)
        rows[f"[{tag}] 무작위 선정 · 같은 규칙"] = median_of(rnd, alloc=alloc)
        for acc in (0.55, 0.65):
            rows[f"[{tag}] 가상 예측기 적중 {acc:.0%} · 같은 규칙"] = median_of(lambda s, acc=acc: oracle_picks(c, v, test, a.q, acc, 500 + s), alloc=alloc)
    rows["[참고] 기사 원본(손절·기간정리 없음, 1주씩)"] = simulate(dates, o, h, l, c, ml_picks, first, a.capital, cost, take_profit=tp)

    table = pd.DataFrame({k: stats(r, first, a.capital) for k, r in rows.items()}).T
    pd.set_option("display.width", 280)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 70)
    print(f"\n== 시나리오별 결과 (자본 {a.capital:,.0f}원, 시험 {dates[first].date()} ~ {dates[-1].date()}, {(dates[-1] - dates[first]).days / 365.25:.1f}년)")
    cols = ["최종자산(만원)", "총수익률%", "연환산%", "일평균%", "최대낙폭%", "매도건수", "승률%", "거래당순수익%", "금액가중수익%", "평균보유일", "익절%", "손절%", "만료%", "평균투자비중%"]
    print(table[cols].round(2).to_string())

    # 회계 검증: 최종 현금 = 자본 + 확정 손익 - 미청산 원가
    worst = max(abs(r.final_cash - (a.capital + r.closed_pnl - r.open_cost)) for r in rows.values())
    print(f"\n[회계 검증] 모든 시나리오에서 '최종 현금 = 자본 + 확정손익 - 미청산 원가' 최대 오차 {worst:,.4f}원")

    key = f"[균등] ML 예측 · {label}"
    # 같은 날 손절·익절 순서 가정에 따른 범위(균등·ML): 일봉으로는 알 수 없어서 양 끝과 기댓값을 모두 보여 준다
    rng_rows = {}
    for mode, nm in (("stop", "손절 먼저(보수적)"), ("random", "동전 던지기(기댓값)"), ("target", "익절 먼저(낙관적)")):
        s = stats(run(ml_picks, same_bar=mode), first, a.capital)
        rng_rows[nm] = {"연환산%": s["연환산%"], "총수익률%": s["총수익률%"], "최대낙폭%": s["최대낙폭%"], "승률%": s["승률%"], "익절%": s["익절%"], "손절%": s["손절%"]}
    print("\n== 같은 날 손절·익절이 함께 닿을 때의 가정에 따른 범위 (균등 배분 · ML 예측 · 비용 반영)")
    print(pd.DataFrame(rng_rows).T.round(2).to_string())

    best = rows[key]
    n = max(1, best.trades_closed)
    print(f"\n[{key}] 청산 구성: 익절 {best.exits_take_profit / n:.1%} · 손절 {best.exits_stop / n:.1%} · 기간만료 {best.exits_expire / n:.1%}"
          f" | 거래 {best.trades_closed:,}건, 거래당 순수익 {best.avg_trade_net * 100:+.3f}%, 금액가중 {best.closed_pnl / max(best.closed_cost, 1) * 100:+.3f}%")
    print(f"[참고] 익절 +{tp:.0%} / 손절 -{sl:.0%} 에서 비용 없이 본전인 승률은 {sl / (tp + sl) * 100:.0f}%, 왕복 비용 약 {(cost.commission * 2 + cost.sell_tax + cost.slippage * 2) * 100:.2f}%를 감안하면 더 높다")

    # 민감도(균등 배분, ML 신호 고정, 비용 반영): 손절 폭 × 보유 기간
    grid = {}
    for stop_v in (0.01, 0.02, 0.03, 0.05, None):
        row = {}
        for hold_v in (3, 5, 10):
            r = run(ml_picks, stop=stop_v, max_hold=hold_v)
            row[f"{hold_v}영업일"] = stats(r, first, a.capital)["연환산%"]
        grid[f"손절 -{stop_v:.0%}" if stop_v else "손절 없음"] = row
    print(f"\n== 민감도(균등 배분): 익절 +{tp:.0%} 고정, 연환산 수익률(%) — 손절 폭 × 보유 기간")
    print(pd.DataFrame(grid).T.round(1).to_string())

    rets = c[first:] / c[first - 1:-1] - 1
    ew = np.nanmean(np.where(np.isfinite(rets), rets, np.nan), axis=1)
    ew_series = pd.Series(ew, index=dates[first:first + len(ew)]).fillna(0.0)
    ew_total = float(np.prod(1 + ew_series.to_numpy()) - 1) * 100
    kospi = load_index("^KS11", str(dates[first].date()))
    print(f"\n기준선: 같은 종목 동일비중 보유(매일 재조정, 비용 제외) {ew_total:+.1f}%"
          + (f" | 코스피 {kospi.iloc[-1] / kospi.iloc[0] * 100 - 100:+.1f}%" if kospi is not None and len(kospi) > 10 else ""))

    show = [key, "[균등] 무작위 선정 · 같은 규칙", "[기사식] ML 예측 · " + label, "[참고] 기사 원본(손절·기간정리 없음, 1주씩)"]
    short = {key: "균등·ML", "[균등] 무작위 선정 · 같은 규칙": "균등·무작위", "[기사식] ML 예측 · " + label: "기사식·ML", "[참고] 기사 원본(손절·기간정리 없음, 1주씩)": "기사 원본"}
    y = pd.DataFrame({short[k]: yearly(rows[k], first) for k in show}).round(1)
    y["동일비중 보유"] = ew_series.groupby(ew_series.index.year).apply(lambda s: (np.prod(1 + s) - 1) * 100).round(1)
    print("\n== 연도별 수익률(%)")
    print(y.to_string())
    recent = {short[k]: {f"최근 {yr}년": window_return(rows[k], dates, yr) for yr in (1, 3, 5)} for k in show}
    print("\n== 최근 구간 수익률(%)")
    print(pd.DataFrame(recent).T.round(1).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"donga_backtest_custom_{a.dataset}.csv"
    table.round(4).to_csv(out, encoding="utf-8-sig")
    print(f"\n저장: {out}")


def regime_signal(series: np.ndarray, rule: str) -> np.ndarray:
    """T 일 종가까지의 지수만으로 만든 '급락기' 판단(True=현금). 미래 정보를 쓰지 않는다."""
    s = pd.Series(series)
    if rule == "ma20":
        off = s < s.rolling(20, min_periods=20).mean()
    elif rule == "ma60":
        off = s < s.rolling(60, min_periods=60).mean()
    elif rule == "dd7":
        off = s / s.rolling(20, min_periods=20).max() - 1 <= -0.07
    elif rule == "ret5":
        off = s.pct_change(5, fill_method=None) <= -0.05
    else:
        raise ValueError(rule)
    return off.fillna(False).to_numpy(dtype=bool)


def load_status(db: str) -> Dict[str, str]:
    try:
        con = sqlite3.connect(f"file:{DATA / db}?mode=ro", uri=True)
        rows = con.execute("select symbol, status from meta").fetchall()
        con.close()
        return {s: st for s, st in rows}
    except Exception:  # noqa: BLE001
        return {}


def regime_report(a, dates, o, h, l, c, v, first, test, ml, syms) -> None:
    """상장폐지 종목을 포함한 전종목 데이터에서, 급락기를 '그날까지의 지수 흐름'으로 판단해 현금으로 피하는 효과를 본다."""
    cost = Costs()
    adv = pd.DataFrame(c * v).rolling(20, min_periods=15).mean().to_numpy()
    status = load_status("candles_krx_all.sqlite")
    delisted = np.array([status.get(s) == "DELISTED" for s in syms])
    D = len(dates)

    rets = c[1:] / c[:-1] - 1
    ew_daily = np.nanmean(np.where(np.isfinite(rets), rets, np.nan), axis=1)
    ew_series_all = pd.Series(np.concatenate([[0.0], np.nan_to_num(ew_daily)]), index=dates)
    ew_idx = (1 + ew_series_all).cumprod().to_numpy()
    kospi = load_index("^KS11", "2013-06-01")
    indexes = {"전종목 동일비중": ew_idx}
    if kospi is not None and len(kospi) > 100:
        indexes = {"코스피": kospi.reindex(dates).ffill().to_numpy(), **indexes}
    rules = {"ma20": "20일선 아래", "ma60": "60일선 아래", "dd7": "20일 고점 -7%", "ret5": "5일 -5% 급락"}

    print(f"\n상장폐지 종목 {int(delisted.sum()):,}개 포함(전체 {len(syms):,}개). 시험 기간 중 시세가 끝난 종목 수: "
          f"{int(sum(1 for k in range(len(syms)) if delisted[k] and np.isfinite(c[first:, k]).any())):,}")

    offs: Dict[Tuple[str, str], np.ndarray] = {(iname, rk): regime_signal(iv, rk) for iname, iv in indexes.items() for rk in rules}
    print("\n== 급락기 판단 비중(시험 기간 중 현금으로 쉬는 날의 비율 %)")
    print(pd.DataFrame({iname: {rules[rk]: float(offs[(iname, rk)][first:].mean() * 100) for rk in rules} for iname in indexes}).round(1).to_string())

    def run(picks, stop, off=None, rec=1.0, costs=cost, tp=a.threshold):
        return simulate(dates, o, h, l, c, picks, first, a.capital, costs, take_profit=tp, stop=stop, max_hold=a.max_hold, skip_held=True,
                        alloc="equal", max_positions=a.max_positions, same_bar=a.same_bar, adv=adv, adv_cap=a.adv_cap, risk_off=off, delist_recovery=rec)

    def median_run(make, stop, off=None):
        res = [run(make(s), stop, off) for s in range(a.seeds)]
        mid = float(np.median([x.final_net for x in res]))
        return min(res, key=lambda r: abs(r.final_net - mid))

    cols = ["최종자산(만원)", "총수익률%", "연환산%", "최대낙폭%", "승률%", "평균투자비중%", "매도건수", "급락정리%", "폐지정리건수"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)
    all_rows: Dict[str, Result] = {}
    for tag, stop in (("손절 -%d%%" % round(a.stop * 100), a.stop), ("손절 없음", None)):
        rows: Dict[str, Result] = {"필터 없음": run(ml, stop)}
        for iname in indexes:
            for rk, rname in rules.items():
                rows[f"{iname} {rname}"] = run(ml, stop, offs[(iname, rk)])
        rows["[비교] 무작위 선정 · 필터 없음"] = median_run(lambda s: random_picks(c, v, test, a.q, 600 + s), stop)
        best_iname = list(indexes)[0]
        rows[f"[비교] 무작위 선정 · {best_iname} 60일선"] = median_run(lambda s: random_picks(c, v, test, a.q, 600 + s), stop, offs[(best_iname, "ma60")])
        surv = {d: idx[~delisted[idx]] for d, idx in ml.items()}
        rows["[비교] 상장폐지 종목 제외(생존자만) · 필터 없음"] = run(surv, stop)
        rows[f"[비교] 상장폐지 제외 · {best_iname} 60일선"] = run(surv, stop, offs[(best_iname, "ma60")])
        for nm, r in rows.items():
            all_rows[f"{tag} | {nm}"] = r
        table = pd.DataFrame({k: stats(r, first, a.capital) for k, r in rows.items()}).T
        print(f"\n== ML 예측 · 익절 +{a.threshold:.0%} · {tag} · {a.max_hold}영업일 후 정리 · 균등 {a.max_positions}종목 · 비용 반영 · 상장폐지 포함({a.capital:,.0f}원)")
        print(table[cols].round(1).to_string())

    print("\n== 상장폐지 종목을 들고 있을 때 회수율 가정(손절 없음 · 필터 없음)")
    rec_rows = {f"마지막 종가의 {int(rec * 100)}%": stats(run(ml, None, rec=rec), first, a.capital) for rec in (1.0, 0.5, 0.0)}
    print(pd.DataFrame(rec_rows).T[["총수익률%", "연환산%", "최대낙폭%", "폐지정리건수"]].round(1).to_string())

    pick = [f"{tag} | {nm}" for tag in ("손절 -%d%%" % round(a.stop * 100), "손절 없음") for nm in ("필터 없음", f"{list(indexes)[0]} 60일선 아래", "전종목 동일비중 60일선 아래")]
    ew_year = ew_series_all.iloc[first:].groupby(ew_series_all.index[first:].year).apply(lambda s: (np.prod(1 + s) - 1) * 100)
    y = pd.DataFrame({k.replace(" | ", "·"): yearly(all_rows[k], first) for k in pick if k in all_rows}).round(1)
    y["동일비중 보유"] = ew_year.round(1)
    if kospi is not None and len(kospi) > 100:
        ks = kospi[kospi.index >= dates[first] - pd.Timedelta(days=7)]
        ky = ks.groupby(ks.index.year).last()
        y["코스피"] = ((ky / ky.shift(1) - 1) * 100).reindex(y.index).round(1)
    print("\n== 연도별 수익률(%)")
    print(y.to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "donga_backtest_regime.csv"
    pd.DataFrame({k: stats(r, first, a.capital) for k, r in all_rows.items()}).T.round(4).to_csv(out, encoding="utf-8-sig")
    print(f"\n저장: {out}")


def member_mask(dates: pd.DatetimeIndex, syms: List[str], db: str) -> np.ndarray:
    """(날짜, 종목) → 그날 S&P 500 구성종목이면 True. 편입일~편출일(포함) 기준이라 미래에 편입될 종목을 미리 쓰지 않는다."""
    con = sqlite3.connect(f"file:{DATA / db}?mode=ro", uri=True)
    rows = con.execute("select ticker, start_date, end_date from member").fetchall()
    con.close()
    pos = {s: k for k, s in enumerate(syms)}
    mask = np.zeros((len(dates), len(syms)), dtype=bool)
    for t, s, e in rows:
        k = pos.get(t)
        if k is None:
            continue
        lo = dates.searchsorted(pd.Timestamp(s))
        hi = dates.searchsorted(pd.Timestamp(e), side="right") if e else len(dates)
        mask[lo:hi, k] = True
    return mask


def us_report(a, dates, syms, o, h, l, c, v) -> None:
    """미국: 그날의 S&P 500 구성종목(또는 그중 거래대금 상위)에서만 ML 로 고르고, 같은 매매 규칙·급락기 필터를 적용한다."""
    db = "candles_us_sp500.sqlite"
    capital = a.capital if a.capital != 30_000_000 else 50_000.0
    cost = Costs(commission=0.0001, sell_tax=0.00003, slippage=0.0005)      # 수수료 1bp, 매도 규정 수수료 0.3bp, 슬리피지 5bp
    first = max(int(dates.searchsorted(pd.Timestamp(a.start or "2016-01-01"))), 80)
    test = range(first, len(dates) - 1)
    train_days = min(a.train_days, first - 60)
    members = member_mask(dates, syms, db)
    has = np.isfinite(c) & (v > 0)
    print(f"데이터: us | 종목 {len(syms):,} | {dates[0].date()} ~ {dates[-1].date()} ({len(dates):,}거래일) | 시험 {dates[first].date()} ~ (자본 ${capital:,.0f}, 비용: 수수료 1bp·매도 0.3bp·슬리피지 5bp)", flush=True)

    # 구성종목 중 시세가 실제로 있는 비율(파산·피인수 종목은 Yahoo 에 없어 빠진다)
    cov = {}
    for y in range(dates[first].year, dates[-1].year + 1):
        m = np.asarray(dates.year == y)
        cov[y] = float((members[m] & has[m]).sum() / max(1, members[m].sum())) * 100
    print("\n== 그날의 구성종목 중 시세가 확보된 비율(%) — 낮을수록 사라진 종목이 빠져 있어 낙관적")
    print(pd.Series(cov).round(1).to_string())

    adv = pd.DataFrame(c * v).rolling(20, min_periods=15).mean().to_numpy()
    rank = pd.DataFrame(np.where(members & has, adv, np.nan)).rank(axis=1, ascending=False).to_numpy()
    universes = {"S&P500 그날 구성종목": (members & has, a.q), "구성종목 중 거래대금 상위 100": (members & has & (rank <= 100), 0.30)}

    feat = build_features(o, h, l, c, v)
    picks: Dict[str, Dict[int, np.ndarray]] = {}
    for name, (m, q) in universes.items():
        ml, diag = walk_forward_picks(feat, c, v, test, q, train_days, a.retrain_every, a.threshold, lambda s: None, mask=m)
        picks[name] = ml
        print(f"  [{name}] 신호 {diag['pickDays']}일, 하루 평균 {diag['avgPicksPerDay']:.0f}종목 | 다음날 상승 적중 {diag['pickUpRate']:.1%} | "
              f"다음날 +{a.threshold:.0%} 이상 {diag['pickHitThrRate']:.1%} (전체 평균 {diag['baseHitThrRate']:.1%})", flush=True)
    del feat

    gspc = load_index("^GSPC", "2011-06-01", db)
    idx = gspc.reindex(dates).ffill().to_numpy()
    offs = {"S&P500 60일선 아래": regime_signal(idx, "ma60"), "S&P500 20일선 아래": regime_signal(idx, "ma20")}
    print("\n== 급락기 판단 비중(%): " + ", ".join(f"{k} {ov[first:].mean() * 100:.0f}" for k, ov in offs.items()))

    def run(pk, stop, off=None, costs=cost, **kw):
        return simulate(dates, o, h, l, c, pk, first, capital, costs, take_profit=a.threshold, stop=stop, max_hold=a.max_hold, skip_held=True,
                        alloc="equal", max_positions=a.max_positions, same_bar=kw.pop("same_bar", a.same_bar), adv=adv, adv_cap=a.adv_cap,
                        risk_off=off, delist_recovery=1.0, usd=True, **kw)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["총수익률%", "연환산%", "최대낙폭%", "승률%", "거래당순수익%", "평균투자비중%", "매도건수", "급락정리%"]
    allrows: Dict[str, Result] = {}
    stop_tag = "손절 -%d%%" % round(a.stop * 100)
    for uname, pk in picks.items():
        for tag, stop in ((stop_tag, a.stop), ("손절 없음", None)):
            rows = {"필터 없음": run(pk, stop)}
            for on, ov in offs.items():
                rows[on] = run(pk, stop, ov)
            rows["비용 없음"] = run(pk, stop, costs=Costs.none())
            m, q = universes[uname]
            rnd = [run(random_picks(c, v, test, q, 700 + s, mask=m), stop) for s in range(a.seeds)]
            mid = float(np.median([x.final_net for x in rnd]))
            rows["[비교] 무작위 선정"] = min(rnd, key=lambda r: abs(r.final_net - mid))
            t = pd.DataFrame({k: stats(r, first, capital) for k, r in rows.items()}).T
            t.insert(0, "최종자산($)", [rows[k].final_net for k in t.index])
            print(f"\n== [{uname}] ML 예측 · 익절 +{a.threshold:.0%} · {tag} · {a.max_hold}영업일 후 정리 · 균등 {a.max_positions}종목 · 자본 ${capital:,.0f}")
            print(t[["최종자산($)"] + cols].round(1).to_string())
            for k, r in rows.items():
                allrows[f"{uname} | {tag} | {k}"] = r

    main_pk = picks["S&P500 그날 구성종목"]
    print("\n== 같은 날 손절·익절이 함께 닿을 때의 가정에 따른 범위 (구성종목 · 필터 없음)")
    rng_rows = {}
    for tag, stop in ((stop_tag, a.stop), ("손절 없음", None)):
        for mode, nm in (("stop", "손절 먼저"), ("random", "동전 던지기"), ("target", "익절 먼저")):
            s = stats(run(main_pk, stop, same_bar=mode), first, capital)
            rng_rows[f"{tag} · {nm}"] = {k: s[k] for k in ("연환산%", "총수익률%", "최대낙폭%", "승률%")}
    print(pd.DataFrame(rng_rows).T.round(1).to_string())

    rets = np.where(members[1:] & has[1:] & has[:-1], c[1:] / c[:-1] - 1, np.nan)
    ew = pd.Series(np.concatenate([[0.0], np.nan_to_num(np.nanmean(rets, axis=1))]), index=dates)
    yrs = (dates[-1] - dates[first]).days / 365.25
    ew_total = float(np.prod(1 + ew.iloc[first:].to_numpy()) - 1)
    sp_total = float(idx[-1] / idx[first] - 1)
    print(f"\n기준선: S&P 500 지수 {sp_total * 100:+.0f}% (연 {((1 + sp_total) ** (1 / yrs) - 1) * 100:.1f}%) | "
          f"그날 구성종목 동일비중 보유(매일 재조정, 비용 제외) {ew_total * 100:+.0f}% (연 {((1 + ew_total) ** (1 / yrs) - 1) * 100:.1f}%)")

    u = "S&P500 그날 구성종목"
    pick_rows = [f"{u} | {stop_tag} | 필터 없음", f"{u} | 손절 없음 | 필터 없음", f"{u} | 손절 없음 | S&P500 60일선 아래"]
    y = pd.DataFrame({k.split(" | ", 1)[1].replace(" | ", "·"): yearly(allrows[k], first) for k in pick_rows}).round(1)
    y["구성종목 동일비중 보유"] = ew.iloc[first:].groupby(ew.index[first:].year).apply(lambda s: (np.prod(1 + s) - 1) * 100).round(1)
    gy = gspc[gspc.index >= dates[first] - pd.Timedelta(days=7)]
    gy = gy.groupby(gy.index.year).last()
    y["S&P500"] = ((gy / gy.shift(1) - 1) * 100).reindex(y.index).round(1)
    print("\n== 연도별 수익률(%)")
    print(y.to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "donga_backtest_us.csv"
    pd.DataFrame({k: stats(r, first, capital) for k, r in allrows.items()}).T.round(4).to_csv(out, encoding="utf-8-sig")
    print(f"\n저장: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["long", "broad", "all", "us"])
    ap.add_argument("--regime", action="store_true", help="급락기 현금 보유 필터를 비교한다(dataset=all 권장: 상장폐지 종목 포함)")
    ap.add_argument("--reuse-picks", action="store_true", help="저장된 ML 신호(pickle)가 있으면 다시 학습하지 않는다")
    ap.add_argument("--start", default=None, help="시험 시작일(기본: long=2016-01-01, broad=전체의 뒤 40%)")
    ap.add_argument("--q", type=float, default=0.15, help="하루에 고르는 종목 비율(기사: 3000개 중 400~500개 ≈ 15%)")
    ap.add_argument("--capital", type=float, default=30_000_000)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--train-days", type=int, default=250)
    ap.add_argument("--retrain-every", type=int, default=21)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--same-bar", choices=["stop", "target", "random"], default="random", help="같은 날 손절·익절이 모두 닿을 때 순서 가정(기본 random=동전 던지기)")
    ap.add_argument("--adv-cap", type=float, default=0.01, help="균등 배분에서 한 종목 매수 금액 상한: 직전 20일 평균 거래대금의 비율(0 이면 제한 없음)")
    ap.add_argument("--max-positions", type=int, default=20, help="--custom 균등 배분 시 동시 보유 종목 수")
    ap.add_argument("--custom", action="store_true", help="익절·손절·N영업일 후 정리 규칙으로 예측 매수·매도를 반복한 결과를 낸다")
    ap.add_argument("--stop", type=float, default=0.02, help="--custom 손절 폭(기본 2%)")
    ap.add_argument("--max-hold", type=int, default=5, help="--custom 매수일로부터 정리까지의 영업일(기본 5)")
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731

    if a.dataset == "long":
        dates, syms, p = load_panel("candles_max.sqlite", "KRX", "2012-01-01", 600)
        start = a.start or "2016-01-01"
    elif a.dataset == "us":
        dates, syms, p = load_panel("candles_us_sp500.sqlite", "US", "2012-01-01", 120)
        us_report(a, dates, syms, p["open"], p["high"], p["low"], p["close"], p["volume"])
        return
    elif a.dataset == "all":
        dates, syms, p = load_panel("candles_krx_all.sqlite", "KRX", "2014-01-01", 120)
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

    cache = OUT / f"donga_picks_{a.dataset}_{a.q}_{train_days}.pkl"
    if a.reuse_picks and cache.exists():
        import pickle
        ml, diag = pickle.loads(cache.read_bytes())
        log(f"  저장된 ML 신호 사용: {cache}")
    else:
        log("  ML 모델 학습·신호 생성…")
        ml, diag = walk_forward_picks(feat, c, v, test, a.q, train_days, a.retrain_every, a.threshold, lambda s: None)
        import pickle
        OUT.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps((ml, diag)))
    del feat
    log(f"  신호 {diag['pickDays']}일, 하루 평균 {diag['avgPicksPerDay']:.0f}종목 | 고른 종목의 다음날 상승률(적중) {diag['pickUpRate']:.1%} | "
        f"다음날 +{a.threshold:.0%} 이상 {diag['pickHitThrRate']:.1%} (전체 평균 {diag['baseHitThrRate']:.1%})")

    if a.regime:
        regime_report(a, dates, o, h, l, c, v, first, test, ml, syms)
        return
    if a.custom:
        custom_report(a, dates, o, h, l, c, v, first, test, ml)
        return

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
