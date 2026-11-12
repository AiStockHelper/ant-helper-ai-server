"""긴 데이터(약 20년)로 여러 전략을 같은 방식으로 돌려 비교하는 탐색 도구 (연구용, 서비스에는 포함되지 않는다).

데이터: python -m tools.fetch_max 로 받은 data/candles_max.sqlite (수정주가: 분할·배당 반영, 지수는 가격지수).
체결: 신호는 종가에 알고 다음 날 종가에 체결(하루 지연). 거래비용: 비중 변화량 x 편도 비용.
평가 구간: 2007-12 ~ 현재. '과거 구간(IS)' = ~2016-12, '검증 구간(OOS)' = 2017-01~. 전략은 IS 샤프로 고르고 OOS 에서 확인한다.
  (OOS 도 이미 알려진 과거라서 완전히 깨끗한 검증은 아니다. 변형을 많이 돌릴수록 '운 좋게 잘 나온 것'이 섞이므로 변형 전체의 중앙값을 함께 본다.)
생존자 편향: 지수·ETF 는 작고, 개별 종목(US/KRX)은 '지금 거래대금이 큰 종목'만 있어서 크다 -> 종목 전략은 후보 전체 동일 비중과 지수를 함께 보여 준다.
사용: python -m tools.strategy_lab [us|kr|stocks|all]
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
KR_CASH_YEARLY = 0.02

US_RISK = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC"]


# ───────────────────────────── 데이터 / 엔진 ─────────────────────────────

def load(markets: tuple[str, ...]):
    con = sqlite3.connect(DB)
    q = f"select market,symbol,date,close,volume from candle where market in ({','.join('?' * len(markets))}) order by date"
    df = pd.read_sql_query(q, con, params=markets)
    df["date"] = pd.to_datetime(df["date"])
    df["key"] = df["symbol"]
    close = df.pivot_table(index="date", columns="key", values="close")
    vol = df.pivot_table(index="date", columns="key", values="volume")
    return close, vol.reindex_like(close)


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).last().values)


def simulate(ret: pd.DataFrame, w_target: pd.DataFrame, side_cost: float) -> pd.Series:
    """w_target(결정일 x 자산) -> 일별 순수익. 결정일 종가에 알고 다음 날 종가에 체결(수익은 그 다음 날부터)."""
    cols = [c for c in w_target.columns if c in ret.columns]
    w = w_target[cols].reindex(ret.index).ffill().shift(2).fillna(0.0)
    gross = (w * ret[cols].fillna(0.0)).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turn * side_cost


def metrics(r: pd.Series, rf: pd.Series | float) -> dict:
    r = r[r.index >= EVAL_START].dropna()
    if len(r) < 252:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / 252
    rfx = rf.reindex(r.index).fillna(0.0) if isinstance(rf, pd.Series) else rf
    ex = r - rfx
    return {
        "cagr": eq.iloc[-1] ** (1 / years) - 1,
        "vol": r.std() * np.sqrt(252),
        "sharpe": ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan,
        "mdd": (eq / eq.cummax() - 1).min(),
    }


def evaluate(r: pd.Series, rf) -> dict:
    full = metrics(r, rf)
    ins = metrics(r[r.index <= IS_END], rf)
    oos = metrics(r[r.index > IS_END], rf)
    if not full or not ins or not oos:
        return {}
    return {"cagr": full["cagr"], "vol": full["vol"], "sharpe": full["sharpe"], "mdd": full["mdd"],
            "is_sharpe": ins["sharpe"], "is_cagr": ins["cagr"], "oos_sharpe": oos["sharpe"], "oos_cagr": oos["cagr"], "oos_mdd": oos["mdd"]}


class Family:
    def __init__(self, title: str, bench: dict[str, dict] | None = None):
        self.title, self.rows, self.bench = title, {}, bench or {}

    def add(self, name: str, ev: dict) -> None:
        if ev:
            self.rows[name] = ev

    def show(self, top: int = 8) -> None:
        print(f"\n######## {self.title} ########")
        hdr = f"{'전략':44s} {'CAGR':>6s} {'변동성':>6s} {'샤프':>5s} {'MDD':>7s} | {'IS샤프':>6s} {'IS CAGR':>7s} | {'OOS샤프':>6s} {'OOS CAGR':>8s} {'OOS MDD':>8s}"
        print(hdr)

        def line(n, e, mark=""):
            print(f"{(mark + n):44s} {e['cagr']*100:+5.1f}% {e['vol']*100:5.1f}% {e['sharpe']:5.2f} {e['mdd']*100:6.1f}% | {e['is_sharpe']:6.2f} {e['is_cagr']*100:+6.1f}% | {e['oos_sharpe']:6.2f} {e['oos_cagr']*100:+7.1f}% {e['oos_mdd']*100:7.1f}%")

        for n, e in self.bench.items():
            line(n, e, "[기준] ")
        df = pd.DataFrame(self.rows).T
        if df.empty:
            return
        best = df.sort_values("is_sharpe", ascending=False)
        for n in best.index[:top]:
            line(n, self.rows[n])
        sel = best.index[0]
        print(f"  -> 변형 {len(df)}개. 과거(IS) 샤프 1등 '{sel}' 의 검증(OOS) 샤프 {df.loc[sel, 'oos_sharpe']:.2f} / OOS CAGR {df.loc[sel, 'oos_cagr']*100:+.1f}%")
        print(f"     변형 전체 중앙값: IS 샤프 {df['is_sharpe'].median():.2f}, OOS 샤프 {df['oos_sharpe'].median():.2f}, OOS CAGR {df['oos_cagr'].median()*100:+.1f}%, "
              f"OOS 샤프가 기준(첫 기준)보다 높은 변형 {(df['oos_sharpe'] > next(iter(self.bench.values()))['oos_sharpe']).mean()*100:.0f}%" if self.bench else "")


# ───────────────────────────── 전략들 ─────────────────────────────

def bh(ret: pd.DataFrame, col: str, cost: float, rf) -> dict:
    w = pd.DataFrame({col: 1.0}, index=ret.index)
    return evaluate(simulate(ret, w, cost), rf)


def trend_filter(close, ret, asset, window, cash, cost, rf, daily=True):
    sig = (close[asset] > close[asset].rolling(window, min_periods=int(window * 0.95)).mean()).astype(float)
    if not daily:
        sig = sig.reindex(month_ends(close.index))
    w = pd.DataFrame({asset: sig, cash: 1 - sig})
    return evaluate(simulate(ret, w, cost), rf)


def ts_momentum(close, ret, asset, months, cash, cost, rf):
    me = month_ends(close.index)
    c = close[asset].reindex(me)
    sig = (c / c.shift(months) - 1 > 0).astype(float)
    return evaluate(simulate(ret, pd.DataFrame({asset: sig, cash: 1 - sig}), cost), rf)


def vol_target(close, ret, asset, target, cash, cost, rf):
    rv = ret[asset].rolling(20).std() * np.sqrt(252)
    lev = (target / rv).clip(upper=1.0).fillna(0.0)
    me = month_ends(close.index)
    lev = lev.reindex(me)
    return evaluate(simulate(ret, pd.DataFrame({asset: lev, cash: 1 - lev}), cost), rf)


def rotation(close, ret, universe, k, lookbacks, cash, cost, rf, abs_filter=True):
    me = month_ends(close.index)
    c = close[universe].reindex(me)
    score = sum(c / c.shift(m) - 1 for m in lookbacks) / len(lookbacks)
    rows = {}
    for t, s in score.iterrows():
        s = s.dropna()
        w = {cash: 0.0}
        if len(s) >= k:
            top = s.nlargest(k)
            for a, v in top.items():
                if abs_filter and v <= 0:
                    w[cash] += 1.0 / k
                else:
                    w[a] = w.get(a, 0.0) + 1.0 / k
        rows[t] = w
    w = pd.DataFrame(rows).T.fillna(0.0)
    return evaluate(simulate(ret, w, cost), rf)


def gem(close, ret, cost, rf, cash="BIL"):
    me = month_ends(close.index)
    c = close[["SPY", "EFA", "AGG", cash]].reindex(me)
    m = c / c.shift(12) - 1
    rows = {}
    for t, r in m.iterrows():
        if r.isna().any():
            continue
        if r["SPY"] > r[cash]:
            rows[t] = {"SPY": 1.0} if r["SPY"] >= r["EFA"] else {"EFA": 1.0}
        else:
            rows[t] = {"AGG": 1.0}
    return evaluate(simulate(ret, pd.DataFrame(rows).T.fillna(0.0), cost), rf)


def leveraged_trend(close, ret, asset, window, lev, cost, rf, loan_spread=0.01, vol_cap=None):
    """추세(이동평균 위)일 때만 lev 배로 보유. lev>1 이면 모자란 돈은 BIL+연 1%에 빌린 것으로 계산(레버리지 ETF 의 비용·괴리는 반영 안 함)."""
    sig = (close[asset] > close[asset].rolling(window, min_periods=int(window * 0.95)).mean()).astype(float)
    wa = sig * lev
    if vol_cap:
        rv = ret[asset].rolling(20).std() * np.sqrt(252)
        wa = wa * (vol_cap / rv).clip(upper=lev).div(lev).fillna(0.0)
    cash_w = 1 - wa
    w = pd.DataFrame({asset: wa, "BIL": cash_w.clip(lower=0), "LOAN": cash_w.clip(upper=0)})
    r2 = ret.copy()
    r2["LOAN"] = ret["BIL"] + loan_spread / 252
    return evaluate(simulate(r2, w, cost), rf)


def sixty_forty(close, ret, cost, rf, a="SPY", b="TLT", wa=0.6):
    me = month_ends(close.index)
    w = pd.DataFrame({a: wa, b: 1 - wa}, index=me)
    return evaluate(simulate(ret, w, cost), rf)


def equal_weight(close, ret, assets, cost, rf):
    me = month_ends(close.index)
    w = pd.DataFrame({a: 1.0 / len(assets) for a in assets}, index=me)
    return evaluate(simulate(ret, w, cost), rf)


# ───────────────────────────── 미국 / 한국 / 종목 ─────────────────────────────

def run_us() -> None:
    close, vol = load(("ETF", "IDX"))
    close = close[[c for c in close.columns if c in set(US_RISK + ["AGG", "BIL", "SHY", "^GSPC", "^IXIC", "^RUT", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"])]]
    ret = close.pct_change().fillna(0.0)
    rf = ret["BIL"]
    cost = 0.0005
    bench = {"SPY 사서 보유": bh(ret, "SPY", cost, rf), "QQQ 사서 보유": bh(ret, "QQQ", cost, rf),
             "60/40 (SPY/TLT, 월 조정)": sixty_forty(close, ret, cost, rf), "ETF 12종 동일 비중(월 조정)": equal_weight(close, ret, US_RISK, cost, rf)}
    f = Family("미국: 지수/ETF 추세 필터 (이동평균 위면 보유, 아니면 단기채 BIL) - 매일 신호", bench)
    for a in ("SPY", "QQQ", "EFA", "EEM", "IWM"):
        for w in (50, 100, 150, 200, 250):
            f.add(f"{a} SMA{w}", trend_filter(close, ret, a, w, "BIL", cost, rf))
    f.show()
    f = Family("미국: 시계열 모멘텀 (n개월 수익이 +면 보유, 월말 판단)", {"SPY 사서 보유": bench["SPY 사서 보유"]})
    for a in ("SPY", "QQQ", "EFA", "EEM", "IWM"):
        for m in (3, 6, 9, 12):
            f.add(f"{a} {m}개월", ts_momentum(close, ret, a, m, "BIL", cost, rf))
    f.show()
    f = Family("미국: 변동성 타깃 (최근 20일 변동성이 목표보다 크면 비중 축소, 월 조정)", {"SPY 사서 보유": bench["SPY 사서 보유"]})
    for a in ("SPY", "QQQ"):
        for t in (0.08, 0.10, 0.12, 0.15):
            f.add(f"{a} 목표변동성 {int(t*100)}%", vol_target(close, ret, a, t, "BIL", cost, rf))
    f.show()
    f = Family("미국: ETF 로테이션 (12종 중 모멘텀 상위 K개 동일 비중, 모멘텀 -이면 BIL, 월 조정)", {"SPY 사서 보유": bench["SPY 사서 보유"], "ETF 12종 동일 비중": bench["ETF 12종 동일 비중(월 조정)"]})
    for k in (1, 2, 3, 5):
        for name, lb in (("3개월", (3,)), ("6개월", (6,)), ("12개월", (12,)), ("1·3·6·12개월 평균", (1, 3, 6, 12))):
            f.add(f"상위 {k}개 / {name}", rotation(close, ret, US_RISK, k, lb, "BIL", cost, rf))
    f.add("듀얼 모멘텀(GEM: SPY/EFA/AGG)", gem(close, ret, cost, rf))
    f.show()
    f = Family("미국: 추세 필터 + 레버리지 (이동평균 위일 때만 L배, 아니면 BIL; 빌린 돈은 BIL+1%)", {"SPY 사서 보유": bench["SPY 사서 보유"], "QQQ 사서 보유": bench["QQQ 사서 보유"]})
    for a in ("SPY", "QQQ"):
        for w in (150, 200):
            for lev in (1.0, 1.5, 2.0):
                f.add(f"{a} SMA{w} x{lev:g}", leveraged_trend(close, ret, a, w, lev, cost, rf))
    for a in ("SPY", "QQQ"):
        f.add(f"{a} SMA150 x2 + 변동성 상한 15%", leveraged_trend(close, ret, a, 150, 2.0, cost, rf, vol_cap=0.15))
    f.show(top=12)


def run_kr() -> None:
    close, vol = load(("IDX", "KETF"))
    cols = ["^KS11", "^KQ11", "069500.KS", "229200.KS", "114800.KS", "132030.KS", "148070.KS"]
    close = close[[c for c in cols if c in close.columns]]
    close = close.dropna(subset=["^KS11"])
    close["CASH"] = (1 + KR_CASH_YEARLY / 252) ** np.arange(len(close))
    ret = close.pct_change().fillna(0.0)
    rf = ret["CASH"]
    cost = 0.001
    bench = {"코스피 지수 보유(배당 제외)": bh(ret, "^KS11", cost, rf), "코스닥 지수 보유(배당 제외)": bh(ret, "^KQ11", cost, rf)}
    f = Family("한국: 지수 추세 필터 (이동평균 위면 보유, 아니면 현금 연 2%) - 매일 신호", bench)
    for a in ("^KS11", "^KQ11"):
        for w in (50, 100, 150, 200, 250):
            f.add(f"{a} SMA{w}", trend_filter(close, ret, a, w, "CASH", cost, rf))
    f.show()
    f = Family("한국: 시계열 모멘텀 (n개월 수익이 +면 보유, 월말 판단)", {"코스피 보유": bench["코스피 지수 보유(배당 제외)"]})
    for a in ("^KS11", "^KQ11"):
        for m in (3, 6, 9, 12):
            f.add(f"{a} {m}개월", ts_momentum(close, ret, a, m, "CASH", cost, rf))
    f.show()


def stock_family(market: str, index_sym: str, cost: float, min_px: float, tv_floor: float, label: str) -> None:
    close, vol = load((market, "IDX"))
    idx = close[index_sym].dropna()
    close = close[[c for c in close.columns if c != index_sym and not str(c).startswith("^")]]
    # 분할 미반영 등 데이터 오류 제거: 한국은 하루 등락 제한(±30%)을 넘는 값이 있는 종목, 미국은 하루 +/-100% 넘는 종목을 뺀다.
    limit = 0.31 if market == "KRX" else 1.0
    bad = close.pct_change().abs().max() > limit
    print(f"[{label}] 데이터 오류로 제외한 종목: {int(bad.sum())}/{close.shape[1]}")
    close = close.loc[:, ~bad.values]
    close = close.loc[idx.index.min():]
    vol = vol.reindex_like(close)
    idx = idx.reindex(close.index).ffill()
    cash = pd.Series((1 + KR_CASH_YEARLY / 252) ** np.arange(len(close)), index=close.index)
    ret = close.pct_change().fillna(0.0)
    ret["CASH"] = cash.pct_change().fillna(0.0)
    rf = ret["CASH"]
    daily = close.pct_change()
    tv = (close * vol).rolling(60, min_periods=40).mean()
    feat = {
        "mom12_1": close.shift(21) / close.shift(252) - 1,
        "mom6_1": close.shift(21) / close.shift(126) - 1,
        "low_vol": -daily.rolling(120, min_periods=100).std(),
        "quality": (close.shift(21) / close.shift(147) - 1) / daily.rolling(120, min_periods=100).std() / np.sqrt(120),
        "rev_1m": -(close / close.shift(21) - 1),
    }
    risk_on = (idx > idx.rolling(200, min_periods=180).mean())
    me = [d for d in month_ends(close.index) if d >= EVAL_START - pd.Timedelta(days=5)]

    def build(factor: str, n: int, filt: bool, universe: int = 200):
        rows = {}
        for t in me:
            ok = (close.loc[t] >= min_px) & (tv.loc[t] >= tv_floor)
            u = tv.loc[t][ok].nlargest(universe).index
            if len(u) < max(60, n * 2):
                continue
            if filt and not bool(risk_on.loc[t]):
                rows[t] = {"CASH": 1.0}
                continue
            if factor == "ALL":
                pick = u
            else:
                s = feat[factor].loc[t, u].dropna()
                pick = s.nlargest(n).index
            rows[t] = {a: 1.0 / len(pick) for a in pick}
        return pd.DataFrame(rows).T.fillna(0.0)

    idx_ret = pd.DataFrame({"IDX": idx.pct_change().fillna(0.0), "CASH": ret["CASH"]})
    bench = {f"{label} 지수 보유(배당 제외)": evaluate(simulate(idx_ret, pd.DataFrame({"IDX": 1.0}, index=idx_ret.index), cost), rf),
             f"{label} 후보 200종목 동일 비중(월 조정, 생존자 편향)": evaluate(simulate(ret, build("ALL", 0, False), cost), rf)}
    f = Family(f"{label} 개별 종목(생존자 편향 있음): 월 조정, 거래대금 상위 200종목 중 상위 N개 동일 비중", bench)
    for factor in ("mom12_1", "mom6_1", "low_vol", "quality", "rev_1m"):
        for n in (20, 50):
            for filt in (False, True):
                f.add(f"{factor} 상위 {n}{' + 지수 200일선 필터' if filt else ''}", evaluate(simulate(ret, build(factor, n, filt), cost), rf))
    f.show(top=10)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("us", "all"):
        run_us()
    if which in ("kr", "all"):
        run_kr()
    if which in ("stocks", "all"):
        stock_family("US", "^GSPC", 0.001, 5.0, 2e7, "미국")
        stock_family("KRX", "^KS11", 0.002, 1000.0, 1e9, "한국")


if __name__ == "__main__":
    main()
