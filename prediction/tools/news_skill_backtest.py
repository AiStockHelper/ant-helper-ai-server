"""뉴스 AI 자동매매 스킬(news-ai-trader)을 과거 뉴스·시세로 돌려 보는 백테스트.

스킬의 하루 흐름을 그대로 옮긴다.
  장 시작 직후: ① (보유 중인데) AI 가 피하라고 한 종목은 시가에 판다 ② AI 가 제안한 종목 중 확신도 ≥ 기준인 것을 코드 검증(NewsPickValidator)을 거쳐
  빈 자리만큼 시가에 산다 ③ 장중 보유 종목은 익절·손절, 보유일 초과는 종가 직전 정리 ④ 누적 손실 한도에 닿으면 전량 정리하고 멈춘다.
검증(스킬의 NewsPickValidator): 시세가 있는 종목, 종목명 일치(한국만; 미국은 이름 자료가 없다), 거래대금 상위 종목(한국 상위 1000, 미국은 보유 데이터 안),
  오늘 시가가 전일 종가보다 chase 이상 오른 종목은 쫓지 않음. 산업 ETF 참고 목록은 명칭·거래대금 검사 예외.

미래 정보 방지: 판단일 D 의 뉴스는 D 이전 날짜의 기사만 쓴다. AI 의 지식 시점(2025년 중반) 이후 구간만 쓴다.

사용
  python -m tools.news_skill_backtest build-days --market kr|us [--start 2026-01-02 --end 2026-11-12]
  python -m tools.news_skill_backtest run --market kr|us --conds news nonews [--budget N] [--no-loss-limit]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PREDICTION_DATA_DIR", ROOT / "data"))
STUDY = DATA / "news_study"

US_REF_ETFS = ["XLK", "XLE", "XLF", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]
KR_REF_ETFS = ["091160", "266370", "305720", "244580", "266420", "091180", "091170", "102970", "140700", "117700", "117680", "117460",
               "140710", "266390", "266410", "300950"]
BENCH = {"kr": "069500", "us": "SPY"}


# ─────────────────────────── 시세 묶음 ───────────────────────────

@dataclass
class Panel:
    dates: pd.DatetimeIndex
    syms: List[str]
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    names: Dict[str, str] = field(default_factory=dict)      # 코드 → 종목명(없으면 비어 있다)
    liquid: Optional[np.ndarray] = None                      # (날짜, 종목) → 그날 장 시작 전 거래대금 상위 N 이면 True
    ref_etfs: Set[str] = field(default_factory=set)
    krx: bool = True

    def idx(self) -> Dict[str, int]:
        return {s: i for i, s in enumerate(self.syms)}


def norm_name(s: Optional[str]) -> str:
    return re.sub(r"[^\w]", "", (s or "").lower())


def name_matches(given: str, actual: Optional[str]) -> bool:
    g, a = norm_name(given), norm_name(actual)
    if len(g) < 2 or len(a) < 2:
        return False
    if g == a:
        return True
    short, long_ = (g, a) if len(g) <= len(a) else (a, g)
    return len(short) >= 3 and short in long_


def liquidity_matrix(c: np.ndarray, v: np.ndarray, top_n: int, window: int = 20) -> np.ndarray:
    """liquid[d, s] = d 일 장 시작 전(전일까지)의 최근 window 일 평균 거래대금이 top_n 위 안."""
    val = pd.DataFrame(c * v).rolling(window, min_periods=int(window * 0.75)).mean().shift(1)
    rank = val.rank(axis=1, ascending=False)
    return (rank <= top_n).to_numpy()


# ─────────────────────────── 시뮬레이션 ───────────────────────────

@dataclass
class Params:
    budget: float = 5_000_000.0
    max_positions: int = 3
    min_conviction: int = 4
    take_profit: float = 0.08
    stop_loss: float = 0.05
    max_hold_days: int = 10
    chase: float = 0.08                    # 시가가 전일 종가보다 이만큼 오른 종목은 쫓지 않는다(0 이면 끔)
    loss_limit: Optional[float] = 0.10     # 누적 손실이 시드의 이 비율에 닿으면 전량 정리하고 멈춘다(None 이면 끔)
    commission: float = 0.00015
    sell_tax: float = 0.0020               # 주식 매도세. 산업 ETF 는 sell_tax_etf 를 쓴다
    sell_tax_etf: float = 0.0
    slippage: float = 0.0005
    same_bar: str = "stop"                 # 같은 날 손절·익절이 다 닿으면 stop=손절 먼저(보수적), target=익절 먼저
    ai_exits: bool = True                  # AI 가 피하라고 한 보유 종목을 판다


@dataclass
class Trade:
    sym: str
    open_idx: int
    close_idx: int
    qty: int
    entry: float
    exit: float
    net_pct: float
    pnl: float
    reason: str


@dataclass
class Sim:
    dates: pd.DatetimeIndex
    equity: np.ndarray
    trades: List[Trade]
    proposals: int = 0
    accepted: int = 0
    rejects: Dict[str, int] = field(default_factory=dict)
    killed_at: Optional[int] = None
    invested_share: float = 0.0


def simulate(p: Panel, decisions: Dict[pd.Timestamp, dict], prm: Params, first: int, last: int) -> Sim:
    idx = p.idx()
    cash = prm.budget
    pos: Dict[int, dict] = {}                         # 종목 인덱스 → {qty, cost(수수료 포함 총원가), entry, opened}
    trades: List[Trade] = []
    equity = np.full(len(p.dates), np.nan)
    rejects: Dict[str, int] = {}
    proposals = accepted = 0
    killed: Optional[int] = None
    inv_sum = 0.0

    def etf(s: int) -> bool:
        return p.syms[s] in p.ref_etfs

    def sell(s: int, price: float, d: int, reason: str, market_order: bool) -> None:
        nonlocal cash
        pos_s = pos.pop(s)
        fill = price * (1 - prm.slippage) if market_order else price
        tax = prm.sell_tax_etf if etf(s) else prm.sell_tax
        proceeds = pos_s["qty"] * fill * (1 - prm.commission - tax)
        cash += proceeds
        pnl = proceeds - pos_s["cost"]
        trades.append(Trade(p.syms[s], pos_s["opened"], d, pos_s["qty"], pos_s["entry"], fill, pnl / pos_s["cost"] * 100, pnl, reason))

    for d in range(first, last + 1):
        day = p.dates[d]
        dec = decisions.get(day) if killed is None else None

        # 1) AI 가 피하라고 한 보유 종목을 시가에 판다
        if dec and prm.ai_exits:
            for sym in dec.get("avoid", []):
                s = idx.get(str(sym).upper())
                if s is not None and s in pos and np.isfinite(p.o[d, s]) and p.o[d, s] > 0:
                    sell(s, float(p.o[d, s]), d, "AI 회피", True)

        # 2) 매수 제안 검증 후 시가 매수
        if dec and killed is None:
            buys = [b for b in dec.get("buys", []) if int(b.get("conviction", 0)) >= prm.min_conviction]
            buys.sort(key=lambda b: -int(b.get("conviction", 0)))
            slots = prm.max_positions - len(pos)
            picks: List[int] = []
            for b in buys:
                proposals += 1
                if len(picks) >= max(0, slots):
                    break
                sym = str(b.get("symbol", "")).strip().upper()
                s = idx.get(sym)
                why = None
                if s is None:
                    why = "시세 자료 없음"
                elif s in pos or s in picks:
                    why = "이미 보유/중복"
                elif not (np.isfinite(p.o[d, s]) and p.o[d, s] > 0 and np.isfinite(p.c[d - 1, s])):
                    why = "시세 없음"
                elif sym not in p.ref_etfs:
                    if p.krx and not name_matches(str(b.get("name", "")), p.names.get(sym)):
                        why = "종목명 불일치"
                    elif p.liquid is not None and not p.liquid[d, s]:
                        why = "거래대금 상위 아님"
                if why is None and prm.chase > 0 and p.o[d, s] / p.c[d - 1, s] - 1 >= prm.chase:
                    why = "급등 추격 금지"
                if why:
                    rejects[why] = rejects.get(why, 0) + 1
                    continue
                picks.append(s)
            if picks:
                invested = sum(v["cost"] for v in pos.values())
                available = max(0.0, cash)
                each = min(available / len(picks), prm.budget / prm.max_positions)
                for s in picks:
                    px = float(p.o[d, s]) * (1 + prm.slippage)
                    unit = px * (1 + prm.commission)
                    qty = int(each // unit)
                    if qty < 1:
                        rejects["1주도 못 삼"] = rejects.get("1주도 못 삼", 0) + 1
                        continue
                    cost = qty * unit
                    cash -= cost
                    pos[s] = {"qty": qty, "cost": cost, "entry": px, "opened": d}
                    accepted += 1

        # 3) 장중 익절·손절, 4) 보유일 초과는 종가에 정리
        for s in list(pos):
            e = pos[s]["entry"]
            hi, lo, op, cl = p.h[d, s], p.l[d, s], p.o[d, s], p.c[d, s]
            if not (np.isfinite(hi) and np.isfinite(lo) and np.isfinite(cl)):
                continue
            tp_px, sl_px = e * (1 + prm.take_profit), e * (1 - prm.stop_loss)
            hit_tp, hit_sl = hi >= tp_px, lo <= sl_px
            if hit_tp and hit_sl:
                if prm.same_bar == "target":
                    hit_sl = False
                else:
                    hit_tp = False
            if hit_sl:
                sell(s, min(float(op), sl_px) if np.isfinite(op) else sl_px, d, "손절", True)
            elif hit_tp:
                sell(s, max(float(op), tp_px) if np.isfinite(op) and pos[s]["opened"] != d else tp_px, d, "익절", False)
            elif prm.max_hold_days > 0 and (day.date() - p.dates[pos[s]["opened"]].date()).days >= prm.max_hold_days:
                sell(s, float(cl), d, "보유일 초과", True)

        # 5) 종가 평가와 누적 손실 한도
        value = sum(v["qty"] * (p.c[d, s] if np.isfinite(p.c[d, s]) else v["entry"]) for s, v in pos.items())
        eq = cash + value
        if killed is None and prm.loss_limit is not None and eq <= prm.budget * (1 - prm.loss_limit):
            for s in list(pos):
                sell(s, float(p.c[d, s]) if np.isfinite(p.c[d, s]) else pos[s]["entry"], d, "손실 한도", True)
            eq = cash
            killed = d
        equity[d] = eq
        inv_sum += value / eq if eq > 0 else 0.0

    # 남은 보유분은 마지막 종가로 평가해 정리한 것으로 본다(손익 집계용)
    for s in list(pos):
        sell(s, float(p.c[last, s]) if np.isfinite(p.c[last, s]) else pos[s]["entry"], last, "기간 종료", True)
    equity[last] = cash
    return Sim(p.dates, equity, trades, proposals, accepted, rejects, killed, inv_sum / max(1, last - first + 1))


def metrics(sim: Sim, first: int, last: int, budget: float) -> dict:
    eq = sim.equity[first:last + 1]
    eq = pd.Series(eq).ffill().to_numpy()
    days = (sim.dates[last] - sim.dates[first]).days
    total = eq[-1] / budget - 1
    peak = np.maximum.accumulate(eq)
    wins = [t for t in sim.trades if t.pnl > 0]
    return {
        "총수익률%": total * 100,
        "연환산%": ((1 + total) ** (365.25 / max(days, 1)) - 1) * 100 if total > -1 else float("nan"),
        "최대낙폭%": float(np.min(eq / peak - 1)) * 100,
        "거래수": len(sim.trades),
        "승률%": len(wins) / len(sim.trades) * 100 if sim.trades else float("nan"),
        "거래당평균%": float(np.mean([t.net_pct for t in sim.trades])) if sim.trades else float("nan"),
        "제안": sim.proposals,
        "통과": sim.accepted,
        "평균투자비중%": sim.invested_share * 100,
        "정지": "예" if sim.killed_at is not None else "-",
    }


# ─────────────────────────── 제안의 신호 품질 ───────────────────────────

def signal_quality(p: Panel, decisions: Dict[pd.Timestamp, dict], min_conviction: int, first: int, last: int, chase: float,
                   horizons=(1, 5, 10)) -> pd.DataFrame:
    """검증을 통과한 매수 제안(확신도 ≥ 기준)의 시가 매수 후 수익률을, 그날의 거래대금 상위 종목 평균과 비교한다(슬롯·청산 규칙 무관)."""
    idx = p.idx()
    rows = {h: {"pick": [], "base": []} for h in horizons}
    for d in range(first, last + 1):
        dec = decisions.get(p.dates[d])
        if not dec:
            continue
        seen = set()
        for b in dec.get("buys", []):
            if int(b.get("conviction", 0)) < min_conviction:
                continue
            sym = str(b.get("symbol", "")).strip().upper()
            s = idx.get(sym)
            if s is None or sym in seen or not (np.isfinite(p.o[d, s]) and p.o[d, s] > 0 and np.isfinite(p.c[d - 1, s])):
                continue
            if sym not in p.ref_etfs:
                if p.krx and not name_matches(str(b.get("name", "")), p.names.get(sym)):
                    continue
                if p.liquid is not None and not p.liquid[d, s]:
                    continue
            if chase > 0 and p.o[d, s] / p.c[d - 1, s] - 1 >= chase:
                continue
            seen.add(sym)
            for h in horizons:
                if d + h - 1 > last or d + h - 1 >= len(p.dates):
                    continue
                exit_c = p.c[d + h - 1, s]
                if not np.isfinite(exit_c):
                    continue
                pool = p.liquid[d] if p.liquid is not None else np.ones(len(p.syms), dtype=bool)
                ok = pool & np.isfinite(p.o[d]) & (p.o[d] > 0) & np.isfinite(p.c[d + h - 1])
                base = np.nanmean(p.c[d + h - 1, ok] / p.o[d, ok] - 1)
                rows[h]["pick"].append(exit_c / p.o[d, s] - 1)
                rows[h]["base"].append(base)
    out = {}
    for h, r in rows.items():
        if not r["pick"]:
            continue
        a, b = np.array(r["pick"]), np.array(r["base"])
        ex = a - b
        t = ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex))) if len(ex) > 2 and ex.std(ddof=1) > 0 else float("nan")
        out[f"{h}일째 종가" if h > 1 else "당일 종가"] = {"제안 수": len(a), "평균수익%": a.mean() * 100, "같은날 거래대금상위 평균%": b.mean() * 100,
                                                      "초과%p": ex.mean() * 100, "t값": t, "상승 비율%": float(np.mean(a > 0)) * 100}
    return pd.DataFrame(out).T


# ─────────────────────────── 무작위 대조군 ───────────────────────────

def random_decisions(p: Panel, decisions: Dict[pd.Timestamp, dict], min_conviction: int, seed: int) -> Dict[pd.Timestamp, dict]:
    """같은 날 같은 수의 제안을, 검증을 통과할 만한 종목(거래대금 상위) 중에서 무작위로 낸다(확신도 5)."""
    rng = np.random.default_rng(seed)
    out: Dict[pd.Timestamp, dict] = {}
    for day, dec in decisions.items():
        d = int(p.dates.get_loc(day)) if day in p.dates else None
        if d is None or d < 1:
            continue
        n = sum(1 for b in dec.get("buys", []) if int(b.get("conviction", 0)) >= min_conviction)
        if n == 0:
            continue
        pool = np.flatnonzero((p.liquid[d] if p.liquid is not None else True) & np.isfinite(p.o[d]) & (p.o[d] > 0) & np.isfinite(p.c[d - 1]))
        if len(pool) == 0:
            continue
        pick = rng.choice(pool, size=min(n * 2, len(pool)), replace=False)       # 슬롯이 남도록 넉넉히 내고 시뮬레이터가 자리만큼만 산다
        out[day] = {"buys": [{"symbol": p.syms[i], "name": p.names.get(p.syms[i], ""), "conviction": 5} for i in pick], "avoid": []}
    return out


# ─────────────────────────── 데이터 ───────────────────────────

def _read(db: str, sql: str, params=()):
    con = sqlite3.connect(f"file:{DATA / db}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def load_panel(market: str, start: str) -> Panel:
    if market == "kr":
        df = _read("candles_krx_all.sqlite", "select symbol,date,open,high,low,close,volume from candle where market='KRX' and date>=? order by date", (start,))
        meta = _read("candles_krx_all.sqlite", "select symbol, name from meta")
        names = dict(zip(meta["symbol"], meta["name"]))
        etf = _read("sector_etfs.sqlite", "select symbol,date,open,high,low,close,volume from price where market='KR' and date>=? order by date", (start,))
        etf_meta = _read("sector_etfs.sqlite", "select symbol, name from meta where market='KR'")
        names.update({s: n for s, n in zip(etf_meta["symbol"], etf_meta["name"])})
        refs = set(KR_REF_ETFS) | {BENCH["kr"]}
    else:
        df = _read("candles_us_sp500.sqlite", "select symbol,date,open,high,low,close,volume from candle where market='US' and date>=? order by date", (start,))
        names = {}
        etf = _read("sector_etfs.sqlite", "select symbol,date,open,high,low,close,volume from price where market='US' and date>=? order by date", (start,))
        refs = set(US_REF_ETFS) | {BENCH["us"]}
    all_ = pd.concat([df, etf[~etf["symbol"].isin(set(df["symbol"]))]], ignore_index=True)
    all_["date"] = pd.to_datetime(all_["date"])
    bench = all_[all_["symbol"] == BENCH[market]]
    dates = pd.DatetimeIndex(sorted(bench["date"].unique()))
    wide = {k: all_.pivot(index="date", columns="symbol", values=k).reindex(dates) for k in ("open", "high", "low", "close", "volume")}
    syms = list(wide["close"].columns)
    arr = {k: wide[k].reindex(columns=syms).to_numpy(dtype=float) for k in wide}
    panel = Panel(dates, syms, arr["open"], arr["high"], arr["low"], arr["close"], arr["volume"], names=names, ref_etfs=set(refs) & set(syms), krx=(market == "kr"))
    stock_mask = np.array([s not in panel.ref_etfs for s in syms])
    liq = liquidity_matrix(np.where(stock_mask, panel.c, np.nan), np.where(stock_mask, panel.v, np.nan), 1000 if market == "kr" else 10_000)
    panel.liquid = liq
    return panel


def load_decisions(cond: str, market: str) -> Dict[pd.Timestamp, dict]:
    d: Dict[pd.Timestamp, dict] = {}
    folder = STUDY / f"skill_decisions_{cond}" / market
    for f in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
            d[pd.Timestamp(f.stem)] = j
        except Exception as exc:  # noqa: BLE001
            print(f"  [경고] {f}: {exc}")
    return d


# ─────────────────────────── 하루치 뉴스 요약 만들기 ───────────────────────────

def build_days(market: str, start: str, end: str, max_headlines: int = 70, max_wiki: int = 12) -> None:
    panel_dates = load_panel(market, "2025-12-01").dates
    days = [d for d in panel_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    con = sqlite3.connect(f"file:{STUDY / 'gnews_hist.sqlite'}?mode=ro", uri=True)
    hl = pd.read_sql_query("select day, title, source from hl where lang=?", con, params=(market,))
    con.close()
    hl["day"] = pd.to_datetime(hl["day"])
    wcon = sqlite3.connect(f"file:{DATA / 'news_wiki.sqlite'}?mode=ro", uri=True)
    wiki = pd.read_sql_query("select date, text from news where date>=? order by date", wcon, params=("2025-12-01",))
    wcon.close()
    wiki["date"] = pd.to_datetime(wiki["date"])
    kr = market == "kr"
    refs = KR_REF_ETFS if kr else US_REF_ETFS
    panel = load_panel(market, "2025-12-01")
    names = panel.names
    out = STUDY / "skill_days" / market
    out_no = STUDY / "skill_days_nonews" / market
    out.mkdir(parents=True, exist_ok=True)
    out_no.mkdir(parents=True, exist_ok=True)
    etf_line = "Reference sector ETFs: " + ", ".join(f"{s} ({names.get(s, s)})" for s in refs if s in panel.syms)
    sizes = []
    for day in days:
        prev = panel_dates[panel_dates.get_loc(day) - 1]
        lo, hi = prev, day - pd.Timedelta(days=1)          # 직전 거래일 ~ 판단일 전날(주말·휴일 뉴스 포함). 판단일 당일 기사는 쓰지 않는다
        sub = hl[(hl["day"] >= lo) & (hl["day"] <= hi)].drop_duplicates("title")
        sub = sub.sort_values("day", ascending=False)
        if len(sub) > max_headlines:
            pos = np.linspace(0, len(sub) - 1, max_headlines).round().astype(int)
            sub = sub.iloc[sorted(set(pos))]
        lines = [f"{r.day.strftime('%m-%d')} {r.title}" + (f" ({r.source})" if r.source else "") for r in sub.itertuples()]
        wsub = wiki[(wiki["date"] >= lo) & (wiki["date"] <= hi)]
        wlines: List[str] = []
        for r in wsub.itertuples():
            cat = ""
            for ln in r.text.split("\n"):
                if ln.startswith("["):
                    cat = ln.strip("[]")
                elif ln.startswith("- ") and cat in ("Business and economy", "Science and technology", "International relations", "Armed conflicts and attacks"):
                    wlines.append(f"{r.date.strftime('%m-%d')} {ln[2:220]}")
        if len(wlines) > max_wiki:
            pos = np.linspace(0, len(wlines) - 1, max_wiki).round().astype(int)
            wlines = [wlines[i] for i in sorted(set(pos))]
        head = (f"DECISION DAY: {day.date()} ({'Korea' if kr else 'US'} market open). You decide at the open, using only news published BEFORE this day.\n"
                f"Market: {'Korean stocks (KRX): 6-digit codes, Korean company names' if kr else 'US-listed stocks: tickers, English company names'}.\n{etf_line}\n")
        text = head + "\n## Market headlines (Google News)\n" + "\n".join(lines or ["(none)"]) + "\n\n## World events (Wikipedia current events)\n" + "\n".join(wlines or ["(none)"]) + "\n"
        (out / f"{day.date()}.txt").write_text(text, encoding="utf-8")
        (out_no / f"{day.date()}.txt").write_text(head + "\n(No news provided.)\n", encoding="utf-8")
        sizes.append((len(text), len(lines), len(wlines)))
    s = np.array(sizes)
    print(f"{market}: 거래일 {len(days)}일 → {out} | 글자 평균 {s[:, 0].mean():,.0f} 최대 {s[:, 0].max():,} | 헤드라인 평균 {s[:, 1].mean():.0f}건, 위키 {s[:, 2].mean():.0f}건")


# ─────────────────────────── 실행·보고 ───────────────────────────

def run(market: str, conds: List[str], budget: Optional[float], loss_limit: Optional[float], start: str, end: str, seeds: int, min_conviction: int = 4) -> None:
    kr = market == "kr"
    panel = load_panel(market, "2025-10-01")
    first = int(panel.dates.searchsorted(pd.Timestamp(start)))
    last = int(panel.dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    budget = budget or (5_000_000.0 if kr else 3_000.0)
    base = dict(budget=budget, loss_limit=loss_limit, min_conviction=min_conviction)
    if kr:
        base.update(commission=0.00015, sell_tax=0.0020, sell_tax_etf=0.0, slippage=0.0007)
    else:
        base.update(commission=0.0001, sell_tax=0.00003, sell_tax_etf=0.00003, slippage=0.0005)
    prm = Params(**base)
    idx = panel.idx()
    bench = idx[BENCH[market]]
    bench_ret = panel.c[last, bench] / panel.c[first - 1, bench] - 1
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"\n######## {'한국' if kr else '미국'} | 확신도 ≥ {min_conviction} | 시험 {panel.dates[first].date()} ~ {panel.dates[last].date()} ({last - first + 1}거래일) | 시드 {budget:,.0f} · {prm.max_positions}종목 · "
          f"익절 {prm.take_profit:.0%} 손절 {prm.stop_loss:.0%} 보유 {prm.max_hold_days}일 · 손실한도 {'없음' if loss_limit is None else f'{loss_limit:.0%}'}")
    print(f"기준선: {BENCH[market]} 보유 {bench_ret * 100:+.1f}%")
    rows = {}
    decs = {}
    for cond in conds:
        decs[cond] = load_decisions(cond, market)
        dd = {k: v for k, v in decs[cond].items() if panel.dates[first] <= k <= panel.dates[last]}
        if not dd:
            print(f"  [{cond}] 판단 파일이 없습니다."); continue
        sim = simulate(panel, dd, prm, first, last)
        rows[f"[{cond}] AI 판단 ({len(dd)}일)"] = metrics(sim, first, last, budget)
        reasons = pd.Series([t.reason for t in sim.trades]).value_counts().to_dict()
        print(f"  [{cond}] 거래 청산 사유: {reasons} | 제안 거부 사유: {sim.rejects}")
        if cond == conds[0]:
            ref_dec = dd
            ref_sim = sim
    if conds and conds[0] in decs and decs[conds[0]]:
        rnd = []
        for sd in range(seeds):
            rd = random_decisions(panel, {k: v for k, v in decs[conds[0]].items() if panel.dates[first] <= k <= panel.dates[last]}, prm.min_conviction, 100 + sd)
            rnd.append(metrics(simulate(panel, rd, prm, first, last), first, last, budget))
        rt = pd.DataFrame(rnd)
        rows[f"[대조] 무작위 선정 (평균 {seeds}회)"] = rt.mean(numeric_only=True).to_dict() | {"정지": f"{(rt['정지'] == '예').mean():.0%}"}
        print(f"  무작위 선정 {seeds}회 총수익률% 범위: {rt['총수익률%'].min():.1f} ~ {rt['총수익률%'].max():.1f}")
    print(pd.DataFrame(rows).T.round(2).to_string())
    for cond in conds:
        if decs.get(cond):
            dd = {k: v for k, v in decs[cond].items() if panel.dates[first] <= k <= panel.dates[last]}
            sq = signal_quality(panel, dd, prm.min_conviction, first, last, prm.chase)
            print(f"\n-- [{cond}] 제안의 신호 품질(검증 통과 제안, 시가 매수 후 N일 종가, 슬롯·청산 규칙 없이)")
            print(sq.round(2).to_string() if len(sq) else "(통과한 제안 없음)")
    # 월별
    if "news" in conds and decs.get("news"):
        dd = {k: v for k, v in decs["news"].items() if panel.dates[first] <= k <= panel.dates[last]}
        sim = simulate(panel, dd, prm, first, last)
        s = pd.Series(sim.equity[first:last + 1], index=panel.dates[first:last + 1]).ffill()
        m = s.groupby([s.index.year, s.index.month]).last()
        pm = pd.concat([pd.Series([budget]), m]).pct_change().iloc[1:] * 100
        pm.index = [f"{a}-{b:02d}" for a, b in m.index]
        bm = pd.Series(panel.c[:, bench], index=panel.dates)
        bm = bm[(bm.index >= panel.dates[first - 1]) & (bm.index <= panel.dates[last])]
        bmm = bm.groupby([bm.index.year, bm.index.month]).last().pct_change().iloc[1:] * 100
        print("\n-- 월별 수익률(%): news vs 지수")
        print(pd.DataFrame({"news": pm.round(1), BENCH[market]: pd.Series(bmm.to_numpy(), index=[f"{a}-{b:02d}" for a, b in bmm.index]).round(1)}).T.to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-days", "run"])
    ap.add_argument("--market", choices=["kr", "us"], required=True)
    ap.add_argument("--start", default="2026-01-02")
    ap.add_argument("--end", default="2026-11-12")
    ap.add_argument("--conds", nargs="+", default=["news"])
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--no-loss-limit", action="store_true")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--min-conviction", type=int, default=4, help="스킬의 매수 기준 확신도(3~5)")
    a = ap.parse_args()
    if a.cmd == "build-days":
        build_days(a.market, a.start, a.end)
    else:
        run(a.market, a.conds, a.budget, None if a.no_loss_limit else 0.10, a.start, a.end, a.seeds, a.min_conviction)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
