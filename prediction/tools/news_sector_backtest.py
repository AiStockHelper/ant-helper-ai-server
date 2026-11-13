"""월말 뉴스 요약으로 '다음 달에 유망한 산업(섹터 ETF)'을 고르는 방식의 백테스트.

흐름
  1) build-news : data/news_wiki.sqlite(위키피디아 일자별 뉴스) → 월별 요약 파일 data/news_study/months/YYYY-MM.txt
                 (그 달 뉴스만, 선택 가능한 섹터 ETF 목록 포함. 미래 정보가 들어갈 수 없게 월별로 따로 만든다)
  2) (AI 판단)  : 각 월 파일만 읽고 섹터 순위를 data/news_study/decisions_<조건>/YYYY-MM.json 에 쓴다
                 {"month": "2020-03", "us": ["XLK", ...좋을 것 같은 순서로 전부], "kr": ["091160", ...]}
                 조건: news(뉴스 제공), nonews(달 이름만 제공 — AI 기억만으로 맞히는지 확인하는 대조군), shuffled(다른 달 뉴스)
  3) evaluate   : 월말 결정 → 다음 달 첫 거래일 시가 매수, 그 달 마지막 거래일 종가 매도. 상위 3개 동일비중.
                 비교: 같은 시점 전체 섹터 동일비중, 지수(SPY/KODEX200), 무작위 3개(순열 검정), 대조군.

사용: python -m tools.news_sector_backtest build-news | evaluate [--conds news nonews] [--top 3]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PREDICTION_DATA_DIR", ROOT / "data"))
STUDY = DATA / "news_study"

PRIORITY = [("Business and economy", 9000), ("Science and technology", 3000), ("Health and environment", 2500), ("International relations", 2500),
            ("Armed conflicts and attacks", 2000), ("Politics and elections", 2500), ("Disasters and accidents", 1000), ("Law and crime", 500)]
ITEM_CHARS = 260
SIDE_COST = {"us": 0.0003, "kr": 0.0012}                      # ETF 편도 비용(수수료+호가차·슬리피지). 국내 주식형 ETF 는 거래세·매매차익세가 없다.
BENCH = {"us": "SPY", "kr": "069500"}


def conn(name: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DATA / name}?mode=ro", uri=True)


def etf_meta() -> Dict[str, pd.DataFrame]:
    con = conn("sector_etfs.sqlite")
    m = pd.read_sql_query("select market, symbol, name, first_date from meta", con)
    con.close()
    m["first_date"] = pd.to_datetime(m["first_date"])
    return {k.lower(): g.set_index("symbol") for k, g in m.groupby("market")}


def etf_prices() -> Dict[str, Dict[str, pd.DataFrame]]:
    con = conn("sector_etfs.sqlite")
    df = pd.read_sql_query("select market, symbol, date, open, close from price order by date", con)
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for (mk, sym), g in df.groupby(["market", "symbol"]):
        out.setdefault(mk.lower(), {})[sym] = g.set_index("date")[["open", "close"]]
    return out


def decision_months(first: str = "2015-12", last: str = "2026-10") -> List[pd.Period]:
    return list(pd.period_range(first, last, freq="M"))


def available(meta: pd.DataFrame, month: pd.Period, bench: str) -> pd.DataFrame:
    """결정일(월말) 20일 전에 이미 상장돼 있던 섹터 ETF(벤치마크 제외)."""
    cutoff = month.end_time.normalize() - pd.Timedelta(days=20)
    return meta[(meta["first_date"] <= cutoff) & (meta.index != bench)]


# ───────────────────────── 1) 월별 뉴스 요약 ─────────────────────────

def parse_day(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    cur = None
    for line in text.split("\n"):
        m = re.fullmatch(r"\[(.+)\]", line.strip())
        if m:
            cur = m.group(1)
            out.setdefault(cur, [])
        elif line.startswith("- ") and cur:
            out[cur].append(line[2:].strip())
    return out


def spread(items: List[tuple], budget: int) -> List[tuple]:
    """항목을 한 달에 고르게 흩어서 글자 예산 안에 들어가게 고른다."""
    total = sum(min(len(t), ITEM_CHARS) + 8 for _, t in items)
    if total <= budget:
        return items
    keep = max(1, int(len(items) * budget / total))
    pos = np.linspace(0, len(items) - 1, keep).round().astype(int)
    return [items[i] for i in sorted(set(pos))]


def build_news() -> None:
    news = pd.read_sql_query("select date, text from news order by date", conn("news_wiki.sqlite"))
    news["date"] = pd.to_datetime(news["date"])
    meta = etf_meta()
    (STUDY / "months").mkdir(parents=True, exist_ok=True)
    sizes = []
    for mo in decision_months():
        sub = news[(news["date"] >= mo.start_time) & (news["date"] <= mo.end_time)]
        if sub.empty:
            continue
        cats: Dict[str, List[tuple]] = {}
        for d, t in zip(sub["date"], sub["text"]):
            for cat, items in parse_day(t).items():
                cats.setdefault(cat, []).extend((d.strftime("%m-%d"), x) for x in items)
        parts = []
        for cat, budget in PRIORITY:
            items = spread(cats.get(cat, []), budget)
            if items:
                parts.append(f"## {cat}\n" + "\n".join(f"{d} {x[:ITEM_CHARS]}" for d, x in items))
        us = available(meta["us"], mo, BENCH["us"])
        kr = available(meta["kr"], mo, BENCH["kr"])
        head = (f"NEWS DIGEST — events reported during {mo.strftime('%B %Y')} (you are deciding on the last trading day of the month).\n"
                f"Available US sector ETFs: " + ", ".join(f"{s} ({n})" for s, n in zip(us.index, us["name"])) + "\n"
                f"Available Korea sector ETFs: " + ", ".join(f"{s} ({n})" for s, n in zip(kr.index, kr["name"])) + "\n")
        text = head + "\n" + "\n\n".join(parts) + "\n"
        (STUDY / "months" / f"{mo}.txt").write_text(text, encoding="utf-8")
        (STUDY / "months_nonews").mkdir(parents=True, exist_ok=True)
        nohead = head.replace(f"NEWS DIGEST — events reported during {mo.strftime('%B %Y')} (you are deciding on the last trading day of the month).", f"NO NEWS PROVIDED — you are deciding on the last trading day of {mo.strftime('%B %Y')}.")
        (STUDY / "months_nonews" / f"{mo}.txt").write_text(nohead, encoding="utf-8")
        sizes.append(len(text))
    print(f"월별 요약 {len(sizes)}개 → {STUDY / 'months'} | 글자 수 평균 {np.mean(sizes):,.0f}, 최대 {max(sizes):,}")


# ───────────────────────── 3) 평가 ─────────────────────────

def hold_return(px: pd.DataFrame, month: pd.Period) -> Optional[float]:
    """결정월 다음 달: 첫 거래일 시가 → 마지막 거래일 종가."""
    nxt = month + 1
    s = px[(px.index >= nxt.start_time) & (px.index <= nxt.end_time)]
    if len(s) < 10 or not (s["open"].iloc[0] > 0):
        return None
    return float(s["close"].iloc[-1] / s["open"].iloc[0] - 1)


def load_decisions(cond: str) -> Dict[pd.Period, dict]:
    d = {}
    for p in sorted((STUDY / f"decisions_{cond}").glob("*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            d[pd.Period(j.get("month") or p.stem, freq="M")] = j
        except Exception as exc:  # noqa: BLE001
            print(f"  [경고] {p.name}: {exc}")
    return d


def evaluate_market(mk: str, conds: List[str], top: int, perm: int, since: Optional[str]) -> None:
    meta, px = etf_meta()[mk], etf_prices()[mk]
    dec = {c: load_decisions(c) for c in conds}
    months = [m for m in decision_months() if m in set.intersection(*[set(dec[c]) for c in conds])]
    if since:
        months = [m for m in months if m >= pd.Period(since, freq="M")]
    if not months:
        print(f"[{mk}] 판단 결과가 없습니다."); return
    rng = np.random.default_rng(5)
    rows: Dict[str, List[dict]] = {c: [] for c in conds}
    ew_series, bench_series, perm_pool = [], [], []
    for m in months:
        avail = available(meta, m, BENCH[mk])
        rets = {s: hold_return(px[s], m) for s in avail.index if s in px}
        rets = {s: r for s, r in rets.items() if r is not None}
        if len(rets) < 4:
            continue
        b = hold_return(px[BENCH[mk]], m)
        ew = float(np.mean(list(rets.values())))
        ew_series.append((m, ew)); bench_series.append((m, b))
        perm_pool.append((m, np.array(list(rets.values()))))
        for c in conds:
            order = [s for s in (str(x) for x in dec[c][m].get(mk, [])) if s in rets]
            order += [s for s in rets if s not in order]                      # 빠뜨린 섹터는 맨 뒤
            sel = order[:top]
            pr = float(np.mean([rets[s] for s in sel]))
            rank = {s: i for i, s in enumerate(order)}
            ic = pd.Series({s: -rank[s] for s in rets}).rank().corr(pd.Series(rets).rank())      # 스피어만 순위상관
            rows[c].append(dict(month=m, ret=pr, ew=ew, bench=b, sel=sel, ic=ic, n=len(rets), allret=rets, order=order))
    n = len(rows[conds[0]])
    print(f"\n######## {mk.upper()} 섹터 ETF | 판단 {n}개월 ({months[0]} ~ {months[-1]} 결정, 다음 달 보유) | 상위 {top}개 동일비중, 월 1회 교체")
    side = SIDE_COST[mk]

    def apply_costs(rs: List[dict]) -> np.ndarray:
        out, prev = [], set()
        for r in rs:
            cur = set(r["sel"])
            turn = len(cur ^ prev) / (2 * top) if prev else 1.0                  # 바뀐 비중(매수+매도)
            out.append(r["ret"] - 2 * turn * side)
            prev = cur
        return np.array(out)

    ew_arr = np.array([x for _, x in ew_series])
    b_arr = np.array([x if x is not None else np.nan for _, x in bench_series])
    yrs = n / 12

    def summ(arr: np.ndarray) -> dict:
        eq = np.cumprod(1 + arr)
        return {"총수익률%": (eq[-1] - 1) * 100, "연환산%": (eq[-1] ** (1 / yrs) - 1) * 100, "최대낙폭%": float(np.min(eq / np.maximum.accumulate(eq) - 1)) * 100,
                "월평균%": arr.mean() * 100, "월승률%": float(np.mean(arr > 0)) * 100}

    table = {"전체 섹터 동일비중(기준)": summ(ew_arr), f"지수({BENCH[mk]})": summ(np.nan_to_num(b_arr))}
    stat = {}
    for c in conds:
        gross = np.array([r["ret"] for r in rows[c]])
        net = apply_costs(rows[c])
        table[f"[{c}] 상위{top} (비용 반영)"] = summ(net)
        ex = gross - ew_arr
        t = ex.mean() / (ex.std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan
        # 순열 검정: 같은 달 같은 후보에서 무작위로 top 개를 골랐을 때 평균 초과수익이 이만큼 나올 확률
        draws = np.array([np.mean([rng.choice(pool, size=min(top, len(pool)), replace=False).mean() - pool.mean() for _, pool in perm_pool]) for _ in range(perm)])
        pval = float(np.mean(draws >= ex.mean()))
        stat[c] = {"월평균 초과(전체 동일비중 대비)%": ex.mean() * 100, "t값": t, "초과 월승률%": float(np.mean(ex > 0)) * 100,
                   "순위상관(IC) 평균": float(np.nanmean([r["ic"] for r in rows[c]])), "무작위 대비 p값": pval, "비용 전 연환산%": (np.prod(1 + gross) ** (1 / yrs) - 1) * 100}
    pd.set_option("display.width", 220)
    print(pd.DataFrame(table).T.round(1).to_string())
    print("\n-- 예측력 지표(비용 전): 월평균 초과수익, t값, 순위상관, 무작위 3개 선택 대비 p값(작을수록 우연이 아님)")
    print(pd.DataFrame(stat).T.round(3).to_string())

    ys = {}
    for c in conds:
        s = pd.Series(apply_costs(rows[c]), index=[r["month"] + 1 for r in rows[c]])
        ys[f"[{c}]"] = s.groupby(s.index.year).apply(lambda x: (np.prod(1 + x) - 1) * 100)
    e = pd.Series(ew_arr, index=[m + 1 for m, _ in ew_series]); ys["동일비중"] = e.groupby(e.index.year).apply(lambda x: (np.prod(1 + x) - 1) * 100)
    bb = pd.Series(np.nan_to_num(b_arr), index=[m + 1 for m, _ in bench_series]); ys["지수"] = bb.groupby(bb.index.year).apply(lambda x: (np.prod(1 + x) - 1) * 100)
    print("\n-- 연도별 수익률(%, 비용 반영, 보유한 달 기준)")
    print(pd.DataFrame(ys).round(1).to_string())
    c0 = conds[0]
    picks = pd.Series([s for r in rows[c0] for s in r["sel"]]).value_counts().head(6)
    print(f"\n[{c0}] 가장 자주 고른 섹터: " + ", ".join(f"{meta['name'].get(s, s)}({k}회)" for s, k in picks.items()))
    out = STUDY / f"result_{mk}.csv"
    pd.DataFrame({c: pd.Series({str(r['month']): r["ret"] for r in rows[c]}) for c in conds} | {"ew": pd.Series({str(m): x for m, x in ew_series})}).to_csv(out, encoding="utf-8-sig")
    print(f"저장: {out}")


def make_momentum(lookbacks=(3, 6)) -> None:
    """뉴스 없이 가격만으로 만든 기준선: 결정일까지 최근 k개월 수익률이 높은 섹터 순. decisions_mom<k>/ 에 같은 형식으로 쓴다."""
    meta, px = etf_meta(), etf_prices()
    for k in lookbacks:
        out = STUDY / f"decisions_mom{k}"
        out.mkdir(parents=True, exist_ok=True)
        for m in decision_months():
            j = {"month": str(m)}
            for mk in ("us", "kr"):
                score = {}
                for s in available(meta[mk], m, BENCH[mk]).index:
                    if s not in px[mk]:
                        continue
                    close = px[mk][s]["close"]
                    end = close[close.index <= m.end_time]
                    start = close[close.index <= (m - k).end_time]
                    if len(end) and len(start) and start.iloc[-1] > 0:
                        score[s] = end.iloc[-1] / start.iloc[-1] - 1
                j[mk] = sorted(score, key=lambda s: -score[s])
            (out / f"{m}.json").write_text(json.dumps(j), encoding="utf-8")
    print(f"모멘텀 기준선 {lookbacks}개월 → {STUDY}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-news", "evaluate", "make-momentum"])
    ap.add_argument("--conds", nargs="+", default=["news"])
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--since", default=None, help="이 달 이후 결정만(예: 2025-07)")
    ap.add_argument("--market", nargs="+", default=["us", "kr"])
    a = ap.parse_args()
    if a.cmd == "build-news":
        build_news()
    elif a.cmd == "make-momentum":
        make_momentum()
    else:
        for mk in a.market:
            evaluate_market(mk, a.conds, a.top, a.perm, a.since)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
