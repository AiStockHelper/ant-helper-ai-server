"""야간 배치: 받은 종목 목록의 일봉을 수집·저장하고, 유동성 좋은 종목을 골라 전부 스코어링한다.

Spring 이 종목 목록을 넘기고(`/v1/batch/screen`), 작업 상태를 확인하다가(`/v1/batch/{id}`),
끝나면 결과를 가져가 DB 에 저장한다(`/v1/batch/{id}/result`). 이 서버는 결과를 영구 보관하지 않는다.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import marketdata
from .currency import DEFAULT_CURRENCY
from . import factors
from .indicators import InsufficientDataError, prepare_frame
from .screener import analyze_frame

logger = logging.getLogger(__name__)

# 시장별 유동성·가격 하한(해당 통화). 요청으로 바꿀 수 있다.
DEFAULT_MIN_TRADE_VALUE = {"KRX": 1_000_000_000.0, "US": 20_000_000.0}
DEFAULT_MIN_PRICE = {"KRX": 1_000.0, "US": 5.0}
CURRENCY = {"KRX": "KRW", "US": "USD"}
LIQUIDITY_WINDOW = 20
STALE_DAYS = 7
CANDLES_PER_STOCK = 160

Candle6 = Tuple[str, float, float, float, float, float]


class JobBusy(RuntimeError):
    """같은 시장의 작업이 이미 실행 중."""


@dataclass
class Job:
    id: str
    market: str
    status: str = "RUNNING"  # RUNNING | DONE | FAILED
    phase: str = "대기"
    done: int = 0
    total: int = 0
    error: Optional[str] = None
    trade_date: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    items: List[Dict[str, Any]] = field(default_factory=list)


_jobs: Dict[str, Job] = {}
_lock = threading.Lock()


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def start_job(
    market: str,
    symbols: Sequence[Dict[str, Optional[str]]],
    top_liquid: int,
    min_trade_value: Optional[float] = None,
    min_price: Optional[float] = None,
    store: Optional[marketdata.CandleStore] = None,
    fetcher: Optional[Callable[..., List[marketdata.Row]]] = None,
    workers: Optional[int] = None,
) -> Job:
    with _lock:
        if any(j.market == market and j.status == "RUNNING" for j in _jobs.values()):
            raise JobBusy(f"{market} 배치가 이미 실행 중입니다.")
        job = Job(id=uuid.uuid4().hex[:12], market=market, total=len(symbols))
        _jobs[job.id] = job
        for old in sorted(_jobs.values(), key=lambda j: j.started_at)[:-6]:
            if old.status != "RUNNING":
                _jobs.pop(old.id, None)
    thread = threading.Thread(
        target=_run,
        args=(job, list(symbols), top_liquid,
              min_trade_value if min_trade_value is not None else DEFAULT_MIN_TRADE_VALUE[market],
              min_price if min_price is not None else DEFAULT_MIN_PRICE[market],
              store, fetcher or marketdata.fetch, workers),
        name=f"batch-{market}-{job.id}",
        daemon=True,
    )
    thread.start()
    return job


def _run(job: Job, symbols: List[Dict[str, Optional[str]]], top_liquid: int, min_trade_value: float,
         min_price: float, store: Optional[marketdata.CandleStore], fetcher: Callable[..., List[marketdata.Row]],
         workers: Optional[int]) -> None:
    try:
        store = store or marketdata.CandleStore()
        syms = [s["symbol"] for s in symbols if s.get("symbol")]
        meta = {s["symbol"]: s for s in symbols if s.get("symbol")}

        job.phase = "일봉 수집"
        job.total, job.done = len(syms), 0

        def progress(done: int, total: int) -> None:
            job.done, job.total = done, total

        rows = fetcher(job.market, syms, store.last_dates(job.market), progress)
        store.upsert(job.market, rows)
        store.prune()
        logger.info("[%s] 일봉 %d건 저장 (%d종목 요청)", job.market, len(rows), len(syms))

        job.phase = "유동성 필터"
        candles = store.load(job.market, syms, CANDLES_PER_STOCK)
        picked = select_liquid(candles, top_liquid, min_trade_value, min_price)
        if not picked:
            raise RuntimeError("조건을 만족하는 종목이 없습니다. 데이터 수집이 실패했을 수 있습니다.")
        job.trade_date = max(p[3][-1][0] for p in picked)

        job.phase = "스코어링"
        job.total, job.done = len(picked), 0
        currency = CURRENCY[job.market]
        tasks = [(sym, (meta[sym].get("name") or sym), currency, bars) for sym, _tv, _px, bars in picked]
        trade_value = {sym: tv for sym, tv, _px, _b in picked}
        results = screen_all(tasks, progress, workers)
        for r in results:
            r["market"] = meta[r["symbol"]].get("market")
            r["avgTradeValue"] = trade_value[r["symbol"]]
        factors.apply_ranking(results, factors.RANK_UNIVERSE)
        results.sort(key=lambda r: r["score"], reverse=True)
        job.items = results
        job.phase = "완료"
        job.status = "DONE"
        logger.info("[%s] 배치 완료: 후보 %d → 결과 %d (기준일 %s)", job.market, len(picked), len(results), job.trade_date)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] 배치 실패", job.market)
        job.status = "FAILED"
        job.error = str(exc)[:500]
    finally:
        job.finished_at = time.time()


def select_liquid(candles: Dict[str, List[Candle6]], top_n: int, min_trade_value: float, min_price: float):
    """최근 20일 평균 거래대금이 큰 순으로 top_n 개. 가격 하한·최신성 조건을 만족해야 한다.

    반환: [(symbol, 평균거래대금, 마지막 종가, 일봉)] 거래대금 내림차순
    """
    latest = max((bars[-1][0] for bars in candles.values() if bars), default=None)
    if latest is None:
        return []
    from datetime import date, timedelta
    cutoff = (date.fromisoformat(latest) - timedelta(days=STALE_DAYS)).isoformat()
    out = []
    for sym, bars in candles.items():
        if len(bars) < 60 or bars[-1][0] < cutoff:
            continue
        recent = bars[-LIQUIDITY_WINDOW:]
        value = sum(b[4] * b[5] for b in recent) / len(recent)
        last_close = bars[-1][4]
        if value < min_trade_value or last_close < min_price:
            continue
        out.append((sym, value, last_close, bars))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]


def screen_one(task: Tuple[str, str, str, List[Candle6]]) -> Optional[Dict[str, Any]]:
    """프로세스 풀에서 실행되는 순수 함수. 실패하면 None."""
    symbol, name, currency, bars = task
    try:
        candles = [SimpleNamespace(date=b[0], open=b[1], high=b[2], low=b[3], close=b[4], volume=b[5]) for b in bars]
        df = prepare_frame(candles)
        rec = analyze_frame(df, symbol, name, currency or DEFAULT_CURRENCY)
        rec["tradeDate"] = bars[-1][0]
        return rec
    except InsufficientDataError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("스코어링 실패 %s: %s", symbol, exc)
        return None


def screen_all(tasks: List[Tuple[str, str, str, List[Candle6]]], progress: Callable[[int, int], None],
               workers: Optional[int] = None) -> List[Dict[str, Any]]:
    total = len(tasks)
    out: List[Dict[str, Any]] = []
    # 프로세스마다 pandas/scikit-learn 이 올라가 메모리를 쓰므로 기본은 최대 4개. PREDICTION_WORKERS 로 조정한다.
    default_workers = int(os.environ.get("PREDICTION_WORKERS", max(1, min(4, (os.cpu_count() or 2) - 1))))
    n = workers if workers is not None else default_workers
    if n <= 1 or total < 20:
        for i, t in enumerate(tasks, 1):
            r = screen_one(t)
            if r:
                out.append(r)
            if i % 50 == 0:
                progress(i, total)
        progress(total, total)
        return out
    try:
        with ProcessPoolExecutor(max_workers=n) as pool:
            for i, r in enumerate(pool.map(screen_one, tasks, chunksize=10), 1):
                if r:
                    out.append(r)
                if i % 50 == 0:
                    progress(i, total)
        progress(total, total)
        return out
    except Exception as exc:  # noqa: BLE001 - 프로세스 풀을 못 쓰는 환경이면 한 줄로 처리한다
        logger.warning("프로세스 풀 실패, 순차 처리로 전환: %s", exc)
        return screen_all(tasks, progress, workers=1)
