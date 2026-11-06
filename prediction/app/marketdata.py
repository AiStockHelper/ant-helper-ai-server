"""대량 일봉 수집. 한국은 pykrx, 미국은 yfinance 로 받아 로컬 SQLite 에 쌓아 두고 증분 갱신한다.

야간 배치 전용이다(요청 처리 중에는 쓰지 않는다). 최신 시세·주문은 Spring 이 증권사(KIS)로 직접 처리한다.
DB 파일은 지워도 다음 배치에서 다시 채워지는 캐시일 뿐이다.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("PREDICTION_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DB_PATH = DATA_DIR / "candles.sqlite"

# 처음 채울 때 거슬러 올라갈 달력일 수(예측에는 120 거래일 안팎이면 충분하다)
BACKFILL_DAYS = 300
OVERLAP_DAYS = 7
US_CHUNK = 200

Row = Tuple[str, str, float, float, float, float, float]  # symbol, date, open, high, low, close, volume
Progress = Callable[[int, int], None]

_SEOUL = ZoneInfo("Asia/Seoul")
_NEW_YORK = ZoneInfo("America/New_York")


class CandleStore:
    """시장별 일봉 저장소(SQLite)."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS candle ("
                "market TEXT NOT NULL, symbol TEXT NOT NULL, date TEXT NOT NULL,"
                "open REAL, high REAL, low REAL, close REAL, volume REAL,"
                "PRIMARY KEY (market, symbol, date))"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def last_dates(self, market: str) -> Dict[str, str]:
        with self._lock, self._conn() as c:
            cur = c.execute("SELECT symbol, MAX(date) FROM candle WHERE market = ? GROUP BY symbol", (market,))
            return {s: d for s, d in cur.fetchall()}

    def upsert(self, market: str, rows: Iterable[Row]) -> int:
        batch = [(market,) + tuple(r) for r in rows]
        if not batch:
            return 0
        with self._lock, self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO candle VALUES (?,?,?,?,?,?,?,?)", batch)
        return len(batch)

    def load(self, market: str, symbols: Sequence[str], limit: int = 200) -> Dict[str, List[Tuple[str, float, float, float, float, float]]]:
        """symbol -> [(date, open, high, low, close, volume)] 날짜 오름차순, 최근 limit 개"""
        out: Dict[str, List[Tuple[str, float, float, float, float, float]]] = {}
        with self._lock, self._conn() as c:
            for sym in symbols:
                cur = c.execute(
                    "SELECT date, open, high, low, close, volume FROM candle "
                    "WHERE market = ? AND symbol = ? ORDER BY date DESC LIMIT ?",
                    (market, sym, limit),
                )
                rows = cur.fetchall()
                rows.reverse()
                out[sym] = rows
        return out

    def prune(self, keep_days: int = 420) -> None:
        cutoff = (date.today() - timedelta(days=keep_days)).isoformat()
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM candle WHERE date < ?", (cutoff,))


# ───────────────────────── 미완성 봉 제거 ─────────────────────────

def drop_open_session_bar(rows: List[Row], market: str, now: Optional[datetime] = None) -> List[Row]:
    """장이 아직 끝나지 않은 날의 봉은 하루치가 아니므로 빼고 쓴다."""
    if not rows:
        return rows
    if market == "KRX":
        local = (now or datetime.now(_SEOUL)).astimezone(_SEOUL)
        closed = local.hour * 60 + local.minute >= 15 * 60 + 40
    else:
        local = (now or datetime.now(_NEW_YORK)).astimezone(_NEW_YORK)
        closed = local.hour * 60 + local.minute >= 16 * 60 + 10
    today = local.date().isoformat()
    if closed:
        return rows
    return [r for r in rows if r[1] != today]


# ───────────────────────── 한국: pykrx ─────────────────────────

def fetch_krx(symbols: Sequence[str], last_dates: Dict[str, str], progress: Optional[Progress] = None) -> List[Row]:
    from pykrx import stock  # 지연 import: 테스트·미사용 환경에서 필요 없게

    end = date.today().strftime("%Y%m%d")
    total = len(symbols)

    def one(sym: str) -> List[Row]:
        last = last_dates.get(sym)
        start = date.fromisoformat(last) - timedelta(days=OVERLAP_DAYS) if last else date.today() - timedelta(days=BACKFILL_DAYS)
        df = None
        for attempt in range(3):
            try:
                df = stock.get_market_ohlcv_by_date(start.strftime("%Y%m%d"), end, sym)
                break
            except Exception as exc:  # noqa: BLE001 - 외부 라이브러리 오류는 종목 단위로 흡수
                logger.debug("pykrx %s 실패(%d): %s", sym, attempt + 1, exc)
                time.sleep(0.5 * (attempt + 1))
        rows: List[Row] = []
        if df is not None and len(df) > 0:
            for idx, r in zip(df.index, df.itertuples(index=False)):
                close = float(r[3])
                if close > 0:
                    rows.append((sym, idx.strftime("%Y-%m-%d"), float(r[0]), float(r[1]), float(r[2]), close, float(r[4])))
        return rows

    # pykrx 는 스레드 여러 개로 동시에 부르면 멈추는 경우가 있어 한 줄로 호출한다(종목당 약 25ms, 2,700종목 약 1분).
    out: List[Row] = []
    for i, sym in enumerate(symbols, 1):
        out.extend(one(sym))
        if progress and i % 100 == 0:
            progress(i, total)
    if progress:
        progress(total, total)
    return drop_open_session_bar(out, "KRX")


# ───────────────────────── 미국: yfinance ─────────────────────────

def fetch_us(symbols: Sequence[str], last_dates: Dict[str, str], progress: Optional[Progress] = None) -> List[Row]:
    import pandas as pd
    import yfinance as yf

    known = [last_dates[s] for s in symbols if s in last_dates]
    fresh_enough = len(known) >= 0.9 * len(symbols) if symbols else False
    kwargs: Dict[str, object]
    if fresh_enough:
        start = date.fromisoformat(min(known)) - timedelta(days=OVERLAP_DAYS)
        kwargs = {"start": start.isoformat()}
    else:
        kwargs = {"period": "9mo"}

    out: List[Row] = []
    total = len(symbols)
    for i in range(0, total, US_CHUNK):
        chunk = list(symbols[i:i + US_CHUNK])
        if len(chunk) == 1:
            chunk = chunk * 2  # 단일 종목일 때 열 구조가 달라지는 것을 피한다
        df = None
        for attempt in range(3):
            try:
                df = yf.download(chunk, interval="1d", group_by="ticker", auto_adjust=False, threads=True,
                                 progress=False, **kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("yfinance 묶음 실패(%d회): %s", attempt + 1, exc)
                time.sleep(3 * (attempt + 1))
        if df is not None and not df.empty:
            level0 = set(df.columns.get_level_values(0)) if isinstance(df.columns, pd.MultiIndex) else set()
            for sym in dict.fromkeys(chunk):
                if sym not in level0:
                    continue
                sub = df[sym].dropna(subset=["Close"])[["Open", "High", "Low", "Close", "Volume"]]
                for idx, o, h, l, c, v in sub.itertuples():
                    if c and c > 0:
                        out.append((sym, idx.strftime("%Y-%m-%d"), float(o), float(h), float(l), float(c), float(v or 0)))
        if progress:
            progress(min(i + US_CHUNK, total), total)
    return drop_open_session_bar(out, "US")


def fetch(market: str, symbols: Sequence[str], last_dates: Dict[str, str], progress: Optional[Progress] = None) -> List[Row]:
    if os.environ.get("PREDICTION_OFFLINE") == "1":
        # 개발용: 새로 내려받지 않고 이미 저장된 일봉만으로 분석한다(외부 사이트가 느리거나 막혔을 때).
        return []
    return fetch_krx(symbols, last_dates, progress) if market == "KRX" else fetch_us(symbols, last_dates, progress)
