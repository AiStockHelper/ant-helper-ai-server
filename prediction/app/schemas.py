"""요청/응답 모델. JSON 필드명은 docs/DESIGN.md 5절과 동일(camelCase)."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

Direction = Literal["UP", "DOWN", "FLAT"]
Tone = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
Currency = Literal["KRW", "USD"]

MAX_CANDLES = 2000
MAX_SCREEN_ITEMS = 200


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Candle(CamelModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictRequest(CamelModel):
    symbol: str = Field(min_length=1)
    name: Optional[str] = None
    currency: Currency = "KRW"
    candles: List[Candle] = Field(max_length=MAX_CANDLES)


class ScreenRequestItem(CamelModel):
    symbol: str = Field(min_length=1)
    name: Optional[str] = None
    currency: Currency = "KRW"
    candles: List[Candle] = Field(max_length=MAX_CANDLES)


class ScreenRequest(CamelModel):
    items: List[ScreenRequestItem] = Field(max_length=MAX_SCREEN_ITEMS)
    top_n: int = Field(default=5, ge=1, le=100)
    min_score: float = 0.0


class Signal(CamelModel):
    name: str
    value: str
    tone: Tone


class PredictResult(CamelModel):
    symbol: str
    name: str
    currency: Currency
    current_price: float
    predicted_price: float
    predicted_change_rate: float
    direction: Direction
    confidence: float
    summary: str
    signals: List[Signal]
    indicators: Dict[str, float]
    as_of: str


class ScreenItem(CamelModel):
    symbol: str
    name: str
    currency: Currency
    current_price: float
    score: float
    predicted_change_rate: float
    direction: Direction
    confidence: float
    reasons: List[str]


class ScreenResponse(CamelModel):
    ranked: List[ScreenItem]


class HealthResponse(CamelModel):
    status: str = "UP"


# ───────────────────────── 야간 배치 ─────────────────────────

Market = Literal["KRX", "US"]
MAX_BATCH_SYMBOLS = 20000


class BatchSymbol(CamelModel):
    symbol: str = Field(min_length=1, max_length=20)
    name: Optional[str] = None
    market: Optional[str] = None  # 거래소 구분(NASDAQ/NYSE/AMEX 등). 결과에 그대로 되돌려 준다.


class BatchStartRequest(CamelModel):
    market: Market
    symbols: List[BatchSymbol] = Field(min_length=1, max_length=MAX_BATCH_SYMBOLS)
    top_liquid: int = Field(default=1000, ge=10, le=5000)
    min_trade_value: Optional[float] = Field(default=None, ge=0)
    min_price: Optional[float] = Field(default=None, ge=0)


class BatchStatus(CamelModel):
    job_id: str
    market: Market
    status: Literal["RUNNING", "DONE", "FAILED"]
    phase: str
    done: int
    total: int
    count: int = 0
    trade_date: Optional[str] = None
    error: Optional[str] = None


class BatchItem(CamelModel):
    symbol: str
    name: str
    market: Optional[str] = None
    currency: Currency
    trade_date: str
    current_price: float
    score: float
    predicted_change_rate: float
    direction: Direction
    confidence: float
    reasons: List[str]
    avg_trade_value: float


class BatchResult(CamelModel):
    status: BatchStatus
    items: List[BatchItem]