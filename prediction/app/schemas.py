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
