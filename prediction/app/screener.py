"""종목 스크리너. 후보 목록을 요인 점수(factors)로 순위 매겨 상위를 돌려준다."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import pandas as pd

from . import factors
from .currency import DEFAULT_CURRENCY, MIN_TRADE_VALUE
from .indicators import InsufficientDataError, add_indicators, prepare_frame
from .predictor import predict_frame
from .schemas import ScreenItem

logger = logging.getLogger(__name__)


def analyze_frame(df: pd.DataFrame, symbol: str, name, currency: str) -> Dict[str, Any]:
    """일봉 한 종목을 분석해 순위 매기기 전 상태의 기록으로 만든다(모델 예측은 참고값으로만 담는다)."""
    pred = predict_frame(df, symbol, name, currency or DEFAULT_CURRENCY)
    facts = factors.compute_facts(add_indicators(df))
    min_value = MIN_TRADE_VALUE.get(pred.currency, MIN_TRADE_VALUE[DEFAULT_CURRENCY])
    return {
        "illiquid": pred.indicators.get("tradeValue5d", min_value) < min_value,
        "symbol": symbol, "name": pred.name, "currency": pred.currency, "currentPrice": pred.current_price,
        "predictedChangeRate": pred.predicted_change_rate, "direction": pred.direction, "confidence": pred.confidence,
        "facts": facts,
    }


def screen_items(
    items: Sequence[object],
    top_n: int = 5,
    min_score: float = 0.0,
) -> List[ScreenItem]:
    """각 항목(symbol, name, candles)을 분석해 후보끼리 비교한 점수 순으로 상위 top_n 을 돌려준다.

    캔들이 부족한 항목은 건너뛴다. 순위는 입력으로 준 후보들 사이의 상대 비교다.
    """
    records: List[Dict[str, Any]] = []
    for item in items:
        symbol = getattr(item, "symbol")
        try:
            df = prepare_frame(getattr(item, "candles"))
            records.append(analyze_frame(df, symbol, getattr(item, "name", None), getattr(item, "currency", DEFAULT_CURRENCY)))
        except InsufficientDataError as exc:
            logger.info("스크리닝 제외 %s: %s", symbol, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("스크리닝 실패 %s: %s", symbol, exc)

    factors.apply_ranking(records, universe_n=max(1, len(records)))
    ranked = [
        ScreenItem(
            symbol=r["symbol"], name=r["name"], currency=r["currency"], current_price=r["currentPrice"], score=r["score"],
            predicted_change_rate=r["predictedChangeRate"], direction=r["direction"], confidence=r["confidence"],
            reasons=r["reasons"],
        )
        for r in records if r["score"] >= min_score
    ]
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_n]