"""종목 스코어링. 예측 결과(지표 포함)로 점수를 매기고 순위를 만든다."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from .currency import DEFAULT_CURRENCY, MIN_TRADE_VALUE
from .indicators import InsufficientDataError, prepare_frame
from .predictor import predict_frame
from .schemas import PredictResult, ScreenItem

logger = logging.getLogger(__name__)

W_SIGNAL = 1.0
W_TREND = 1.0
W_MOMENTUM = 0.7
W_VOLUME = 0.6


def score_prediction(pred: PredictResult) -> Tuple[float, List[str]]:
    """예측 결과를 -100~100 점수와 한국어 근거 목록으로 변환한다."""
    ind = pred.indicators
    price = float(ind.get("close", pred.current_price))
    score = 0.0
    reasons: List[str] = []
    conf = pred.confidence

    if pred.direction == "UP":
        score += conf * W_SIGNAL
        reasons.append(f"내일 상승 예상 (+{abs(pred.predicted_change_rate):.2f}%, 신뢰도 {conf:.0f}%)")
    elif pred.direction == "DOWN":
        score -= conf * W_SIGNAL
        reasons.append(f"내일 하락 예상 (-{abs(pred.predicted_change_rate):.2f}%, 신뢰도 {conf:.0f}%)")
    else:
        reasons.append(f"내일 보합 예상 (신뢰도 {conf:.0f}%)")

    ma5, ma20, ma60 = ind.get("ma5", 0.0), ind.get("ma20", 0.0), ind.get("ma60", 0.0)
    if ma5 > ma20 > ma60 > 0 and price > ma5:
        score += 10 * W_TREND
        reasons.append("5일선 > 20일선 > 60일선 정배열 — 강한 상승 추세")
    elif price > ma20 > ma60 > 0:
        score += 5 * W_TREND
        reasons.append("가격이 20일선·60일선 위 — 중기 상승 추세")
    elif ma20 > 0 and price < ma20 and ma5 < ma20:
        score -= 6 * W_TREND
        reasons.append("5일선·가격이 20일선 아래 — 하락 추세")

    ret5, ret20 = ind.get("ret5d", 0.0), ind.get("ret20d", 0.0)
    if 0 < ret20 <= 25:
        score += min(8.0, ret20 * 0.5) * W_MOMENTUM
        reasons.append(f"20일 수익률 {ret20:+.1f}% — 건강한 상승")
    elif ret20 > 35:
        score -= 4 * W_MOMENTUM
        reasons.append(f"20일 수익률 {ret20:+.1f}% — 과열 주의")
    elif ret20 < -15:
        score -= 6 * W_MOMENTUM
        reasons.append(f"20일 수익률 {ret20:+.1f}% — 약세")
    if ret5 < -8:
        score -= 3 * W_MOMENTUM
        reasons.append(f"5일 급락 {ret5:+.1f}% — 단기 변동성 큼")

    vol_ratio = ind.get("volumeRatio", 1.0)
    if vol_ratio > 1.8:
        score += 5 * W_VOLUME
        reasons.append(f"거래량 급증 (5일 평균의 {vol_ratio:.1f}배)")
    elif vol_ratio > 1.3:
        score += 2.5 * W_VOLUME
        reasons.append(f"거래량 증가 (5일 평균의 {vol_ratio:.1f}배)")

    bb_pct = ind.get("bbPercentB")
    if bb_pct is not None:
        if 0 <= bb_pct <= 0.2:
            score += 4 * W_TREND
            reasons.append("볼린저밴드 하단 — 저가 매수 구간")
        elif bb_pct >= 0.9:
            score -= 4 * W_TREND
            reasons.append("볼린저밴드 상단 — 고점 근접")

    rsi = ind.get("rsi14")
    if rsi is not None:
        if rsi < 30:
            score += 4 * W_TREND
            reasons.append(f"RSI {rsi:.0f} — 과매도 구간")
        elif 40 <= rsi <= 65:
            score += 3 * W_TREND
            reasons.append(f"RSI {rsi:.0f} — 안정적인 구간")
        elif rsi > 78:
            score -= 5 * W_TREND
            reasons.append(f"RSI {rsi:.0f} — 과열 구간")

    min_trade_value = MIN_TRADE_VALUE.get(pred.currency, MIN_TRADE_VALUE[DEFAULT_CURRENCY])
    if ind.get("tradeValue5d", min_trade_value) < min_trade_value:
        score -= 5
        reasons.append("최근 거래대금이 적어 유동성이 낮음")

    return round(max(-100.0, min(100.0, score)), 1), reasons


def screen_items(
    items: Sequence[object],
    top_n: int = 5,
    min_score: float = 0.0,
) -> List[ScreenItem]:
    """각 항목(symbol, name, candles)을 스코어링해 score 내림차순 상위 top_n 을 반환한다.

    캔들이 부족한 항목은 건너뛴다. 점수가 같으면 입력 순서를 유지한다.
    """
    ranked: List[ScreenItem] = []
    for item in items:
        symbol = getattr(item, "symbol")
        try:
            df: pd.DataFrame = prepare_frame(getattr(item, "candles"))
            pred = predict_frame(
                df, symbol, getattr(item, "name", None), getattr(item, "currency", DEFAULT_CURRENCY)
            )
        except InsufficientDataError as exc:
            logger.info("스크리닝 제외 %s: %s", symbol, exc)
            continue
        except Exception as exc:
            logger.warning("스크리닝 실패 %s: %s", symbol, exc)
            continue

        score, reasons = score_prediction(pred)
        if score < min_score:
            continue
        ranked.append(
            ScreenItem(
                symbol=pred.symbol,
                name=pred.name,
                currency=pred.currency,
                current_price=pred.current_price,
                score=score,
                predicted_change_rate=pred.predicted_change_rate,
                direction=pred.direction,
                confidence=pred.confidence,
                reasons=reasons,
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_n]
