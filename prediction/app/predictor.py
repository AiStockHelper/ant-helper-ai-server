"""다음 거래일 예측: 규칙 기반(기술지표) + 단순 ML(GradientBoosting) 앙상블.

입력은 정제된 OHLCV DataFrame 뿐이며 외부 시스템에 의존하지 않는다.
모든 모델은 random_state 를 고정해 같은 입력이면 같은 결과를 낸다.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .currency import DEFAULT_CURRENCY, format_price, round_price
from .indicators import add_indicators
from .schemas import PredictResult, Signal

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error

    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

KST = timezone(timedelta(hours=9))

RULE_SIGNAL_THRESHOLD = 2.2
ML_MIN_USABLE_ROWS = 60
ML_MIN_VALID_ROWS = 12
ML_MAX_RETURN_PCT = 5.0
ML_SHRINK = 0.5
FLAT_MIN_PCT = 0.1
MAX_CONFIDENCE = 85.0

ML_FEATURES = [
    "px_ma5", "px_ma20", "ma5_ma20", "rsi14", "macd_pct", "macd_sig_pct",
    "macd_hist_pct", "bb_pct_b", "bb_width", "stoch_k", "stoch_d", "atr_pct",
    "change_pct", "ret_5d", "ret_20d", "volume_ratio", "volatility",
]


@dataclass
class MLForecast:
    return_pct: float
    direction_accuracy: float
    mae_pct: float
    weight: float
    train_rows: int


@dataclass
class RuleResult:
    score: float = 0.0
    signals: List[Signal] = field(default_factory=list)


def _num(row: pd.Series, key: str) -> Optional[float]:
    v = row.get(key)
    if v is None or pd.isna(v):
        return None
    return float(v)


def _sig(rule: RuleResult, delta: float, name: str, value: str, tone: str) -> None:
    rule.score += delta
    rule.signals.append(Signal(name=name, value=value, tone=tone))


def evaluate_rules(df: pd.DataFrame) -> RuleResult:
    """지표 DataFrame 의 마지막 봉을 규칙으로 평가한다 (양수 = 상승 우위)."""
    last, prev = df.iloc[-1], df.iloc[-2]
    price = float(last["close"])
    ma5, ma20, ma60 = _num(last, "ma5"), _num(last, "ma20"), _num(last, "ma60")
    rsi = _num(last, "rsi14")
    macd, macd_sig = _num(last, "macd"), _num(last, "macd_signal")
    prev_macd, prev_macd_sig = _num(prev, "macd"), _num(prev, "macd_signal")
    stoch_k, stoch_d = _num(last, "stoch_k"), _num(last, "stoch_d")
    bb_upper, bb_lower = _num(last, "bb_upper"), _num(last, "bb_lower")
    vol_ratio = _num(last, "volume_ratio") or 1.0
    ret5, ret20 = _num(last, "ret_5d") or 0.0, _num(last, "ret_20d") or 0.0

    rule = RuleResult()

    # 추세
    if ma5 and ma20:
        if ma60 and ma5 > ma20 > ma60 and price > ma5:
            _sig(rule, 1.6, "이동평균", "5일선 > 20일선 > 60일선 — 완전한 정배열, 강한 상승 추세", "POSITIVE")
        elif ma5 > ma20 and price > ma20:
            _sig(rule, 0.8, "이동평균", "5일선이 20일선 위 — 상승 추세", "POSITIVE")
        elif ma5 < ma20 and price < ma20:
            _sig(rule, -1.5, "이동평균", "5일선이 20일선 아래, 가격도 20일선 아래 — 하락 추세", "NEGATIVE")
        else:
            _sig(rule, 0.0, "이동평균", "5일선과 20일선이 엇갈림 — 방향성 불분명", "NEUTRAL")

        prev_close, prev_ma20 = float(prev["close"]), _num(prev, "ma20")
        if prev_ma20:
            if prev_close <= prev_ma20 and price > ma20:
                _sig(rule, 1.4, "20일선 돌파", "가격이 20일선을 상향 돌파", "POSITIVE")
            elif prev_close >= prev_ma20 and price < ma20:
                _sig(rule, -1.0, "20일선 이탈", "가격이 20일선을 하향 이탈", "NEGATIVE")

    # MACD
    if None not in (macd, macd_sig, prev_macd, prev_macd_sig):
        if macd > macd_sig and prev_macd <= prev_macd_sig:
            _sig(rule, 1.3, "MACD", "골든크로스 발생 — 상승 전환 신호", "POSITIVE")
        elif macd < macd_sig and prev_macd >= prev_macd_sig:
            _sig(rule, -1.3, "MACD", "데드크로스 발생 — 하락 전환 신호", "NEGATIVE")
        elif macd > macd_sig:
            _sig(rule, 0.4, "MACD", "시그널선 위 — 상승 모멘텀 유지", "POSITIVE")
        else:
            _sig(rule, -0.4, "MACD", "시그널선 아래 — 상승 모멘텀 약함", "NEGATIVE")

    # RSI
    if rsi is not None:
        r = f"{rsi:.0f}"
        if rsi < 32:
            _sig(rule, 1.0, "RSI", f"{r} — 과매도 구간, 반등 기대", "POSITIVE")
        elif rsi < 45:
            _sig(rule, 0.0, "RSI", f"{r} — 다소 약한 구간", "NEUTRAL")
        elif rsi <= 68:
            _sig(rule, 0.6, "RSI", f"{r} — 안정적인 상승 구간", "POSITIVE")
        elif rsi <= 78:
            _sig(rule, -0.2, "RSI", f"{r} — 과열 구간 접근", "NEUTRAL")
        else:
            _sig(rule, -0.5, "RSI", f"{r} — 과열 구간, 조정 가능성", "NEGATIVE")

    # 스토캐스틱
    if stoch_k is not None and stoch_d is not None:
        if stoch_k < 25 and stoch_k > stoch_d:
            _sig(rule, 0.8, "스토캐스틱", f"{stoch_k:.0f} — 저점에서 반등 시작", "POSITIVE")
        elif stoch_k > 80 and stoch_k < stoch_d:
            _sig(rule, -0.8, "스토캐스틱", f"{stoch_k:.0f} — 고점에서 꺾임", "NEGATIVE")

    # 볼린저밴드
    if bb_upper and bb_lower:
        if price <= bb_lower * 1.03:
            _sig(rule, 0.9, "볼린저밴드", "하단 부근 — 저가 매수세 기대", "POSITIVE")
        elif price >= bb_upper * 0.98:
            _sig(rule, -0.8, "볼린저밴드", "상단 부근 — 단기 고점 경계", "NEGATIVE")
        else:
            _sig(rule, 0.0, "볼린저밴드", "밴드 중간권 — 중립", "NEUTRAL")

    # 거래량
    if vol_ratio > 1.6:
        if ret5 > 0:
            _sig(rule, 0.8, "거래량", f"5일 평균의 {vol_ratio:.1f}배 급증, 5일 상승 — 매수세 유입", "POSITIVE")
        elif ret5 < 0:
            _sig(rule, -0.8, "거래량", f"5일 평균의 {vol_ratio:.1f}배 급증, 5일 하락 — 매도 압력", "NEGATIVE")

    # 과열 / 약세
    if ret5 > 8 and ret20 > 20:
        _sig(rule, -0.6, "단기 과열", f"5일 {ret5:+.1f}%, 20일 {ret20:+.1f}% — 급등 후 조정 주의", "NEGATIVE")
    if ret20 < -12:
        _sig(rule, -1.0, "중기 약세", f"20일 {ret20:+.1f}% — 하락 추세", "NEGATIVE")

    if not rule.signals:
        rule.signals.append(Signal(name="종합", value="뚜렷한 신호 없음", tone="NEUTRAL"))
    return rule


def _rule_confidence(score: float) -> float:
    if score >= RULE_SIGNAL_THRESHOLD or score <= -RULE_SIGNAL_THRESHOLD:
        return min(80.0, 50.0 + abs(score) * 7.0)
    return max(30.0, 55.0 - abs(score) * 10.0)


def _daily_volatility_pct(last: pd.Series) -> float:
    vol = _num(last, "volatility")
    if vol is None:
        atr, close = _num(last, "atr14"), float(last["close"])
        vol = atr / close * 100 if atr and close > 0 else 1.5
    return min(5.0, max(0.5, vol))


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    f["px_ma5"] = close / df["ma5"] - 1
    f["px_ma20"] = close / df["ma20"] - 1
    f["ma5_ma20"] = df["ma5"] / df["ma20"] - 1
    f["rsi14"] = df["rsi14"]
    f["macd_pct"] = df["macd"] / close
    f["macd_sig_pct"] = df["macd_signal"] / close
    f["macd_hist_pct"] = df["macd_hist"] / close
    f["bb_pct_b"] = df["bb_pct_b"].fillna(0.5)
    f["bb_width"] = df["bb_width"]
    f["stoch_k"] = df["stoch_k"].fillna(50.0)
    f["stoch_d"] = df["stoch_d"].fillna(50.0)
    f["atr_pct"] = df["atr14"] / close
    f["change_pct"] = df["change_pct"]
    f["ret_5d"] = df["ret_5d"]
    f["ret_20d"] = df["ret_20d"]
    f["volume_ratio"] = df["volume_ratio"].fillna(1.0)
    f["volatility"] = df["volatility"]
    # 반올림으로 가격 단위/부동소수 오차에 따른 트리 분할 타이 차이를 없앤다
    return f[ML_FEATURES].replace([np.inf, -np.inf], np.nan).round(8)


def ml_forecast(df: pd.DataFrame) -> Optional[MLForecast]:
    """다음 거래일 수익률을 GradientBoosting 으로 추정. 데이터가 부족하면 None."""
    if not _SKLEARN_AVAILABLE:
        return None

    feats = _build_features(df)
    target = (df["close"].shift(-1) / df["close"] - 1).round(10)
    last_x = feats.iloc[[-1]]
    if last_x.isna().any(axis=None):
        return None

    mask = feats.notna().all(axis=1) & target.notna()
    X, y = feats[mask].to_numpy(), target[mask].to_numpy()
    if len(X) < ML_MIN_USABLE_ROWS:
        return None
    split = int(len(X) * 0.8)
    if len(X) - split < ML_MIN_VALID_ROWS:
        return None

    def new_model() -> "GradientBoostingRegressor":
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42
        )

    validator = new_model().fit(X[:split], y[:split])
    pred_val = validator.predict(X[split:])
    mae = float(mean_absolute_error(y[split:], pred_val))
    direction_accuracy = float(np.mean(np.sign(pred_val) == np.sign(y[split:])))

    model = new_model().fit(X, y)
    raw_pct = float(model.predict(last_x.to_numpy())[0]) * 100
    if not math.isfinite(raw_pct):
        return None
    ret_pct = max(-ML_MAX_RETURN_PCT, min(ML_MAX_RETURN_PCT, raw_pct * ML_SHRINK))

    if direction_accuracy >= 0.56:
        weight = 0.4
    elif direction_accuracy >= 0.5:
        weight = 0.25
    else:
        weight = 0.1
    return MLForecast(ret_pct, direction_accuracy, mae * 100, weight, len(X))


def _sign(x: float, threshold: float) -> int:
    return 1 if x >= threshold else (-1 if x <= -threshold else 0)


def _fmt_indicators(last: pd.Series, n: int, rule_score: float, ml: Optional[MLForecast]) -> Dict[str, float]:
    close = float(last["close"])
    atr = _num(last, "atr14")
    raw = {
        "close": close,
        "ma5": _num(last, "ma5"),
        "ma10": _num(last, "ma10"),
        "ma20": _num(last, "ma20"),
        "ma60": _num(last, "ma60"),
        "rsi14": _num(last, "rsi14"),
        "macd": _num(last, "macd"),
        "macdSignal": _num(last, "macd_signal"),
        "bbUpper": _num(last, "bb_upper"),
        "bbLower": _num(last, "bb_lower"),
        "bbPercentB": _num(last, "bb_pct_b"),
        "stochK": _num(last, "stoch_k"),
        "atr14": atr,
        "atrPct": atr / close * 100 if atr and close > 0 else None,
        "volumeRatio": _num(last, "volume_ratio"),
        "ret5d": _num(last, "ret_5d"),
        "ret20d": _num(last, "ret_20d"),
        "volatility": _num(last, "volatility"),
        "historyDays": float(n),
        "ruleScore": rule_score,
        "mlWeight": ml.weight if ml else 0.0,
    }
    return {k: round(float(v), 4) for k, v in raw.items() if v is not None and math.isfinite(float(v))}


def _summary(display: str, direction: str, price: float, predicted: float, change: float,
             confidence: float, signals: List[Signal], currency: str) -> str:
    dir_text = {"UP": "상승", "DOWN": "하락", "FLAT": "보합"}[direction]
    want = {"UP": "POSITIVE", "DOWN": "NEGATIVE"}.get(direction)
    picked = [s for s in signals if want and s.tone == want][:2]
    if not picked:
        picked = [s for s in signals if s.tone != "NEUTRAL"][:2] or signals[:2]
    basis = "; ".join(f"{s.name} {s.value}" for s in picked)
    return (
        f"{display} 내일 {dir_text} 예상 — 현재가 {format_price(price, currency)} → "
        f"예상 {format_price(predicted, currency)} "
        f"({change:+.2f}%), 신뢰도 {confidence:.0f}%. 주요 근거: {basis}. "
        "기술지표와 단순 ML 기반의 참고용 추정이며 실제 결과와 다를 수 있습니다."
    )


def predict_frame(
    df: pd.DataFrame, symbol: str, name: Optional[str] = None, currency: str = DEFAULT_CURRENCY
) -> PredictResult:
    """정제된 캔들(날짜 오름차순, 30개 이상)로 다음 거래일 예측을 만든다.

    모든 지표/판정은 비율 기반이며 가격 단위에 의존하지 않는다. currency 는 표기와 반올림에만 쓴다.
    """
    n = len(df)
    ind = add_indicators(df)
    last = ind.iloc[-1]
    price = float(last["close"])

    rule = evaluate_rules(ind)
    vol = _daily_volatility_pct(last)
    rule_ret = math.tanh(rule.score / 4.0) * vol * 0.5
    rule_conf = _rule_confidence(rule.score)

    try:
        ml = ml_forecast(ind)
    except Exception as exc:  # ML 실패 시 기술지표만 사용
        logger.warning("ML 예측 실패, 기술지표만 사용합니다 (%s): %s", symbol, exc)
        ml = None

    flat_threshold = max(FLAT_MIN_PCT, 0.25 * vol)
    signals = list(rule.signals)
    conflict = agree = False

    if ml:
        change = (1 - ml.weight) * rule_ret + ml.weight * ml.return_pct
        ml_conf = min(70.0, max(30.0, 45.0 + (ml.direction_accuracy - 0.5) * 100))
        confidence = rule_conf * (1 - ml.weight) + ml_conf * ml.weight
        rs, ms = _sign(rule_ret, flat_threshold), _sign(ml.return_pct, flat_threshold)
        conflict = rs * ms < 0
        agree = rs == ms != 0
        if conflict:
            confidence *= 0.6
        elif agree:
            confidence *= 1.05
        ml_tone = "POSITIVE" if ml.return_pct >= flat_threshold else (
            "NEGATIVE" if ml.return_pct <= -flat_threshold else "NEUTRAL")
        signals.append(Signal(
            name="머신러닝",
            value=f"내일 {ml.return_pct:+.2f}% 추정 (검증 방향 적중률 {ml.direction_accuracy * 100:.0f}%)",
            tone=ml_tone,
        ))
        if conflict:
            signals.append(Signal(name="신호 충돌", value="기술지표와 ML 방향이 달라 신뢰도를 낮춤", tone="NEUTRAL"))
        cap = MAX_CONFIDENCE - 5 if ml.direction_accuracy >= 0.56 else 72.0
    else:
        change = rule_ret
        confidence = rule_conf
        cap = 70.0
        signals.append(Signal(name="머신러닝", value="학습 데이터가 부족해 기술지표만 사용", tone="NEUTRAL"))

    if n < 60:
        cap = min(cap, 55.0)
    confidence = round(max(20.0, min(cap, MAX_CONFIDENCE, confidence)), 1)

    direction = "UP" if change >= flat_threshold else ("DOWN" if change <= -flat_threshold else "FLAT")
    change = round(change, 2)
    current_price = round_price(price, currency)
    predicted_price = round_price(price * (1 + change / 100), currency)

    indicators = _fmt_indicators(last, n, rule.score, ml)
    indicators["tradeValue5d"] = round(float((df["close"] * df["volume"]).tail(5).mean()), 2)

    display = f"{name}({symbol})" if name else symbol
    return PredictResult(
        symbol=symbol,
        name=name or symbol,
        currency=currency,
        current_price=current_price,
        predicted_price=predicted_price,
        predicted_change_rate=change,
        direction=direction,
        confidence=confidence,
        summary=_summary(display, direction, current_price, predicted_price, change, confidence, signals, currency),
        signals=signals,
        indicators=indicators,
        as_of=datetime.now(KST).isoformat(timespec="seconds"),
    )
