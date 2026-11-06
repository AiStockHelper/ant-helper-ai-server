"""추천 순위용 요인 점수.

종목마다 일봉에서 뽑은 '사실'(facts: 수익률·변동성·RSI·이평선 이격 등)을 유동성 상위 종목끼리 비교해 점수를 매긴다.

왜 이 방식인가 (tools/factor_ic.py, tools/backtest_factors.py 로 다시 확인할 수 있다):
  2026-04~10 한국·미국 유동성 상위 300종목, 날짜마다 요인 순위와 이후 1·5거래일 수익률의 순위상관(IC)을 보면
  - 이전 방식이 높게 치던 추세·모멘텀(정배열, 20일/60일 수익, 20일선 위)은 오히려 **음(-)의 상관**이었다.
  - 변동성이 낮은 종목(IC 5일 한국 +0.14, 미국 +0.08), 최근 덜 오르거나 눌린 종목(RSI 낮음, 5일 수익 낮음)이 일관되게 유리했다.
  - 종목별 단기 머신러닝의 방향 적중률은 약 50%(동전 던지기 수준)라 순위에는 쓰지 않고 참고로만 보여 준다.
기대치: 상위 10종목이 같은 날 전체 평균보다 5거래일 기준 한국 약 +1.0%p, 미국 약 +0.6%p 나았지만, 검증 구간만 보면 차이가 훨씬 작았다.
즉 '확실히 오를 종목'을 찾는 도구가 아니라 '불리한 종목을 피하고 상대적으로 유리한 쪽을 고르는' 보조 지표다.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# 점수를 매기고 추천 대상으로 삼는 유동성 상위 종목 수(시장별). 그 밖은 점수를 0 이하로 눌러 추천에서 뺀다.
RANK_UNIVERSE = 300

W_LOW_VOL = 0.45
W_OVERSOLD = 0.20
W_SHORT_DIP = 0.15
W_BELOW_MA20 = 0.10
W_MID_REVERSAL = 0.10
PENALTY_KNIFE = 1.0
PENALTY_CHASE = 0.5
SCORE_SCALE = 50.0


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def compute_facts(ind: pd.DataFrame) -> Dict[str, float]:
    """add_indicators 를 거친 프레임의 마지막 날 기준 사실들. 계산 못 하는 항목은 빠진다."""
    last = ind.iloc[-1]
    close = ind["close"]
    n = len(ind)
    price = float(last["close"])
    facts: Dict[str, Optional[float]] = {
        "ret1": _num(last.get("change_pct")),
        "ret5": _num(last.get("ret_5d")),
        "ret20": _num(last.get("ret_20d")),
        "vol20": _num(last.get("volatility")),
        "rsi": _num(last.get("rsi14")),
        "volRatio": _num(last.get("volume_ratio")),
    }
    if n >= 66:
        facts["mom60"] = (float(close.iloc[-6]) / float(close.iloc[-66]) - 1) * 100
    if n >= 40:
        facts["distHigh60"] = (price / float(close.tail(60).max()) - 1) * 100
    for key, col in (("pxMa20", "ma20"), ("pxMa60", "ma60")):
        ma = _num(last.get(col))
        if ma:
            facts[key] = (price / ma - 1) * 100
    ma5, ma20, ma60 = _num(last.get("ma5")), _num(last.get("ma20")), _num(last.get("ma60"))
    if ma5 and ma20 and ma60:
        facts["align"] = 1.0 if (ma5 > ma20 > ma60 and price > ma5) else 0.0
    return {k: round(v, 4) for k, v in facts.items() if v is not None and math.isfinite(v)}


def _centered_rank(values: pd.Series, cohort: pd.Series) -> pd.Series:
    """cohort 안에서의 백분위를 -1~1 로 바꾼다(0 = 중간). 값이 없으면 0."""
    ref = np.sort(cohort.dropna().to_numpy(dtype=float))
    out = pd.Series(0.0, index=values.index)
    if len(ref) == 0:
        return out
    ok = values.notna()
    x = values[ok].to_numpy(dtype=float)
    lo = np.searchsorted(ref, x, side="left")
    hi = np.searchsorted(ref, x, side="right")
    pct = (lo + hi) / 2.0 / len(ref)
    out[ok] = (pct - 0.5) * 2.0
    return out


def apply_ranking(items: List[Dict[str, Any]], universe_n: int = RANK_UNIVERSE) -> None:
    """items(각각 'facts' 와 선택적으로 'avgTradeValue', 모델 참고값)에 score·reasons 를 채워 넣는다."""
    if not items:
        return
    frame = pd.DataFrame([{**(it.get("facts") or {}), "_tv": it.get("avgTradeValue")} for it in items])
    for col in ("vol20", "rsi", "ret5", "ret20", "ret1", "pxMa20", "mom60"):
        if col not in frame:
            frame[col] = np.nan
    if frame["_tv"].notna().any():
        order = frame["_tv"].fillna(0).sort_values(ascending=False)
        in_pool = frame.index.isin(order.index[: max(1, universe_n)])
    else:
        in_pool = np.ones(len(frame), dtype=bool)
    pool = frame[in_pool]

    def z(col: str) -> pd.Series:
        return _centered_rank(frame[col], pool[col])

    score = (W_LOW_VOL * -z("vol20") + W_OVERSOLD * -z("rsi") + W_SHORT_DIP * -z("ret5")
             + W_BELOW_MA20 * -z("pxMa20") + W_MID_REVERSAL * -z("mom60"))
    knife = (frame["ret20"] < -25) | (frame["ret5"] < -15)
    chase = (frame["ret1"] > 12) | (frame["ret5"] > 20)
    score = score - knife.astype(float) * PENALTY_KNIFE - chase.astype(float) * PENALTY_CHASE
    vol_rank = z("vol20")

    for i, it in enumerate(items):
        s = float(score.iloc[i]) * SCORE_SCALE
        pooled = bool(in_pool[i]) and not it.get("illiquid")
        if not pooled:
            s = min(s, 0.0)
        it["score"] = round(max(-100.0, min(100.0, s)), 1)
        it["reasons"] = build_reasons(it, float(vol_rank.iloc[i]), bool(knife.iloc[i]), bool(chase.iloc[i]), pooled)


def build_reasons(it: Dict[str, Any], vol_z: float, knife: bool, chase: bool, pooled: bool) -> List[str]:
    f = it.get("facts") or {}
    reasons: List[str] = []
    vol, rsi, ret5, ret20, ma20 = f.get("vol20"), f.get("rsi"), f.get("ret5"), f.get("ret20"), f.get("pxMa20")
    if vol is not None and vol_z <= -0.3:
        reasons.append(f"20일 변동성 {vol:.1f}% — 비슷한 규모 종목 중 안정적인 편(하위 {max(1, round((vol_z + 1) / 2 * 100))}%)")
    elif vol is not None and vol_z >= 0.5:
        reasons.append(f"20일 변동성 {vol:.1f}% — 등락이 큰 편이라 점수 불리")
    if rsi is not None and rsi <= 40:
        reasons.append(f"RSI {rsi:.0f} — 과매도 쪽이라 반등 여지")
    elif rsi is not None and rsi >= 70:
        reasons.append(f"RSI {rsi:.0f} — 과열 쪽이라 점수 불리")
    if ret5 is not None and ret5 <= -3:
        reasons.append(f"최근 5일 {ret5:+.1f}% — 단기 눌림")
    elif ret5 is not None and ret5 >= 8:
        reasons.append(f"최근 5일 {ret5:+.1f}% — 이미 오른 상태")
    if ma20 is not None and ma20 < -2:
        reasons.append(f"20일선보다 {abs(ma20):.1f}% 아래 — 되돌림 여지")
    if knife:
        reasons.append(f"최근 급락(20일 {ret20:+.1f}%, 5일 {ret5:+.1f}%) — 떨어지는 칼날일 수 있어 감점" if ret20 is not None and ret5 is not None
                       else "최근 급락 — 떨어지는 칼날일 수 있어 감점")
    if chase:
        reasons.append("당일·최근 급등 — 추격 매수 위험으로 감점")
    if it.get("illiquid"):
        reasons.append("최근 거래대금이 너무 작아 유동성이 부족해 추천 대상에서 제외")
    elif not pooled:
        reasons.append("거래대금이 상위권이 아니라 추천 대상에서 제외")
    direction = it.get("direction")
    if direction:
        label = {"UP": "상승", "DOWN": "하락", "FLAT": "보합"}.get(direction, direction)
        rate = it.get("predictedChangeRate")
        conf = it.get("confidence")
        extra = f" {rate:+.2f}%" if isinstance(rate, (int, float)) else ""
        extra += f", 신뢰도 {conf:.0f}" if isinstance(conf, (int, float)) else ""
        reasons.append(f"참고: 단기 모델은 {label}{extra} 예상 — 방향 적중률이 낮아 순위에는 반영하지 않음")
    if len(reasons) == (1 if direction else 0):
        reasons.insert(0, "뚜렷한 장점이나 단점 없이 평균적인 편")
    return reasons
