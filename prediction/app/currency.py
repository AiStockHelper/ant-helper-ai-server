"""통화별 상수와 가격 표기. 통화 관련 임계값은 이 파일에만 둔다."""
from __future__ import annotations

from typing import Dict

DEFAULT_CURRENCY = "KRW"

# 가격 반올림 자릿수
PRICE_DECIMALS: Dict[str, int] = {"KRW": 0, "USD": 2}

# 스크리닝 유동성 감점 기준: 최근 5일 평균 거래대금(해당 통화 단위)
MIN_TRADE_VALUE: Dict[str, float] = {"KRW": 100_000_000.0, "USD": 5_000_000.0}


def round_price(value: float, currency: str) -> float:
    return round(float(value), PRICE_DECIMALS.get(currency, 2))


def format_price(value: float, currency: str) -> str:
    if currency == "USD":
        return f"${value:,.2f}"
    return f"{value:,.0f}원"
