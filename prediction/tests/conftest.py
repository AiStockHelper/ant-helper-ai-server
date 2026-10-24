from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


def make_candles(n=120, seed=7, drift=0.0, vol=0.015, start_price=50000.0, base_volume=1_000_000,
                 decimals=0):
    """랜덤시드 고정 합성 캔들 (오래된 순)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    closes = start_price * np.cumprod(1 + rets)
    start = date(2026, 1, 1)
    candles = []
    prev = start_price
    for i in range(n):
        close = float(round(closes[i], decimals))
        open_ = float(round(prev * (1 + rng.normal(0, 0.003)), decimals))
        high = float(max(open_, close) * (1 + abs(rng.normal(0, 0.004))))
        low = float(min(open_, close) * (1 - abs(rng.normal(0, 0.004))))
        volume = float(int(base_volume * (1 + rng.normal(0, 0.3)) ** 2 + 1000))
        candles.append(
            {
                "date": (start + timedelta(days=i)).isoformat(),
                "open": open_,
                "high": round(high, decimals),
                "low": round(low, decimals),
                "close": close,
                "volume": volume,
            }
        )
        prev = close
    return candles


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture
def candles():
    return make_candles()
