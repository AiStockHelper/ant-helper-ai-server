"""캔들 정제와 기술지표 계산. 순수 함수(입력: pandas DataFrame)."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

MIN_CANDLES = 30
OHLCV = ["open", "high", "low", "close", "volume"]


class InsufficientDataError(ValueError):
    """캔들이 부족하거나 사용할 수 없을 때 (HTTP 422 로 변환)."""


def prepare_frame(candles: Iterable[object], min_candles: int = MIN_CANDLES) -> pd.DataFrame:
    """캔들을 날짜 오름차순 DataFrame 으로 정제한다.

    - 날짜 파싱 실패 행은 오류, 같은 날짜가 중복되면 마지막 값을 사용
    - 종가가 0 이하/비정상인 행 제거, open/high/low 가 비정상이면 종가로 보정
    """
    rows = [
        {
            "date": getattr(c, "date", None),
            "open": getattr(c, "open", np.nan),
            "high": getattr(c, "high", np.nan),
            "low": getattr(c, "low", np.nan),
            "close": getattr(c, "close", np.nan),
            "volume": getattr(c, "volume", 0.0),
        }
        for c in candles
    ]
    if not rows:
        raise InsufficientDataError(f"캔들 데이터가 없습니다 (최소 {min_candles}개 필요)")

    df = pd.DataFrame(rows)
    parsed = pd.to_datetime(df["date"], errors="coerce")
    if parsed.isna().any():
        bad = df.loc[parsed.isna(), "date"].iloc[0]
        raise InsufficientDataError(f"날짜 형식이 올바르지 않습니다: {bad!r} (YYYY-MM-DD)")
    df["date"] = parsed.dt.tz_localize(None).dt.normalize()

    for col in OHLCV:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["close"] > 0]
    for col in ("open", "high", "low"):
        df[col] = df[col].where(df[col] > 0, df["close"])
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)

    df = df.sort_values("date", kind="stable").drop_duplicates("date", keep="last")
    df = df.reset_index(drop=True)

    if len(df) < min_candles:
        raise InsufficientDataError(
            f"캔들 데이터가 부족합니다 (유효 {len(df)}개, 최소 {min_candles}개 필요)"
        )
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기술지표 컬럼을 추가한 복사본을 반환한다. 기간이 모자란 구간은 NaN."""
    df = df.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    for window in (5, 10, 20, 60, 120):
        df[f"ma{window}"] = close.rolling(window, min_periods=window).mean()
    df["ema12"] = close.ewm(span=12, adjust=False).mean()
    df["ema26"] = close.ewm(span=26, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    df["rsi14"] = rsi

    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    std20 = close.rolling(20, min_periods=20).std()
    df["bb_mid"] = df["ma20"]
    df["bb_upper"] = df["bb_mid"] + 2 * std20
    df["bb_lower"] = df["bb_mid"] - 2 * std20
    band = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"] = band / df["bb_mid"].replace(0, np.nan)
    df["bb_pct_b"] = (close - df["bb_lower"]) / band

    lowest = low.rolling(14, min_periods=14).min()
    highest = high.rolling(14, min_periods=14).max()
    df["stoch_k"] = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3, min_periods=3).mean()

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    df["vol_ma5"] = volume.rolling(5, min_periods=5).mean()
    df["volume_ratio"] = volume / df["vol_ma5"].replace(0, np.nan)

    df["change_pct"] = close.pct_change() * 100
    df["ret_5d"] = close.pct_change(5) * 100
    df["ret_20d"] = close.pct_change(20) * 100
    df["volatility"] = df["change_pct"].rolling(20, min_periods=10).std()

    return df.replace([np.inf, -np.inf], np.nan)
