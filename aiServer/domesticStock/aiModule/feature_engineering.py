import pandas as pd
import numpy as np

def EMA(series, span):
    return series.ewm(span=span, adjust=False).mean()

def SMA(series, length):
    return series.rolling(window=length).mean()

def RSI(series, length=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    hist = macd_line - signal_line

    return macd_line, signal_line, hist

def BollingerBands(series, length=20, std=2):
    ma = SMA(series, length)
    std_dev = series.rolling(length).std()

    upper = ma + std_dev * std
    lower = ma - std_dev * std
    return upper, lower, ma

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"]

    # 이동평균선
    df["ma20"] = SMA(close, 20)
    df["ma60"] = SMA(close, 60)

    # RSI
    df["rsi"] = RSI(close, 14)

    # MACD
    df["macd"], df["signal"], df["hist"] = MACD(close)

    # 볼린저 밴드
    df["bb_upper"], df["bb_lower"], df["bb_mid"] = BollingerBands(close, 20, 2)

    df = df.dropna()
    return df