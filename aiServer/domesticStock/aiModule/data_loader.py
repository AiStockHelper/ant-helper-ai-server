from pykrx import stock
import pandas as pd
from datetime import datetime, timedelta

# 특정 주식의 1년간 데이터 조회
def load_ohlcv_1year(ticker: str, days: int = 450) -> pd.DataFrame:
    today = datetime.today()
    start = today - timedelta(days=days)

    df = stock.get_market_ohlcv(
        fromdate=start.strftime("%Y%m%d"),
        todate=today.strftime("%Y%m%d"),
        ticker=ticker
    )

    df.index = pd.to_datetime(df.index)
    return df