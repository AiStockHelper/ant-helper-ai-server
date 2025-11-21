from typing import List

import pandas as pd
from pykrx import stock

from .aiModule.dataset_builder import build_dataset
from .aiModule.data_loader import load_ohlcv_1year
from .aiModule.feature_engineering import add_indicators
from .aiModule.inference import predict_tomorrow
from .aiModule.model_utils import save_model
from .aiModule.train_model import train_model
from .models import DomesticStock

# 국내주식 모두 저장
def save_all_domestic_stocks() -> int:
    """
    오늘 기준 전체 국내 주식 목록으로 DB를 동기화.
    새 종목은 추가하고, 상폐된 종목은 삭제.
    """
    tickers: List[str] = stock.get_market_ticker_list()
    codes_in_market = set(tickers)
    codes_in_db = set(DomesticStock.objects.values_list("product_number", flat=True))

    # 새로 생긴 종목 추가
    new_codes = codes_in_market - codes_in_db
    new_stocks = [
        DomesticStock(product_number=code, name=stock.get_market_ticker_name(code))
        for code in new_codes
    ]
    DomesticStock.objects.bulk_create(new_stocks)

    # 상폐된 종목 삭제
    del_codes = codes_in_db - codes_in_market
    DomesticStock.objects.filter(product_number__in=del_codes).delete()

    return len(new_stocks)

# 내일 주가 예측 모델 생성
def create_tomorrow_price_prediction_model() -> None:
    # domestic_stocks에서 모든 주식코드 조회
    product_numbers = DomesticStock.objects.values_list('product_number', flat=True)

    # DataFrame에 정보 저장
    dfs = []
    for t in product_numbers:
        df = load_ohlcv_1year(t)
        df = df.rename(columns={
            "시가": "Open",
            "고가": "High",
            "저가": "Low",
            "종가": "Close",
            "거래량": "Volume",
            "등락률": "Change"
        })
        df = add_indicators(df)
        dfs.append(df)

    print(dfs)

    # null 제외
    df_all = pd.concat(dfs).dropna()

    # 모델 학습
    X, y, scaler = build_dataset(df_all)
    model = train_model(X, y)

    print("모델 학습 완료")

    # 모델 저장
    save_model(model, scaler)

# 주가 예측
def predict_tomorrow_price() -> None:
    print("삼성전자 내일 예상 종가:", predict_tomorrow("005930"))