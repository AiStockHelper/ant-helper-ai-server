from typing import List
from pykrx import stock
from .models import DomesticStock

# 국내주식 모두 저장
def save_all_domestic_stocks() -> int:
    """
    오늘 기준 전체 국내 주식 목록을 가져와 DB에 저장.
    이미 존재하는 종목(product_number)은 무시.

    Returns:
        int: 저장 시도한 전체 종목 개수
    """
    # 전체 종목코드 조회
    tickers: List[str] = stock.get_market_ticker_list()

    # DB에 저장할 모델 인스턴스 목록
    domestic_stocks: List[DomesticStock] = [
        DomesticStock(product_number=code, name=stock.get_market_ticker_name(code))
        for code in tickers
    ]

    # 중복 무시하고 한 번에 저장
    DomesticStock.objects.bulk_create(domestic_stocks, ignore_conflicts=True)

    return len(domestic_stocks)