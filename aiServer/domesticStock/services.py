from typing import List
from pykrx import stock
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