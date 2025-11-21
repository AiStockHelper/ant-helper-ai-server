from typing import Any

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .services import (
    save_all_domestic_stocks,
    create_tomorrow_price_prediction_model,
    predict_tomorrow_price,
)


class FetchDomesticStocksView(APIView):
    """
    전체 국내 주식 목록을 DB에 저장하는 API
    """
    def post(self, request: Any) -> Response:
        count = save_all_domestic_stocks()
        return Response(
            {"message": f"{count} stocks processed successfully."},
            status=status.HTTP_201_CREATED,
        )


class TrainTomorrowPriceModelView(APIView):
    """
    내일 종가 예측 모델 생성 API
    """
    def post(self, request: Any) -> Response:
        create_tomorrow_price_prediction_model()
        return Response(
            {"message": "Tomorrow price prediction model created successfully."},
            status=status.HTTP_201_CREATED,
        )


class PredictTomorrowPriceView(APIView):
    """
    내일 종가 예측 API
    """
    def get(self, request: Any) -> Response:
        result = predict_tomorrow_price()
        return Response(
            {"result": result},
            status=status.HTTP_200_OK,
        )