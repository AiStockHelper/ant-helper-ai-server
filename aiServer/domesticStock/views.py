from typing import Any

from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .services import (
    save_all_domestic_stocks,
    create_tomorrow_price_prediction_model,
    get_tomorrow_price, predict_all_tomorrow_price,
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
    내일 종가 예측 API (단일 종목)
    GET /stock/predict/
    """
    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(
                "product_number",
                openapi.IN_QUERY,
                description="종목 코드 (예: 005930)",
                type=openapi.TYPE_STRING,
                required=True,
            )
        ]
    )
    def get(self, request):
        product_number = request.query_params.get("product_number")
        if not product_number:
            return Response({"error": "product_number required"}, status=400)

        predicted, today = get_tomorrow_price(product_number)
        return Response({
            "product_number": product_number,
            "predicted_price": predicted,
            "today_price": today
        })


    """
    전체 종목 내일 종가 예측 값 DB 저장 API
    POST /stock/predict/
    """
    def post(self, request: Any) -> Response:
        count = predict_all_tomorrow_price()
        return Response(
            {"saved_predictions": count},
            status=status.HTTP_200_OK,
        )