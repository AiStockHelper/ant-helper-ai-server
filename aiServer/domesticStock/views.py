from typing import Any

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .services import save_all_domestic_stocks

class FetchDomesticStocksView(APIView):

    """
    전체 국내 주식 목록을 DB에 저장하는 API (중복은 자동으로 제외)
    """
    def post(self, request: Any) -> Response:
        count: int = save_all_domestic_stocks()
        return Response(
            {"message": f"{count} stocks processed successfully."},
            status=status.HTTP_201_CREATED,
        )
