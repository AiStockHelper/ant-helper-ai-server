from django.urls import path
from .views import (
    FetchDomesticStocksView,
    TrainTomorrowPriceModelView,
    PredictTomorrowPriceView
)

urlpatterns = [
    path('fetch/', FetchDomesticStocksView.as_view(), name='fetch_domestic_stocks'),
    path('create-model/', TrainTomorrowPriceModelView.as_view(), name='create_price_model'),

    # stock/urls.py
    path("predict/", PredictTomorrowPriceView.as_view(), name="predict_price"),
]