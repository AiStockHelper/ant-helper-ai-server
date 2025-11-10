from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import FetchDomesticStocksView

router = DefaultRouter()

urlpatterns = [
    path('domestic-stocks/fetch/', FetchDomesticStocksView.as_view(), name='fetch_domestic_stocks'),
]