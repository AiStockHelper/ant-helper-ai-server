from django.db import models

# 국내주식 정보
class DomesticStock(models.Model):
    product_number = models.CharField(primary_key=True, max_length=100, unique=True)
    name = models.CharField(max_length=255)

    class Meta:
        db_table = "domestic_stock"

    def __str__(self):
        return f"{self.product_number} - {self.name}"


# 예측 국내주식 주가
# 매일 예측주가를 DB에 저장해둠. 기존 데이터는 삭제 x
class DomesticStockPredictedPrice(models.Model):
    product_number = models.CharField(max_length=100)
    predicted_date = models.DateField()   # 예측 기준 날짜(D+1)
    predicted_price = models.FloatField()
    today_price = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('product_number', 'predicted_date')
        db_table = "domestic_stock_predicted_price"