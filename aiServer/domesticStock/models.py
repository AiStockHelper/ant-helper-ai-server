from django.db import models

# 국내주식 정보
class DomesticStock(models.Model):
    product_number = models.CharField(primary_key=True, max_length=100, unique=True)
    name = models.CharField(max_length=255)

    class Meta:
        db_table = "domestic_stock"

    def __str__(self):
        return f"{self.product_number} - {self.name}"