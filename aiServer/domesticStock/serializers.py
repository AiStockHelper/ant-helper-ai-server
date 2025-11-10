from rest_framework import serializers
from .models import DomesticStock

class DomesticStockSerializer(serializers.ModelSerializer):
    class Meta:
        model = DomesticStock
        fields = '__all__'