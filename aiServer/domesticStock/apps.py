from django.apps import AppConfig


class DomesticStockConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'domesticStock'

    def ready(self):
        from . import scheduler
        scheduler.start()