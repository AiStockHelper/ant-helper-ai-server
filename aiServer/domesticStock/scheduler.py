# stock/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from .services import save_all_domestic_stocks, predict_all_tomorrow_price


def start():
    scheduler = BackgroundScheduler(timezone="Asia/Seoul")

    # 매일 자정(00:00)에 실행
    scheduler.add_job(
        save_all_domestic_stocks,
        trigger=CronTrigger(hour=0, minute=0),
        id="save_all_domestic_stocks",
        replace_existing=True,
    )

    # 매일 오후 9시에 실행(내일 주가 예측)
    scheduler.add_job(
        predict_all_tomorrow_price,
        trigger=CronTrigger(hour=21, minute=0),
        id="predict_all_tomorrow_price",
        replace_existing=True,
    )

    # 서버 시작 시 한 번 즉시 실행
    save_all_domestic_stocks()
    scheduler.start()