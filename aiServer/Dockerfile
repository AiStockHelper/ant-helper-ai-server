# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Seoul
WORKDIR /app

# mysqlclient 빌드/실행에 필요한 패키지
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# requirements.txt 에는 앱이 import 하는 tensorflow / pykrx / holidayskr 와 gunicorn 이 빠져 있다.
# 기존 pin 이 바뀌지 않도록 -c(constraints)로 고정한 채 추가 설치한다.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -c requirements.txt tensorflow-cpu==2.15.1 pykrx holidayskr gunicorn

# secrets.json / my_settings.py 는 이미지에 넣지 않는다(.dockerignore). 실행 시 마운트한다.
COPY . .

EXPOSE 8000
# 기존 배포(migrate → collectstatic → gunicorn)와 같은 순서.
CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py collectstatic --noinput && exec gunicorn aiServer.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 120 --access-logfile - --error-logfile -"]
