#!/usr/bin/env bash
# EC2 에서 실행되는 배포 스크립트 (GitHub Actions 가 ssh 로 stdin 에 흘려 넣는다).
# 필수 환경변수: SERVICE (django | prediction), IMAGE (ECR 이미지 URI)
# ECR 로그인은 워크플로우가 먼저 해 둔다. 새 컨테이너가 헬스 체크를 통과하지 못하면 이전 이미지로 되돌린다.
set -euo pipefail

: "${SERVICE:?SERVICE is required (django|prediction)}"
: "${IMAGE:?IMAGE is required}"

APP_DIR="$HOME/ant-helper"
D="sudo docker"
RUN_ARGS=()

case "$SERVICE" in
  django)
    NAME=ant-helper-ai-django
    HEALTH_URL="http://localhost:8000/"
    # ALLOWED_HOSTS 에 localhost 가 없어서 Host 헤더를 맞춰 준다. 루트가 404 여도 앱이 떠 있으면 정상(<500)으로 본다.
    CURL_ARGS=(-H "Host: ai.ant-helper.com")
    MAX_WAIT_STEPS=60   # 시작 시 migrate/collectstatic/스케줄러 초기 동기화가 있어 오래 걸릴 수 있다.
    STATIC_DIR="$HOME/aiServer/aiServer/staticfiles"   # 기존 배포가 collectstatic 결과를 두던 위치
    mkdir -p "$APP_DIR/django-config" "$STATIC_DIR"
    install -m 600 "$APP_DIR"/incoming/django-config/* "$APP_DIR/django-config/"
    rm -rf "$APP_DIR/incoming/django-config"
    # secrets.json / my_settings.py 는 이미지에 없다. settings.py 가 읽는 위치에 읽기 전용으로 마운트한다.
    RUN_ARGS=(
      -v "$APP_DIR/django-config/secrets.json:/app/aiServer/secrets.json:ro"
      -v "$APP_DIR/django-config/my_settings.py:/app/aiServer/my_settings.py:ro"
      -v "$STATIC_DIR:/app/aiServer/staticfiles"
    )
    # 기존 gunicorn(nohup) / 예전 컨테이너가 8000 을 잡고 있으면 정리한다.
    pkill -9 gunicorn || true
    ;;
  prediction)
    NAME=ant-helper-ai-prediction
    HEALTH_URL="http://localhost:8001/health"
    CURL_ARGS=()
    MAX_WAIT_STEPS=24
    ;;
  *)
    echo "unknown SERVICE: $SERVICE" >&2
    exit 2
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  echo ">> installing docker"
  sudo apt-get update -y
  sudo apt-get install -y docker.io curl
  sudo systemctl enable --now docker
fi

run_container() {
  # host 네트워크: DB 주소 localhost 와 기존 nginx(localhost:8000) 프록시를 그대로 쓴다.
  $D run -d --name "$NAME" --restart unless-stopped --network host "${RUN_ARGS[@]}" "$1" >/dev/null
}

wait_healthy() {
  local i code
  for i in $(seq 1 "$MAX_WAIT_STEPS"); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "${CURL_ARGS[@]}" "$HEALTH_URL" || true)"
    if [ "$code" != "000" ] && [ "$code" -lt 500 ]; then
      echo ">> healthy (HTTP $code) after $((i * 5))s"
      return 0
    fi
    sleep 5
  done
  return 1
}

PREV_IMAGE="$($D inspect -f '{{.Config.Image}}' "$NAME" 2>/dev/null || true)"

echo ">> pulling $IMAGE"
$D pull "$IMAGE"

$D rm -f "$NAME" >/dev/null 2>&1 || true
echo ">> starting $IMAGE"
run_container "$IMAGE"

if wait_healthy; then
  $D image prune -af --filter "until=168h" >/dev/null 2>&1 || true
  exit 0
fi

echo "!! health check failed ($HEALTH_URL). recent logs:"
$D logs --tail 80 "$NAME" || true

if [ -n "$PREV_IMAGE" ] && [ "$PREV_IMAGE" != "$IMAGE" ]; then
  echo "!! rolling back to $PREV_IMAGE"
  $D rm -f "$NAME" >/dev/null 2>&1 || true
  run_container "$PREV_IMAGE"
fi
exit 1
