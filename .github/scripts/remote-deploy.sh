#!/usr/bin/env bash
# EC2 에서 실행되는 배포 스크립트 (GitHub Actions 가 ssh 로 stdin 에 흘려 넣는다).
# 필수 환경변수: IMAGE (ECR 이미지 URI)
# ECR 로그인은 워크플로우가 먼저 해 둔다. 새 컨테이너가 헬스 체크를 통과하지 못하면 이전 이미지로 되돌린다.
set -euo pipefail

: "${IMAGE:?IMAGE is required}"

NAME=ant-helper-ai-prediction
HEALTH_URL="http://localhost:8001/health"
MAX_WAIT_STEPS=24
D="sudo docker"

if ! command -v docker >/dev/null 2>&1; then
  echo ">> installing docker"
  sudo apt-get update -y
  sudo apt-get install -y docker.io curl
  sudo systemctl enable --now docker
fi

run_container() {
  # host 네트워크: 백엔드가 localhost:8001 로 호출한다.
  $D run -d --name "$NAME" --restart unless-stopped --network host "$1" >/dev/null
}

wait_healthy() {
  local i code
  for i in $(seq 1 "$MAX_WAIT_STEPS"); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "$HEALTH_URL" || true)"
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
