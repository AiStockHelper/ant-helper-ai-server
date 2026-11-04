#!/usr/bin/env bash
# EC2 에서 실행되는 prediction 서버 배포 스크립트 (GitHub Actions 가 ssh 로 stdin 에 흘려 넣는다).
# 필수 환경변수: SHA (배포할 커밋)
# 서버 구성: systemd 서비스 ant-prediction (uvicorn, 127.0.0.1:8000),
#   코드 = /opt/antlab/ant-prediction/repo (이 레포의 clone, 앱은 repo/prediction), venv = /opt/antlab/ant-prediction/venv
# 헬스 체크에 실패하면 직전 커밋으로 되돌린다. repo/prediction/data (예측 캐시)는 추적 대상이 아니라 유지된다.
set -euo pipefail

: "${SHA:?SHA is required}"

BASE=/opt/antlab/ant-prediction
REPO="$BASE/repo"
VENV="$BASE/venv"
SERVICE=ant-prediction
HEALTH_URL="http://127.0.0.1:8000/health"

cd "$REPO"
PREV="$(git rev-parse HEAD)"
echo ">> current $PREV -> target $SHA"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "!! server clone has local modifications (will be discarded):"; git status --short --untracked-files=no
fi

wait_healthy() {
  local i code
  for i in $(seq 1 36); do
    code="$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$HEALTH_URL" || true)"
    if [ "$code" = "200" ]; then echo ">> healthy after ~$((i * 5))s"; return 0; fi
    sleep 5
  done
  return 1
}

install_deps() {
  # 메모리가 빠듯한 서버라 우선순위를 낮춰 설치한다. 서비스는 설치가 끝난 뒤에 재시작한다.
  nice -n 10 "$VENV/bin/pip" install --quiet --disable-pip-version-check --no-cache-dir -r "$REPO/prediction/requirements.txt"
}

git fetch --quiet origin "$SHA" || git fetch --quiet origin release
git checkout --quiet -f --detach "$SHA"
echo ">> now at $(git log -1 --format='%h %s')"

install_deps
sudo systemctl restart "$SERVICE"

if wait_healthy; then exit 0; fi

echo "!! health check failed. recent logs:"
sudo journalctl -u "$SERVICE" -n 60 --no-pager | cut -c1-220 || true

echo "!! rolling back to $PREV"
git checkout --quiet -f --detach "$PREV"
install_deps || true
sudo systemctl restart "$SERVICE"
if wait_healthy; then echo ">> rollback OK"; else echo "!! rollback ALSO unhealthy - manual check needed"; fi
exit 1
