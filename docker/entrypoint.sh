#!/bin/bash
set -euo pipefail

cd /app

WORKER_COUNT=$(python - <<'PY'
from seller_validator.config import Config
print(Config().worker_count)
PY
)

if ! [[ "$WORKER_COUNT" =~ ^[0-9]+$ ]] || [ "$WORKER_COUNT" -le 0 ]; then
  echo "Неверное значение worker_count: $WORKER_COUNT" >&2
  exit 1
fi

DISPLAY_NUMBER=${DISPLAY_NUMBER:-99}
XVFB_CMD=(Xvfb ":${DISPLAY_NUMBER}")

# очищаем возможные артефакты от предыдущего запуска
rm -f "/tmp/.X${DISPLAY_NUMBER}-lock"
rm -f "/tmp/.X11-unix/X${DISPLAY_NUMBER}" 2>/dev/null || true
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix

for ((screen=0; screen<WORKER_COUNT; screen++)); do
  XVFB_CMD+=("-screen" "${screen}" "1920x1080x24")
done

"${XVFB_CMD[@]}" &
XVFB_PID=$!

if ! kill -0 "$XVFB_PID" 2>/dev/null; then
  echo "Не удалось запустить Xvfb" >&2
  exit 1
fi

cleanup() {
  if kill -0 "$XVFB_PID" 2>/dev/null; then
    kill "$XVFB_PID"
    wait "$XVFB_PID" || true
  fi
  rm -f "/tmp/.X${DISPLAY_NUMBER}-lock"
  rm -f "/tmp/.X11-unix/X${DISPLAY_NUMBER}" 2>/dev/null || true
}
trap cleanup EXIT

# даём серверу шанс подняться
sleep 2

export PLAYWRIGHT_DISPLAY=":${DISPLAY_NUMBER}"
export DISPLAY=":${DISPLAY_NUMBER}.0"

mkdir -p data

exec python -m seller_validator
