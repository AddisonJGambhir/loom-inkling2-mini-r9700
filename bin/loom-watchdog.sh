#!/bin/bash
# loom-watchdog.sh BPORT TAG [UNIT] [PROXY_UNIT] : if Loom marks the model failed (e.g. GPU memory exhausted),
# restart loom-backend in place and re-pin the model, so :1234 recovers by itself instead of returning 503
# "model is unavailable" forever. Rate-limits restarts to avoid a tight crash loop.
BPORT=$1; TAG=$2; UNIT=${3:-loom-backend}; PUNIT=${4:-}
MAX_RESTARTS="${WATCHDOG_MAX_RESTARTS:-5}"
WINDOW_S="${WATCHDOG_WINDOW_S:-600}"
restarts=()
pin() {
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 30 -X POST -H 'Content-Type: application/json' \
    -d "{\"model\":\"$TAG\"}" "http://127.0.0.1:$BPORT/v1/models/pin" 2>/dev/null)
  [[ "$code" == 2* ]] || echo "$(date '+%F %T') warn: pin returned HTTP ${code:-err}"
}
prune_restarts() {
  local now=$1 kept=() t
  for t in "${restarts[@]}"; do (( now - t < WINDOW_S )) && kept+=("$t"); done
  restarts=("${kept[@]}")
}
while sleep 10; do
  if [[ -n "$PUNIT" ]] && ! systemctl --user is-active -q "$PUNIT"; then
    echo "$(date '+%F %T') proxy unit $PUNIT inactive - restarting"
    systemctl --user restart "$PUNIT"
  fi
  systemctl --user is-active -q "$UNIT" || continue
  s=$(curl -s -m 3 "http://127.0.0.1:$BPORT/v1/status" 2>/dev/null) || continue
  if grep -q '"registration":"failed"' <<<"$s"; then
    now=$(date +%s); prune_restarts "$now"
    if (( ${#restarts[@]} >= MAX_RESTARTS )); then
      echo "$(date '+%F %T') crash-loop guard: ${#restarts[@]} restarts in last ${WINDOW_S}s; pausing 60s before retry"
      sleep 60; continue
    fi
    restarts+=("$now")
    echo "$(date '+%F %T') model failed: $(grep -oE '"error":\{[^}]*\}' <<<"$s" | head -1) - restarting $UNIT (${#restarts[@]}/$MAX_RESTARTS in ${WINDOW_S}s)"
    systemctl --user restart "$UNIT"
    for _ in $(seq 300); do curl -s -m 2 "http://127.0.0.1:$BPORT/v1/status" 2>/dev/null | grep -q '"phase":"idle"' && break; sleep 2; done
    pin; echo "$(date '+%F %T') recovered and re-pinned"
  fi
done
