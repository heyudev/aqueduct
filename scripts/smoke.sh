#!/usr/bin/env bash
# End-to-end smoke test against the Aqueduct gateway.
#
#   ./scripts/smoke.sh                      # uses .env
#   AQ_URL=... AQ_KEY=... ./scripts/smoke.sh
#
# Checks liveness, model listing, non-streaming completion, streaming, and a
# small burst of concurrent requests. Exits non-zero on the first hard failure.

set -uo pipefail

RED=$'\033[31m'; GRN=$'\033[32m'; BLD=$'\033[1m'; RST=$'\033[0m'
fails=0

pass() { printf '  %sPASS%s  %s\n' "$GRN" "$RST" "$1"; }
fail() { printf '  %sFAIL%s  %s\n' "$RED" "$RST" "$1"; fails=$((fails + 1)); }
step() { printf '\n%s== %s ==%s\n' "$BLD" "$1" "$RST"; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; . "$repo_root/.env"; set +a
fi

AQ_URL="${AQ_URL:-http://127.0.0.1:4000}"
# Falls back to the master key so the test works before any virtual key exists.
AQ_KEY="${AQ_KEY:-${LITELLM_MASTER_KEY:-}}"
AQ_MODEL="${AQ_MODEL:-aqueduct-chat}"

if [[ -z "$AQ_KEY" ]]; then
  printf '%sAQ_KEY is empty%s — set AQ_KEY or LITELLM_MASTER_KEY.\n' "$RED" "$RST"
  exit 2
fi

auth=(-H "Authorization: Bearer $AQ_KEY" -H "Content-Type: application/json")

step "Liveness"
if curl -fsS --max-time 10 "$AQ_URL/health/liveliness" >/dev/null 2>&1; then
  pass "gateway is up at $AQ_URL"
else
  fail "gateway not reachable at $AQ_URL"
  printf '\n%sGateway is down; skipping the rest.%s\n' "$RED" "$RST"
  exit 1
fi

step "Model list"
models="$(curl -fsS --max-time 10 "${auth[@]}" "$AQ_URL/v1/models" 2>/dev/null)"
if [[ -n "$models" ]] && grep -q "$AQ_MODEL" <<<"$models"; then
  pass "$AQ_MODEL is registered"
else
  fail "$AQ_MODEL not found in /v1/models — check gateway/config.yaml"
  printf '        response: %s\n' "${models:0:300}"
fi

step "Chat completion (non-streaming)"
start=$(date +%s)
resp="$(curl -fsS --max-time 180 "${auth[@]}" "$AQ_URL/v1/chat/completions" \
  -d "$(cat <<JSON
{
  "model": "$AQ_MODEL",
  "messages": [{"role": "user", "content": "Reply with exactly one word: pong"}],
  "max_tokens": 16,
  "temperature": 0
}
JSON
)" 2>/dev/null)"
elapsed=$(( $(date +%s) - start ))

if [[ -n "$resp" ]] && grep -q '"content"' <<<"$resp"; then
  pass "got a completion in ${elapsed}s"
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "$resp" | python3 -c '
import json, sys
d = json.load(sys.stdin)
msg = d["choices"][0]["message"]["content"].strip()
u = d.get("usage", {})
print("        content: %r" % msg[:80])
print("        tokens : prompt=%s completion=%s" % (u.get("prompt_tokens"), u.get("completion_tokens")))
'
  fi
else
  fail "no valid completion returned"
  printf '        response: %s\n' "${resp:0:300}"
fi

step "Chat completion (streaming)"
chunks="$(curl -fsS --max-time 180 -N "${auth[@]}" "$AQ_URL/v1/chat/completions" \
  -d "$(cat <<JSON
{
  "model": "$AQ_MODEL",
  "messages": [{"role": "user", "content": "Count from 1 to 10."}],
  "max_tokens": 64,
  "temperature": 0,
  "stream": true
}
JSON
)" 2>/dev/null | grep -c '^data: ')"

if (( chunks > 1 )); then
  pass "received $chunks SSE chunks"
else
  fail "streaming returned $chunks chunks — expected many"
fi

step "Concurrency (8 parallel requests)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
start=$(date +%s)
for i in $(seq 1 8); do
  (
    code="$(curl -fsS -o "$tmp/body.$i" -w '%{http_code}' --max-time 240 "${auth[@]}" \
      "$AQ_URL/v1/chat/completions" \
      -d "{\"model\":\"$AQ_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about the number $i.\"}],\"max_tokens\":64}" 2>/dev/null)"
    echo "$code" > "$tmp/code.$i"
  ) &
done
wait
elapsed=$(( $(date +%s) - start ))
ok=$(grep -l '^200$' "$tmp"/code.* 2>/dev/null | wc -l | tr -d ' ')
if [[ "$ok" == "8" ]]; then
  pass "8/8 concurrent requests succeeded in ${elapsed}s"
else
  fail "$ok/8 concurrent requests succeeded — check vLLM logs for OOM or queue rejections"
fi

step "Result"
if (( fails > 0 )); then
  printf '%s%d check(s) failed.%s\n' "$RED" "$fails" "$RST"
  exit 1
fi
printf '%sSmoke test passed.%s\n' "$GRN" "$RST"
