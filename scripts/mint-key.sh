#!/usr/bin/env bash
# Mint an Aqueduct API key for an internal team.
#
#   TEAM=search BUDGET=200 ./scripts/mint-key.sh
#   TEAM=devtools BUDGET=500 MODELS=aqueduct-chat,aqueduct-fast ./scripts/mint-key.sh
#
# BUDGET is in USD-equivalent units. Self-hosted inference has no real dollar
# cost per call, but LiteLLM's budget mechanism is still the simplest way to cap
# a runaway client — set the per-model price in gateway/config.yaml if you want
# the numbers to mean something, otherwise treat the budget as a relative quota.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; . "$repo_root/.env"; set +a
fi

AQ_URL="${AQ_URL:-http://127.0.0.1:4000}"
: "${LITELLM_MASTER_KEY:?LITELLM_MASTER_KEY is not set — check .env}"
: "${TEAM:?set TEAM=<team-name>}"

BUDGET="${BUDGET:-100}"
MODELS="${MODELS:-aqueduct-chat}"
DURATION="${DURATION:-}"

# Turn the comma-separated list into a JSON array.
models_json="$(python3 -c '
import json, sys
print(json.dumps([m.strip() for m in sys.argv[1].split(",") if m.strip()]))
' "$MODELS")"

payload="$(python3 -c '
import json, sys
team, budget, models, duration = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), sys.argv[4]
body = {
    "key_alias": f"aqueduct-{team}",
    "models": models,
    "max_budget": float(budget),
    "budget_duration": "30d",
    "metadata": {"team": team, "issued_by": "scripts/mint-key.sh"},
}
if duration:
    body["duration"] = duration
print(json.dumps(body))
' "$TEAM" "$BUDGET" "$models_json" "$DURATION")"

resp="$(curl -fsS -X POST "$AQ_URL/key/generate" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d "$payload")"

printf '%s' "$resp" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print()
print("  key    :", d.get("key"))
print("  alias  :", d.get("key_alias"))
print("  models :", ", ".join(d.get("models") or ["(all)"]))
print("  budget :", d.get("max_budget"), "per", d.get("budget_duration"))
print()
print("  Hand this key to the team over a secure channel. It is shown once;")
print("  the gateway stores only a hash. Lost keys are re-minted, not recovered.")
print()
'
