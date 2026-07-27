#!/usr/bin/env bash
# Download model weights into MODEL_DIR.
#
#   ./scripts/download-model.sh Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
#   HF_ENDPOINT=https://hf-mirror.com ./scripts/download-model.sh <repo>
#
# Several hundred GB per model. Use a mirror if the direct connection is slow —
# HF_ENDPOINT is honoured by huggingface_hub and by this script's container
# fallback. Downloads resume, so an interrupted run is safe to re-run.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; . "$repo_root/.env"; set +a
fi

MODEL_REPO="${1:?usage: $0 <hf-org/model-repo>}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR in .env}"
TARGET="$MODEL_DIR/$(basename "$MODEL_REPO")"

mkdir -p "$TARGET"

avail_gb="$(df -BG --output=avail "$MODEL_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
echo "Target   : $TARGET"
echo "Free disk: ${avail_gb}GB"
echo "Mirror   : ${HF_ENDPOINT:-https://huggingface.co (direct)}"
echo

if (( avail_gb < 600 )); then
  echo "Warning: less than 600GB free. Large FP8 models will not fit."
  read -r -p "Continue anyway? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi

# Prefer a local hf CLI; otherwise run it in a throwaway container so the host
# needs no Python environment.
if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL_REPO" --local-dir "$TARGET"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$MODEL_REPO" --local-dir "$TARGET"
else
  echo "No hf CLI found; downloading via container."
  docker run --rm \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_ENDPOINT="${HF_ENDPOINT:-}" \
    -v "$TARGET:/out" \
    python:3.11-slim \
    bash -c "pip install -q --no-cache-dir 'huggingface_hub[cli]' && hf download '$MODEL_REPO' --local-dir /out"
fi

echo
echo "Downloaded to $TARGET"
echo "Set MODEL_PATH=/models/$(basename "$MODEL_REPO") in inference/vllm/models/<name>.env"
du -sh "$TARGET" 2>/dev/null || true
