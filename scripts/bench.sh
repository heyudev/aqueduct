#!/usr/bin/env bash
# Load test the inference layer to find the real capacity of this machine.
#
#   ./scripts/bench.sh                                  # default sweep
#   CONCURRENCY="1 8 32 64" ./scripts/bench.sh
#   INPUT_LEN=8192 OUTPUT_LEN=512 ./scripts/bench.sh
#
# Points vLLM's own benchmark tool at the inference server directly, bypassing
# the gateway, so the numbers describe the model and not the proxy. Run the
# gateway comparison separately if you want to know its overhead.
#
# What to look at, given H20's compute-light / bandwidth-heavy profile:
#   TTFT  (time to first token) — dominated by prefill; the weak spot. Watch how
#         it degrades as INPUT_LEN grows. This is what MAX_NUM_BATCHED_TOKENS
#         trades against.
#   TPOT  (time per output token) — dominated by memory bandwidth; should stay
#         good. If it degrades sharply with concurrency, KV cache is the limit.
#   Throughput — the number to quote for capacity planning.
#
# Then re-run with a shared prefix to measure what prefix caching actually buys
# you, since that is the single biggest lever on this hardware.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; . "$repo_root/.env"; set +a
fi

MODEL="${MODEL:-primary}"
MODEL_ENV="$repo_root/inference/vllm/models/${MODEL}.env"
if [[ -f "$MODEL_ENV" ]]; then
  # shellcheck disable=SC1090
  set -a; . "$MODEL_ENV"; set +a
fi

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-aqueduct-chat}"
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-latest}"

CONCURRENCY="${CONCURRENCY:-1 4 16 32 64}"
INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-64}"
RESULT_DIR="${RESULT_DIR:-$repo_root/bench-results}"

mkdir -p "$RESULT_DIR"

if ! curl -fsS --max-time 5 "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
  echo "vLLM is not answering at ${VLLM_HOST}:${VLLM_PORT} — start it with: make up-model"
  exit 1
fi

echo "Benchmarking $SERVED_MODEL_NAME at ${VLLM_HOST}:${VLLM_PORT}"
echo "  input=${INPUT_LEN} output=${OUTPUT_LEN} requests=${REQUESTS_PER_LEVEL} per level"
echo "  results -> $RESULT_DIR"
echo

for c in $CONCURRENCY; do
  echo "--- concurrency $c ---"
  # `vllm bench serve` ships inside the official image. Older releases expose it
  # as benchmarks/benchmark_serving.py in the source tree instead; if the
  # subcommand is missing, check the version of the image you pinned.
  docker run --rm --network host \
    -v "$RESULT_DIR:/results" \
    "vllm/vllm-openai:${VLLM_IMAGE_TAG}" \
    bench serve \
      --backend openai-chat \
      --base-url "http://${VLLM_HOST}:${VLLM_PORT}" \
      --endpoint /v1/chat/completions \
      --model "$SERVED_MODEL_NAME" \
      --dataset-name random \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --num-prompts "$REQUESTS_PER_LEVEL" \
      --max-concurrency "$c" \
      --save-result \
      --result-dir /results \
      --result-filename "c${c}-in${INPUT_LEN}-out${OUTPUT_LEN}.json" \
    || echo "  (level $c failed — see output above)"
  echo
done

echo "Done. Raw results in $RESULT_DIR"
echo "Record the numbers in docs/runbook.md so future tuning has a baseline."
