#!/usr/bin/env bash
# Pre-deployment health check for an Aqueduct GPU host.
#
# Run this first on a freshly delivered machine, before touching anything else.
# It verifies the things that, when wrong, cost hours of confused debugging
# later: card count, NVLink topology, driver, container runtime, disk, RAM.
#
# Exits non-zero if any FAIL is reported. WARNs do not fail the run.

set -uo pipefail

RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLD=$'\033[1m'; RST=$'\033[0m'
fails=0
warns=0

pass() { printf '  %sPASS%s  %s\n' "$GRN" "$RST" "$1"; }
warn() { printf '  %sWARN%s  %s\n' "$YLW" "$RST" "$1"; warns=$((warns + 1)); }
fail() { printf '  %sFAIL%s  %s\n' "$RED" "$RST" "$1"; fails=$((fails + 1)); }
info() { printf '  ----  %s\n' "$1"; }
section() { printf '\n%s== %s ==%s\n' "$BLD" "$1" "$RST"; }

EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
MIN_DISK_GB="${MIN_DISK_GB:-4000}"
MIN_RAM_GB="${MIN_RAM_GB:-1000}"

# Load MODEL_DIR from .env if present so the disk check targets the right mount.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; . "$repo_root/.env"; set +a
fi
MODEL_DIR="${MODEL_DIR:-/models}"

# --------------------------------------------------------------------------
section "GPU"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found — the NVIDIA driver is not installed"
else
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  info "driver version: $driver"

  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  if [[ "$gpu_count" == "$EXPECTED_GPUS" ]]; then
    pass "$gpu_count GPUs visible"
  else
    fail "expected $EXPECTED_GPUS GPUs, found $gpu_count"
  fi

  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS= read -r line; do
    info "$line"
  done

  # This project is specified around the 96GB H20. The 141GB variant also
  # exists and differs by 360GB across 8 cards, which invalidates every capacity
  # number in docs/hardware.md — so flag anything that is not what we ordered.
  mem_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')"
  mem_gb=$(( ${mem_mib:-0} / 1024 ))
  if (( mem_gb == 0 )); then
    warn "could not read GPU memory size"
  elif (( mem_gb >= 85 && mem_gb < 130 )); then
    pass "per-GPU memory ${mem_gb}GB — 96GB class as expected, total $((mem_gb * gpu_count))GB"
  elif (( mem_gb >= 130 )); then
    warn "per-GPU memory ${mem_gb}GB — this is the 141GB variant, not the 96GB one this project assumes"
    warn "  total $((mem_gb * gpu_count))GB: more headroom than planned, but re-run the capacity math in docs/hardware.md"
  else
    warn "per-GPU memory ${mem_gb}GB is below the expected H20 range; check the SKU"
  fi

  # ECC errors on a new machine mean bad hardware. Catch it before it corrupts
  # a week of load testing.
  ecc="$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | grep -v '^\[' | paste -sd, -)"
  if [[ -z "$ecc" ]]; then
    info "ECC error counters unavailable"
  elif [[ "$ecc" =~ ^(0,)*0$ ]]; then
    pass "no uncorrected ECC errors"
  else
    fail "uncorrected ECC errors present: $ecc — this is a hardware problem"
  fi
fi

# --------------------------------------------------------------------------
section "Interconnect"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  info "skipped — no nvidia-smi"
else
  topo="$(nvidia-smi topo -m 2>/dev/null)"
  if [[ -z "$topo" ]]; then
    warn "could not read GPU topology"
  else
    # Tensor parallelism across 8 cards over PCIe instead of NVLink is a large,
    # silent performance loss. Every GPU-to-GPU pair should show NV#.
    if grep -qE 'NV[0-9]+' <<<"$topo"; then
      pass "NVLink links detected"
      if grep -qE '^GPU[0-9]+.*\b(PIX|PXB|PHB|NODE|SYS)\b' <<<"$topo"; then
        warn "some GPU pairs are not NVLink-connected — run 'nvidia-smi topo -m' and check"
      fi
    else
      fail "no NVLink between GPUs (PCIe-only board?) — TP=8 will be slow"
    fi
  fi
fi

# --------------------------------------------------------------------------
section "Container runtime"

if ! command -v docker >/dev/null 2>&1; then
  fail "docker not installed"
else
  pass "docker $(docker --version | awk '{print $3}' | tr -d ',')"

  if docker compose version >/dev/null 2>&1; then
    pass "docker compose plugin present"
  else
    fail "docker compose plugin missing (the standalone docker-compose v1 is not supported)"
  fi

  # Without the NVIDIA container toolkit, containers see no GPUs at all.
  if docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L >/dev/null 2>&1; then
    pass "NVIDIA container toolkit works (GPUs visible inside containers)"
  else
    fail "cannot access GPUs from a container — install/configure nvidia-container-toolkit"
  fi
fi

# --------------------------------------------------------------------------
section "Storage and memory"

if [[ -d "$MODEL_DIR" ]]; then
  avail_gb="$(df -BG --output=avail "$MODEL_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')"
  if [[ -n "$avail_gb" ]] && (( avail_gb >= MIN_DISK_GB )); then
    pass "MODEL_DIR $MODEL_DIR has ${avail_gb}GB free"
  else
    warn "MODEL_DIR $MODEL_DIR has ${avail_gb:-?}GB free, want >= ${MIN_DISK_GB}GB (a 480B FP8 model is ~500GB)"
  fi

  # Weight loading is I/O bound. Spinning disks or network storage turn a
  # 5-minute startup into an hour.
  src="$(df --output=source "$MODEL_DIR" 2>/dev/null | tail -1)"
  info "MODEL_DIR backed by: $src"
else
  warn "MODEL_DIR $MODEL_DIR does not exist yet — create it and set MODEL_DIR in .env"
fi

ram_gb="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null)"
if [[ -n "$ram_gb" ]]; then
  if (( ram_gb >= MIN_RAM_GB )); then
    pass "system RAM ${ram_gb}GB"
  else
    warn "system RAM ${ram_gb}GB, want >= ${MIN_RAM_GB}GB (model loading buffers through host memory)"
  fi
fi

# TP workers exchange tensors through /dev/shm. The docker default of 64MB is
# far too small and produces an opaque hang at startup; compose sets shm_size,
# this checks the host has room to back it.
shm_gb="$(df -BG /dev/shm 2>/dev/null | tail -1 | awk '{print $2}' | tr -dc '0-9')"
if [[ -n "$shm_gb" ]] && (( shm_gb >= 32 )); then
  pass "/dev/shm is ${shm_gb}GB"
else
  warn "/dev/shm is ${shm_gb:-?}GB — tensor-parallel workers need tens of GB"
fi

# --------------------------------------------------------------------------
section "Result"

if (( fails > 0 )); then
  printf '%s%d check(s) failed%s, %d warning(s). Fix the failures before deploying.\n' "$RED" "$fails" "$RST" "$warns"
  exit 1
fi
printf '%sAll checks passed%s (%d warning(s)).\n' "$GRN" "$RST" "$warns"
