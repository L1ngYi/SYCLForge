#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_SYCL="${ENV_SYCL:-/home/l1ngyi/env_sycl.sh}"
BASE_RUN="${BASE_RUN:-syclforge_full_runs/20260623_122215_batch_deepseek-v4-pro}"
WRITE_BACK_DIR="${WRITE_BACK_DIR:-syclforge_best_full}"
BASELINE_DETAIL="${BASELINE_DETAIL:-benchmark/baselines/gemm_detail_newprompt_ds.csv}"
RTOL="${RTOL:-2e-2}"
ATOL="${ATOL:-1e-1}"
A2_ATOL="${A2_ATOL:-2e-1}"
A2_SOURCE="${A2_SOURCE:-$BASE_RUN/gemm_2048_16384_2048/code/round_002_candidate_tensor_free.cpp}"

if [[ -f "$ENV_SYCL" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_SYCL"
fi

if [[ ! -d "$BASE_RUN" ]]; then
  echo "[SYCLForge patch] missing BASE_RUN: $BASE_RUN" >&2
  exit 1
fi

mkdir -p "$WRITE_BACK_DIR"

echo "[SYCLForge patch] base run: $BASE_RUN"
echo "[SYCLForge patch] patch S1 with event-zero wall-clock fallback"
"$PYTHON_BIN" -m syclforge.main_sycl cpu_sycl \
  --case-stem gemm_128_128_128 \
  --resume-run "$BASE_RUN" \
  --round 1 \
  --no-llm \
  --tensor-core \
  --isolated-benchmark \
  --rtol "$RTOL" \
  --atol "$ATOL" \
  --write-back-dir "$WRITE_BACK_DIR"

if [[ -f "$A2_SOURCE" ]]; then
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT
  cp "$A2_SOURCE" "$tmp_dir/gemm_2048_16384_2048.cpp"

  echo "[SYCLForge patch] patch A2 from: $A2_SOURCE"
  echo "[SYCLForge patch] A2 uses A2_ATOL=$A2_ATOL for high-K TF32 accumulation"
  "$PYTHON_BIN" -m syclforge.main_sycl "$tmp_dir/gemm_2048_16384_2048.cpp" \
    --resume-run "$BASE_RUN" \
    --rerun-completed \
    --round 1 \
    --no-llm \
    --tensor-core \
    --require-tensor-core \
    --isolated-benchmark \
    --rtol "$RTOL" \
    --atol "$A2_ATOL" \
    --write-back-dir "$WRITE_BACK_DIR"
else
  echo "[SYCLForge patch] A2 source not found; skipping A2 patch: $A2_SOURCE" >&2
fi

"$PYTHON_BIN" -m syclforge.summarize_full_suite "$BASE_RUN" \
  --baseline-detail "$BASELINE_DETAIL"

echo "[SYCLForge patch] done"
