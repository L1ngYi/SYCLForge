#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_SYCL="${ENV_SYCL:-/home/l1ngyi/env_sycl.sh}"
SYCL_PATH="${SYCL_PATH:-cpu_sycl}"
WORK_DIR="${WORK_DIR:-syclforge_full_runs}"
WRITE_BACK_DIR="${WRITE_BACK_DIR:-syclforge_best_full}"
RESUME_RUN="${RESUME_RUN:-}"
ROUNDS="${ROUNDS:-5}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
RTOL="${RTOL:-2e-2}"
ATOL="${ATOL:-1e-1}"
SERVER_TYPE="${SERVER_TYPE:-deepseek}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4-pro}"
BASELINE_DETAIL="${BASELINE_DETAIL:-benchmark/baselines/gemm_detail_newprompt_ds.csv}"

if [[ -f "$ENV_SYCL" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_SYCL"
fi

mkdir -p "$WORK_DIR" "$WRITE_BACK_DIR"

echo "[SYCLForge suite] source: $SYCL_PATH"
echo "[SYCLForge suite] work dir: $WORK_DIR"
echo "[SYCLForge suite] write-back dir: $WRITE_BACK_DIR"
echo "[SYCLForge suite] rounds: $ROUNDS"
echo "[SYCLForge suite] baseline detail: $BASELINE_DETAIL"
if [[ -n "$RESUME_RUN" ]]; then
  echo "[SYCLForge suite] resume run: $RESUME_RUN"
fi

MAIN_ARGS=(
  -m syclforge.main_sycl "$SYCL_PATH"
  --round "$ROUNDS"
  --server_type "$SERVER_TYPE"
  --model_name "$MODEL_NAME"
  --max_tokens "$MAX_TOKENS"
  --tensor-core
  --rtol "$RTOL"
  --atol "$ATOL"
  --work-dir "$WORK_DIR"
  --write-back-dir "$WRITE_BACK_DIR"
)

if [[ -n "$RESUME_RUN" ]]; then
  MAIN_ARGS+=(--resume-run "$RESUME_RUN")
fi

"$PYTHON_BIN" "${MAIN_ARGS[@]}"

if [[ -n "$RESUME_RUN" ]]; then
  LATEST_RUN="$RESUME_RUN"
else
  LATEST_RUN="$(ls -dt "$WORK_DIR"/* | head -1)"
fi
echo "[SYCLForge suite] latest run: $LATEST_RUN"

"$PYTHON_BIN" -m syclforge.summarize_full_suite "$LATEST_RUN" \
  --baseline-detail "$BASELINE_DETAIL"

echo "[SYCLForge suite] done"
