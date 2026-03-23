#!/usr/bin/env bash
# Wait for the full 30ep ASR benchmark to produce all three RESULTS.md, then run collect_asr_results.py.
# Usage: ./scripts/wait_asr_benchmark_then_collect.sh [max_wait_minutes]
# Default max_wait: 480 (8 hours). Run from repo root.
set -e
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1/exp"
MAX_WAIT="${1:-480}"
POLL_INTERVAL=120
elapsed=0

check_dirs=(
  "asr_train_asr_s3prl_10min_eng1_10min"
  "asr_train_asr_jepa_10min_eng1_10min"
  "asr_train_asr_wavjepa_10min_eng1_10min"
)

all_done() {
  for d in "${check_dirs[@]}"; do
    [ -f "${EXP_DIR}/${d}/RESULTS.md" ] || return 1
  done
  return 0
}

cd "${REPO_ROOT}"
echo "[wait_asr_benchmark_then_collect] Waiting up to ${MAX_WAIT} minutes for all three RESULTS.md (poll every ${POLL_INTERVAL}s)"
while [ $elapsed -lt $MAX_WAIT ]; do
  if all_done; then
    echo "[wait_asr_benchmark_then_collect] All RESULTS.md found. Running collector."
    uv run python scripts/collect_asr_results.py
    echo "[wait_asr_benchmark_then_collect] Done. Table: refs/ASR_RESULTS_TABLE.md"
    exit 0
  fi
  sleep $POLL_INTERVAL
  elapsed=$((elapsed + POLL_INTERVAL / 60))
  echo "[wait_asr_benchmark_then_collect] ${elapsed}/${MAX_WAIT} min ..."
done
echo "[wait_asr_benchmark_then_collect] Timeout. Run collect_asr_results.py manually when benchmark finishes."
exit 1
