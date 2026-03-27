#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1/exp/asr_train_asr_wavjepa_local_ckpt_10min_eng1_10min/RESULTS.md"
MAX_MIN="${1:-240}"
elapsed=0
while [ ${elapsed} -lt ${MAX_MIN} ]; do
  if [ -f "${EXP}" ]; then
    cd "${REPO_ROOT}"
    uv run python scripts/collect_asr_results.py
    echo "[wait_local_wavjepa_then_collect] done"
    exit 0
  fi
  sleep 120
  elapsed=$((elapsed+2))
done
echo "[wait_local_wavjepa_then_collect] timeout"
exit 1
