#!/usr/bin/env bash
# ASR ML-SUPERB eng1 10 min avec checkpoint local WavJEPA *small* (384-d).
# Usage : WAVJEPA_LOCAL_CKPT=/path/to/last.ckpt ./scripts/run_asr_wavjepa_small_local_benchmark.sh [--debug]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
DEBUG=false
while [ $# -gt 0 ]; do
  case "$1" in
    --debug) DEBUG=true; shift ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export MLSUPERB="${MLSUPERB:-${REPO_ROOT}/data/ml_superb}"

if [ -z "${WAVJEPA_LOCAL_CKPT:-}" ] || [ ! -f "${WAVJEPA_LOCAL_CKPT}" ]; then
  echo "Définir WAVJEPA_LOCAL_CKPT vers un .ckpt small (ex. last.ckpt du run research_small)." >&2
  exit 1
fi

CFG_SRC="${RECIPE}/conf/tuning/train_asr_wavjepa_local_small_10min.yaml"
CFG_RUN="${RECIPE}/conf/tuning/train_asr_wavjepa_local_small_ckpt_10min.yaml"
sed "s|WAVJEPA_LOCAL_CKPT_PLACEHOLDER|${WAVJEPA_LOCAL_CKPT}|g" "${CFG_SRC}" > "${CFG_RUN}"

cd "${RECIPE}"
. ./path.sh
. ./cmd.sh
. ./db.sh

_extra=()
if "${DEBUG}"; then
  _extra=(--asr_args "--max_epoch 1 --num_iters_per_epoch 2")
fi

echo "[small_local_benchmark] CKPT=${WAVJEPA_LOCAL_CKPT}"
rm -rf exp/asr_train_asr_wavjepa_local_small_ckpt_10min_eng1_10min exp/asr_stats_eng1_10min
./run_one_lang.sh \
  --single_lang eng1 \
  --duration 10min \
  --asr_config conf/tuning/train_asr_wavjepa_local_small_ckpt_10min.yaml \
  --stage 5 \
  --stop_stage 13 \
  "${_extra[@]}"

echo "Results: ${RECIPE}/exp/asr_train_asr_wavjepa_local_small_ckpt_10min_eng1_10min/RESULTS.md"
