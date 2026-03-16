#!/usr/bin/env bash
# Run ML-SUPERB with WavJEPA-Nat pretrained frontend (labhamlet/wavjepa-nat-base).
# Usage (from snlp repo root):
#   ./scripts/run_ml_superb_wavjepa.sh
#   ./scripts/run_ml_superb_wavjepa.sh --skip-data
#   ./scripts/run_ml_superb_wavjepa.sh --debug
set -e
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
LANGS="${MLSUPERB_JEPA_LANGS:-eng1}"
DURATIONS="${MLSUPERB_JEPA_DURATIONS:-10min}"
SKIP_DATA=false
DEBUG=false
DO_SYNC=true

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-data) SKIP_DATA=true; shift ;;
    --debug)     DEBUG=true; shift ;;
    --no-sync)   DO_SYNC=false; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "[run_ml_superb_wavjepa] REPO_ROOT=${REPO_ROOT} LANGS=${LANGS} DURATIONS=${DURATIONS}"

if "${DO_SYNC}"; then
  echo "[run_ml_superb_wavjepa] Syncing (uv sync, espnet editable)..."
  (cd "${REPO_ROOT}" && uv sync && uv add --editable ./models/espnet)
fi

cd "${RECIPE_DIR}"
. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./db.sh || exit 1

if [ -z "${MLSUPERB:-}" ] || [ ! -d "${MLSUPERB}" ]; then
  echo "[run_ml_superb_wavjepa] MLSUPERB=${MLSUPERB} missing or not a directory." >&2
  exit 1
fi

if ! "${SKIP_DATA}"; then
  for _dur in ${DURATIONS}; do
    for _lang in ${LANGS}; do
      echo "[run_ml_superb_wavjepa] Data prep ${_lang} ${_dur}..."
      ./run_one_lang.sh --single_lang "${_lang}" --duration "${_dur}" --stage 1 --stop_stage 4
    done
  done
fi

WAVJEPA_CONFIG="conf/tuning/train_asr_wavjepa_10min.yaml"
_run_extra=()
if "${DEBUG}"; then
  _run_extra+=(--asr_args "--max_epoch 1 --num_iters_per_epoch 2")
  echo "[run_ml_superb_wavjepa] DEBUG: 1 epoch, 2 iters"
fi

for _dur in ${DURATIONS}; do
  for _lang in ${LANGS}; do
    _tag="asr_train_asr_wavjepa_10min_${_lang}_${_dur}"
    echo "[run_ml_superb_wavjepa] Training + decode ${_lang} ${_dur} -> exp/${_tag}"
    ./run_one_lang.sh \
      --single_lang "${_lang}" \
      --duration "${_dur}" \
      --asr_config "${WAVJEPA_CONFIG}" \
      --stage 5 \
      --stop_stage 13 \
      "${_run_extra[@]}"
  done
done

echo "[run_ml_superb_wavjepa] Done. Check ${RECIPE_DIR}/exp/ for RESULTS.md"
