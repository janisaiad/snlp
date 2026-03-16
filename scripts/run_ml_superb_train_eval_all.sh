#!/usr/bin/env bash
# Run training + eval for all frontends (HuBERT, JEPA minimal, WavJEPA) — full debug first, then optional full.
# Usage (from snlp repo root):
#   ./scripts/run_ml_superb_train_eval_all.sh --debug          # quick: 1 ep, 2 iters each (verify pipeline)
#   ./scripts/run_ml_superb_train_eval_all.sh --full          # full: 30 ep HuBERT/JEPA, 30 ep WavJEPA (~2h each)
#   ./scripts/run_ml_superb_train_eval_all.sh --skip-data --no-sync   # skip data prep and uv sync
set -e
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
LANGS="${MLSUPERB_LANGS:-eng1}"
DURATIONS="${MLSUPERB_DURATIONS:-10min}"
SKIP_DATA=false
DO_SYNC=true
MODE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --debug)      MODE=debug; shift ;;
    --full)       MODE=full; shift ;;
    --skip-data)  SKIP_DATA=true; shift ;;
    --no-sync)    DO_SYNC=false; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [ -z "${MODE}" ]; then
  echo "Usage: $0 --debug | --full [--skip-data] [--no-sync]"
  echo "  --debug: 1 epoch, 2 iters per config (quick sanity check)"
  echo "  --full:  30 epochs per config (long run)"
  exit 1
fi

echo "[run_ml_superb_train_eval_all] MODE=${MODE} LANGS=${LANGS} DURATIONS=${DURATIONS}"

if "${DO_SYNC}"; then
  (cd "${REPO_ROOT}" && uv sync && uv add --editable ./models/espnet)
fi

cd "${RECIPE_DIR}"
. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./db.sh || exit 1

if [ -z "${MLSUPERB:-}" ] || [ ! -d "${MLSUPERB}" ]; then
  echo "MLSUPERB=${MLSUPERB} not set or not a directory." >&2
  exit 1
fi

if ! "${SKIP_DATA}"; then
  for _dur in ${DURATIONS}; do
    for _lang in ${LANGS}; do
      ./run_one_lang.sh --single_lang "${_lang}" --duration "${_dur}" --stage 1 --stop_stage 4
    done
  done
fi

_run_extra=()
if [ "${MODE}" = "debug" ]; then
  _run_extra=(--asr_args "--max_epoch 1 --num_iters_per_epoch 2")
  echo "[run_ml_superb_train_eval_all] DEBUG: 1 epoch, 2 iters per config"
fi

# HuBERT needs S3PRL (pip/conda s3prl); skip if not available
configs=()
names=()
if python3 -c "import s3prl" 2>/dev/null; then
  configs+=("conf/tuning/train_asr_s3prl_10min.yaml")
  names+=("HuBERT")
fi
configs+=("conf/tuning/train_asr_jepa_10min.yaml" "conf/tuning/train_asr_wavjepa_10min.yaml")
names+=("JEPA_minimal" "WavJEPA")
[ ${#configs[@]} -eq 0 ] && { echo "No configs to run (install s3prl for HuBERT)." >&2; exit 1; }

for i in "${!configs[@]}"; do
  cfg="${configs[$i]}"
  name="${names[$i]}"
  for _dur in ${DURATIONS}; do
    for _lang in ${LANGS}; do
      _tag=$(basename "${cfg}" .yaml)_${_lang}_${_dur}
      echo "[run_ml_superb_train_eval_all] ${name}: ${_tag}"
      ./run_one_lang.sh \
        --single_lang "${_lang}" \
        --duration "${_dur}" \
        --asr_config "${cfg}" \
        --stage 5 \
        --stop_stage 13 \
        "${_run_extra[@]}"
    done
  done
done

echo "[run_ml_superb_train_eval_all] Done. Results: ${RECIPE_DIR}/exp/*/RESULTS.md"
echo "Compare: grep -h 'test_10min' ${RECIPE_DIR}/exp/asr_train*/RESULTS.md | head -20"
