#!/usr/bin/env bash
# Run ML-SUPERB SSL (HuBERT frozen + CTC) experiments: data prep + training + decode
# for multiple languages and durations. Satisfies: "pretrained SSL (HuBERT/wav2vec),
# freeze its parameters, ASR with CTC, 10 min / 1h" (project supervisor + rendu1).
#
# Usage (from snlp repo root):
#   ./scripts/run_ml_superb_ssl_experiments.sh
#   ./scripts/run_ml_superb_ssl_experiments.sh --langs "eng1 fra1 deu1" --durations "10min 1h"
#   ./scripts/run_ml_superb_ssl_experiments.sh --skip-data   # data already prepared, only train+decode
#   ./scripts/run_ml_superb_ssl_experiments.sh --dry-run
#
# Prerequisites: uv, MLSUPERB set or default data/ml_superb (see refs/REPRODUCTION.md).
# Optional: GPU for faster training.
set -e
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"

# default: one language, 10min (quick validation); extend with --langs / --durations
LANGS="eng1"
DURATIONS="10min"
SKIP_DATA=false
DRY_RUN=false
DO_SYNC=true
# debug: 1 epoch, 2 iters/epoch for fast pipeline check (you run full training yourself)
DEBUG=false
ASR_EXTRA_ARGS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --langs)        LANGS="$2"; shift 2 ;;
    --durations)    DURATIONS="$2"; shift 2 ;;
    --skip-data)    SKIP_DATA=true; shift ;;
    --dry-run)      DRY_RUN=true; shift ;;
    --no-sync)      DO_SYNC=false; shift ;;
    --debug)        DEBUG=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if "${DEBUG}"; then
  ASR_EXTRA_ARGS="--max_epoch 1 --num_iters_per_epoch 2"
  echo "[run_ml_superb_ssl] DEBUG mode: asr_args=${ASR_EXTRA_ARGS}"
fi

echo "[run_ml_superb_ssl] REPO_ROOT=${REPO_ROOT}"
echo "[run_ml_superb_ssl] RECIPE_DIR=${RECIPE_DIR}"
echo "[run_ml_superb_ssl] LANGS=${LANGS} DURATIONS=${DURATIONS} SKIP_DATA=${SKIP_DATA} DRY_RUN=${DRY_RUN}"

if [ ! -f "${RECIPE_DIR}/asr.sh" ]; then
  echo "Recipe not found at ${RECIPE_DIR}. Ensure models/espnet is present." >&2
  exit 1
fi

if "${DO_SYNC}" && ! "${DRY_RUN}"; then
  echo "[run_ml_superb_ssl] Syncing env (uv sync, espnet editable)..."
  (cd "${REPO_ROOT}" && uv sync && uv add --editable ./models/espnet)
fi

cd "${RECIPE_DIR}"
. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./db.sh || exit 1

if [ -z "${MLSUPERB:-}" ]; then
  echo "MLSUPERB is not set. Set it or ensure db.sh default (data/ml_superb) exists." >&2
  exit 1
fi
echo "[run_ml_superb_ssl] MLSUPERB=${MLSUPERB}"

# Phase 1: data preparation for each (lang, duration)
if ! "${SKIP_DATA}"; then
  for _dur in ${DURATIONS}; do
    for _lang in ${LANGS}; do
      _tag="[data ${_lang} ${_dur}]"
      echo "${_tag} Running stages 1-4 (data prep)..."
      if "${DRY_RUN}"; then
        echo "${_tag} (dry-run) would run: ./run_one_lang.sh --single_lang ${_lang} --duration ${_dur} --stage 1 --stop_stage 4"
      else
        ./run_one_lang.sh --single_lang "${_lang}" --duration "${_dur}" --stage 1 --stop_stage 4
      fi
    done
  done
else
  echo "[run_ml_superb_ssl] Skipping data prep (--skip-data)."
fi

# Phase 2: SSL training + decode for each (lang, duration)
# Config: frozen HuBERT (frontend.upstream) + CTC; 10min -> train_asr_s3prl_10min.yaml, 1h -> train_asr_s3prl_1h.yaml
for _dur in ${DURATIONS}; do
  if [ "${_dur}" = "10min" ]; then
    _asr_config="conf/tuning/train_asr_s3prl_10min.yaml"
  else
    _asr_config="conf/tuning/train_asr_s3prl_1h.yaml"
  fi
  for _lang in ${LANGS}; do
    _tag="[SSL ${_lang} ${_dur}]"
    _exp_tag="$(basename "${_asr_config}" .yaml)_${_lang}_${_dur}"
    echo "${_tag} Training + decode (${_asr_config}) -> exp/${_exp_tag}"
    if "${DRY_RUN}"; then
      echo "${_tag} (dry-run) would run: ./run_one_lang.sh --single_lang ${_lang} --duration ${_dur} --asr_config ${_asr_config} --stage 5 --stop_stage 13${ASR_EXTRA_ARGS:+ --asr_args \"${ASR_EXTRA_ARGS}\"}"
    else
      _run_extra=()
      [ -n "${ASR_EXTRA_ARGS}" ] && _run_extra+=(--asr_args "${ASR_EXTRA_ARGS}")
      ./run_one_lang.sh \
        --single_lang "${_lang}" \
        --duration "${_dur}" \
        --asr_config "${_asr_config}" \
        --stage 5 \
        --stop_stage 13 \
        "${_run_extra[@]}"
    fi
  done
done

echo "[run_ml_superb_ssl] Done. Experiments under ${RECIPE_DIR}/exp/"
echo "  To watch training: tail -f ${RECIPE_DIR}/exp/<asr_tag>/train.log"
echo "  SSL configs use freeze_param: [frontend.upstream] and ctc_weight: 1.0 (HuBERT frozen + CTC)."
