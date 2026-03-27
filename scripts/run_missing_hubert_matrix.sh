#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export MLSUPERB="${MLSUPERB:-${REPO_ROOT}/data/ml_superb}"
ML_SUPERB_NGPU="${ML_SUPERB_NGPU:-1}"
WAVJEPA_NUM_GPUS="${WAVJEPA_NUM_GPUS:-${ML_SUPERB_NGPU}}"
cd "${RECIPE}"
. ./path.sh
. ./cmd.sh
. ./db.sh
runs=(
  "eng1 1h"
  "fra1 10min"
  "fra1 1h"
  "deu1 10min"
  "deu1 1h"
)
for r in "${runs[@]}"; do
  lang="${r%% *}"
  dur="${r##* }"
  if [ "${dur}" = "10min" ]; then
    cfg="conf/tuning/train_asr_s3prl_10min.yaml"
  else
    cfg="conf/tuning/train_asr_s3prl_1h.yaml"
  fi
  prep_log="${LOG_DIR}/missing_hubert_${lang}_${dur}_prep.log"
  asr_log="${LOG_DIR}/missing_hubert_${lang}_${dur}.log"

  echo "[matrix] PREP START ${lang} ${dur} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_DIR}/run_missing_hubert_matrix.log"
  ./run_one_lang.sh \
    --single_lang "${lang}" \
    --duration "${dur}" \
    --asr_config "${cfg}" \
    --stage 1 \
    --stop_stage 4 \
    > "${prep_log}" 2>&1
  echo "[matrix] PREP DONE ${lang} ${dur} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_DIR}/run_missing_hubert_matrix.log"

  echo "[matrix] ASR START ${lang} ${dur} cfg=${cfg} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_DIR}/run_missing_hubert_matrix.log"
  ./run_one_lang.sh \
    --single_lang "${lang}" \
    --duration "${dur}" \
    --asr_config "${cfg}" \
    --stage 5 \
    --stop_stage 13 \
    > "${asr_log}" 2>&1
  echo "[matrix] ASR DONE ${lang} ${dur} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_DIR}/run_missing_hubert_matrix.log"
done
cd "${REPO_ROOT}"
uv run python scripts/collect_asr_results.py > "${LOG_DIR}/collect_asr_results_after_missing_hubert.log" 2>&1
nohup "${REPO_ROOT}/scripts/run_wavjepa_pretrain.sh" --num-gpus 1 --data audioset > "${LOG_DIR}/wavjepa_pretrain_resumed_after_hubert_matrix.log" 2>&1 &
echo "[matrix] Relaunched JEPA pretraining" | tee -a "${LOG_DIR}/run_missing_hubert_matrix.log"
