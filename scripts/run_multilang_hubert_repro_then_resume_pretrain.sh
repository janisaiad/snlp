#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export MLSUPERB="${MLSUPERB:-${REPO_ROOT}/data/ml_superb}"

echo "[repro] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[repro] MLSUPERB=${MLSUPERB}"

langs=(eng1 fra1 deu1)
durs=(10min 1h)

cd "${RECIPE}"
. ./path.sh
. ./cmd.sh
. ./db.sh

for dur in "${durs[@]}"; do
  if [ "${dur}" = "10min" ]; then
    cfg="conf/tuning/train_asr_s3prl_10min.yaml"
  else
    cfg="conf/tuning/train_asr_s3prl_1h.yaml"
  fi
  for lang in "${langs[@]}"; do
    echo "[repro] ${lang} ${dur} cfg=${cfg}"
    ./run_one_lang.sh \
      --single_lang "${lang}" \
      --duration "${dur}" \
      --asr_config "${cfg}" \
      --stage 5 \
      --stop_stage 13 \
      > "${LOG_DIR}/repro_${lang}_${dur}.log" 2>&1
  done
done

echo "[repro] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[repro] resuming JEPA pretraining"
nohup "${REPO_ROOT}/scripts/run_wavjepa_pretrain.sh" --num-gpus 1 --data audioset > "${LOG_DIR}/wavjepa_pretrain_resumed_after_repro.log" 2>&1 &
echo "[repro] launched JEPA pretrain resume"
