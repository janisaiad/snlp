#!/usr/bin/env bash
# we run multilingual ML-SUPERB tracks then LoRA multilingual ASR (sequential, one GPU)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
LOG_ROOT="${REPO_ROOT}/logs/ml_superb_multilingual_peft"
mkdir -p "${LOG_ROOT}"

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export MLSUPERB="${MLSUPERB:-${REPO_ROOT}/data/ml_superb}"

ML_SUPERB_NGPU="${ML_SUPERB_NGPU:-1}"
NJ="${ML_SUPERB_NJ:-8}"
INF_NJ="${ML_SUPERB_INF_NJ:-4}"

MASTER_LOG="${LOG_ROOT}/master.log"
touch "${MASTER_LOG}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${MASTER_LOG}"
}

run_multi_job() {
  local duration="$1"
  local lid="$2"
  local only_lid="$3"
  local cfg="$4"
  local name="$5"
  local logf="${LOG_ROOT}/multi_${name}_${duration}.log"
  log "START ${name} ${duration} cfg=${cfg} lid=${lid} only_lid=${only_lid} -> ${logf}"
  (
    cd "${RECIPE}"
    . ./path.sh
    . ./cmd.sh
    . ./db.sh
    ./run_multi.sh \
      --duration "${duration}" \
      --lid "${lid}" \
      --only_lid "${only_lid}" \
      --asr_config "${cfg}" \
      --stage 1 \
      --stop_stage 13 \
      --nj "${NJ}" \
      --inference_nj "${INF_NJ}" \
      --ngpu "${ML_SUPERB_NGPU}"
  ) >> "${logf}" 2>&1
  log "DONE ${name} ${duration}"
}

log "=== ML-SUPERB multilingual + PEFT queue (NGPU=${ML_SUPERB_NGPU}) ==="

run_multi_job "10min" "false" "false" "conf/tuning/train_asr_s3prl_10min.yaml" "asr_only"
run_multi_job "1h" "false" "false" "conf/tuning/train_asr_s3prl_1h.yaml" "asr_only"

run_multi_job "10min" "false" "true" "conf/tuning/train_asr_s3prl_10min.yaml" "lid_only"
run_multi_job "1h" "false" "true" "conf/tuning/train_asr_s3prl_1h.yaml" "lid_only"

run_multi_job "10min" "true" "false" "conf/tuning/train_asr_s3prl_10min.yaml" "asr_plus_lid"
run_multi_job "1h" "true" "false" "conf/tuning/train_asr_s3prl_1h.yaml" "asr_plus_lid"

run_multi_job "10min" "false" "false" "conf/tuning/train_asr_s3prl_lora_10min.yaml" "lora_asr_only"
run_multi_job "1h" "false" "false" "conf/tuning/train_asr_s3prl_lora_1h.yaml" "lora_asr_only"

cd "${REPO_ROOT}"
uv run python scripts/collect_asr_results.py >> "${LOG_ROOT}/collect.log" 2>&1 || log "collect_asr_results.py exited non-zero (expected if tags not in script)"

log "=== queue finished ==="
