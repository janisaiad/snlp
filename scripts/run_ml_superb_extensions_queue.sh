#!/usr/bin/env bash
# we queue optional ML-SUPERB extension runs in phases to avoid overlapping GPU jobs
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASR1="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
LOG_ROOT="${REPO_ROOT}/logs/ml_superb_extensions"
mkdir -p "${LOG_ROOT}"

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export MLSUPERB="${MLSUPERB:-${REPO_ROOT}/data/ml_superb}"

PHASE="${PHASE:-help}"
DURATIONS="${DURATIONS:-10min 1h}"
# we default to a small subset so accidental full run does not burn weeks
LANG_SUBSET="${LANG_SUBSET:-eng1 fra1 deu1}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG_ROOT}/master.log"
}

run_one_lang_phased() {
  local lang="$1"
  local duration="$2"
  local cfg="$3"
  local tag="$4"
  local log="${LOG_ROOT}/mono_${tag}_${lang}_${duration}.log"
  log "START mono ${tag} ${lang} ${duration} -> ${log}"
  (
    cd "${ASR1}"
    . ./path.sh
    . ./cmd.sh
    . ./db.sh
    ./run_one_lang.sh \
      --single_lang "${lang}" \
      --duration "${duration}" \
      --asr_config "${cfg}" \
      --stage 1 \
      --stop_stage 13
  ) > "${log}" 2>&1
  log "DONE mono ${tag} ${lang} ${duration}"
}

run_multi_phased() {
  local duration="$1"
  local cfg="$2"
  local lid="$3"
  local only_lid="$4"
  local tag="$5"
  local log="${LOG_ROOT}/multi_${tag}_dur-${duration}_lid-${lid}_only-${only_lid}.log"
  log "START multi ${tag} ${duration} lid=${lid} only_lid=${only_lid} -> ${log}"
  (
    cd "${ASR1}"
    . ./path.sh
    . ./cmd.sh
    . ./db.sh
    ./run_multi.sh \
      --duration "${duration}" \
      --asr_config "${cfg}" \
      --stage 1 \
      --stop_stage 13 \
      --lid "${lid}" \
      --only_lid "${only_lid}"
  ) > "${log}" 2>&1
  log "DONE multi ${tag} ${duration} lid=${lid} only_lid=${only_lid}"
}

case "${PHASE}" in
  help)
    cat <<EOF
Usage: PHASE=<name> [DURATIONS="10min 1h"] [LANG_SUBSET="eng1 fra1 deu1"] $0

Phases (run sequentially on one GPU recommended):
  mono_baseline_ssl   HuBERT frozen, run_one_lang for each LANG_SUBSET x DURATIONS
  mono_fbank          Fbank baseline, same grid
  multi_asr           Multilingual ASR (run_multi.sh, lid=false only_lid=false)
  lid_only            LID-only track (only_lid=true)
  asr_plus_lid        Joint ASR+LID (lid=true only_lid=false)
  peft_lora           LoRA on LANG_SUBSET x DURATIONS (long; uses adapter config)
  peft_houlsby        Houlsby on LANG_SUBSET x DURATIONS

Examples:
  PHASE=multi_asr DURATIONS="10min" $0
  PHASE=mono_baseline_ssl LANG_SUBSET="eng1 jpn" DURATIONS="10min 1h" $0

Logs: ${LOG_ROOT}/
EOF
    exit 0
    ;;

  mono_baseline_ssl)
    for dur in ${DURATIONS}; do
      for lang in ${LANG_SUBSET}; do
        run_one_lang_phased "${lang}" "${dur}" "conf/tuning/train_asr_s3prl_${dur}.yaml" "s3prl"
      done
    done
    ;;

  mono_fbank)
    for dur in ${DURATIONS}; do
      for lang in ${LANG_SUBSET}; do
        run_one_lang_phased "${lang}" "${dur}" "conf/tuning/train_asr_fbank_${dur}.yaml" "fbank"
      done
    done
    ;;

  multi_asr)
    for dur in ${DURATIONS}; do
      run_multi_phased "${dur}" "conf/tuning/train_asr_s3prl_${dur}.yaml" "false" "false" "asr"
    done
    ;;

  lid_only)
    for dur in ${DURATIONS}; do
      run_multi_phased "${dur}" "conf/tuning/train_asr_s3prl_${dur}.yaml" "false" "true" "lid_only"
    done
    ;;

  asr_plus_lid)
    for dur in ${DURATIONS}; do
      run_multi_phased "${dur}" "conf/tuning/train_asr_s3prl_${dur}.yaml" "true" "false" "asr_lid"
    done
    ;;

  peft_lora)
    for dur in ${DURATIONS}; do
      for lang in ${LANG_SUBSET}; do
        run_one_lang_phased "${lang}" "${dur}" "conf/tuning/train_asr_s3prl_lora.yaml" "lora"
      done
    done
    ;;

  peft_houlsby)
    for dur in ${DURATIONS}; do
      for lang in ${LANG_SUBSET}; do
        run_one_lang_phased "${lang}" "${dur}" "conf/tuning/train_asr_s3prl_houlsby.yaml" "houlsby"
      done
    done
    ;;

  *)
    log "unknown PHASE=${PHASE}"
    exit 1
    ;;
esac

log "PHASE ${PHASE} completed."
