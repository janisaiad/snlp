#!/usr/bin/env bash
# Chaîne longue de benchmarks (un GPU). Appelé par run_benchmark_suite_all.sh ou à la main.
# Logs : premier argument = répertoire log (obligatoire).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGROOT="${1:-}"
if [ -z "${LOGROOT}" ] || [ ! -d "${LOGROOT}" ]; then
  echo "Usage: $0 <LOGROOT_DIR>" >&2
  exit 1
fi

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export MLSUPERB="${MLSUPERB:-${REPO_ROOT}/data/ml_superb}"

IDENT="Data=AudioSet/Extractor=wavjepa/InSeconds=2.01/BatchSize=32/NrSamples=8/NrGPUs=1/LR=0.0004/TargetProb=0.25/TargetLen=10/ContextProb=0.65/ContextLen=10/MinContextBlock=1/ContextRatio=0.1"
DEFAULT_CKPT="${REPO_ROOT}/logs/wavjepa_pretrain/saved_models_jepa_new_masking/${IDENT}/last.ckpt"
export WAVJEPA_LOCAL_CKPT="${WAVJEPA_LOCAL_CKPT:-${DEFAULT_CKPT}}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOGROOT}/master.log"
}

cd "${REPO_ROOT}"

RECIPE="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
log "Nettoyage exp eng1/10min avant grille longue (CTC / resume propres)"
rm -rf "${RECIPE}/exp/asr_train_asr_s3prl_10min_eng1_10min" \
  "${RECIPE}/exp/asr_train_asr_jepa_10min_eng1_10min" \
  "${RECIPE}/exp/asr_train_asr_wavjepa_10min_eng1_10min" \
  "${RECIPE}/exp/asr_stats_eng1_10min"

log "LONG: research_full (eng1 10min, 30 ep × 3 frontends)"
./scripts/run_research_full.sh --skip-data --no-sync 2>&1 | tee "${LOGROOT}/10_research_full_eng1_10min.log"

if [ -f "${WAVJEPA_LOCAL_CKPT}" ]; then
  log "LONG: WavJEPA local full ASR (eng1 10min, 30 ep)"
  ./scripts/run_asr_wavjepa_local_benchmark.sh 2>&1 | tee "${LOGROOT}/11_wavjepa_local_full_eng1_10min.log"
else
  log "SKIP WavJEPA local full : ckpt absent (${WAVJEPA_LOCAL_CKPT})"
fi

for _lang in fra1 deu1; do
  log "LONG: train_eval_all --full ${_lang} 10min"
  MLSUPERB_LANGS="${_lang}" MLSUPERB_DURATIONS="10min" \
    ./scripts/run_ml_superb_train_eval_all.sh --full --skip-data --no-sync \
    2>&1 | tee "${LOGROOT}/12_train_eval_all_full_${_lang}_10min.log"
done

log "LONG: train_eval_all --full eng1 1h"
MLSUPERB_LANGS="eng1" MLSUPERB_DURATIONS="1h" \
  ./scripts/run_ml_superb_train_eval_all.sh --full --skip-data --no-sync \
  2>&1 | tee "${LOGROOT}/13_train_eval_all_full_eng1_1h.log"

log "LONG: file multilingue + PEFT (queue complète dans run_ml_superb_multilingual_peft_queue.sh)"
./scripts/run_ml_superb_multilingual_peft_queue.sh 2>&1 | tee "${LOGROOT}/14_multilingual_peft_queue.log"

uv run python scripts/collect_asr_results.py 2>&1 | tee "${LOGROOT}/15_collect_final.log" || true
log "=== Chaîne longue terminée ==="
