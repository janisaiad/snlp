#!/usr/bin/env bash
# Orchestre les benchmarks ML-SUPERB / WavJEPA : sanity rapide puis file longue en arrière-plan.
# « Tous les benchmarks » = tout ce que ce dépôt expose (pas les 143 langues paper complètes).
#
# Usage (racine snlp) :
#   ./scripts/run_benchmark_suite_all.sh                # quick + nohup chaîne longue
#   ./scripts/run_benchmark_suite_all.sh --quick        # uniquement debug + collect + ABX
#   ./scripts/run_benchmark_suite_all.sh --long         # chaîne longue au premier plan (pas nohup)
#   ./scripts/run_benchmark_suite_all.sh --long-bg       # uniquement lancer la chaîne longue en nohup (sans quick)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export MLSUPERB="${MLSUPERB:-${REPO_ROOT}/data/ml_superb}"

IDENT="Data=AudioSet/Extractor=wavjepa/InSeconds=2.01/BatchSize=32/NrSamples=8/NrGPUs=1/LR=0.0004/TargetProb=0.25/TargetLen=10/ContextProb=0.65/ContextLen=10/MinContextBlock=1/ContextRatio=0.1"
DEFAULT_CKPT="${REPO_ROOT}/logs/wavjepa_pretrain/saved_models_jepa_new_masking/${IDENT}/last.ckpt"
export WAVJEPA_LOCAL_CKPT="${WAVJEPA_LOCAL_CKPT:-${DEFAULT_CKPT}}"

RUN_QUICK=true
RUN_LONG=true
LONG_FG=false
LONG_BG=true

while [ $# -gt 0 ]; do
  case "$1" in
    --quick)    RUN_QUICK=true; RUN_LONG=false; shift ;;
    --long)     RUN_QUICK=false; RUN_LONG=true; LONG_FG=true; LONG_BG=false; shift ;;
    --long-bg)  RUN_QUICK=false; RUN_LONG=true; LONG_FG=false; LONG_BG=true; shift ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

LOGROOT="${REPO_ROOT}/logs/benchmark_suite_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${LOGROOT}"
LONG_SCRIPT="${REPO_ROOT}/scripts/run_benchmark_suite_long.sh"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOGROOT}/master.log"
}

clean_mono_eng1_10min() {
  local R="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
  log "Nettoyage exp eng1/10min (évite resume CTC vocab incompatible)"
  rm -rf "${R}/exp/asr_train_asr_s3prl_10min_eng1_10min" \
    "${R}/exp/asr_train_asr_jepa_10min_eng1_10min" \
    "${R}/exp/asr_train_asr_wavjepa_10min_eng1_10min" \
    "${R}/exp/asr_stats_eng1_10min"
}

run_quick() {
  clean_mono_eng1_10min
  log "=== Phase rapide : train_eval_all --debug ==="
  (cd "${REPO_ROOT}" && ./scripts/run_ml_superb_train_eval_all.sh --debug --skip-data --no-sync) \
    2>&1 | tee "${LOGROOT}/01_train_eval_all_debug.log"

  log "=== Phase rapide : WavJEPA local --debug (CKPT=${WAVJEPA_LOCAL_CKPT}) ==="
  if [ -f "${WAVJEPA_LOCAL_CKPT}" ]; then
    (cd "${REPO_ROOT}" && ./scripts/run_asr_wavjepa_local_benchmark.sh --debug) \
      2>&1 | tee "${LOGROOT}/02_wavjepa_local_debug.log"
  else
    log "SKIP local WavJEPA debug : ckpt absent"
  fi

  log "=== collect_asr_results.py ==="
  (cd "${REPO_ROOT}" && uv run python scripts/collect_asr_results.py) \
    2>&1 | tee "${LOGROOT}/03_collect_eng1.log" || log "collect exit non-zero"

  log "=== ABX vs ASR (dev_10min) ==="
  if [ -f "${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1/data/dev_10min/wav.scp" ]; then
    (cd "${REPO_ROOT}" && uv run python scripts/run_abx_vs_asr.py \
      --recipe_dir models/espnet/egs2/ml_superb/asr1 \
      --data_name dev_10min) \
      2>&1 | tee "${LOGROOT}/04_abx_vs_dev10min.log" || log "ABX a échoué (ex. fastabx/torchdtw)"
  else
    log "SKIP ABX : dev_10min absent"
  fi
  log "=== Fin phase rapide ==="
}

if "${RUN_QUICK}"; then
  run_quick
fi

if "${RUN_LONG}"; then
  if "${LONG_BG}"; then
    log "Lancement nohup : ${LONG_SCRIPT} ${LOGROOT}"
    nohup bash "${LONG_SCRIPT}" "${LOGROOT}" >> "${LOGROOT}/long_runner.out" 2>&1 &
    echo $! > "${LOGROOT}/long_chain.pid"
    log "PID chaîne longue : $(cat "${LOGROOT}/long_chain.pid") | tail -f ${LOGROOT}/long_runner.out"
  else
    log "Chaîne longue au premier plan"
    bash "${LONG_SCRIPT}" "${LOGROOT}"
  fi
fi

log "Répertoire logs : ${LOGROOT}"
echo "${LOGROOT}"
