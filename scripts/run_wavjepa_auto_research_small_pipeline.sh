#!/usr/bin/env bash
# Pipeline « auto research » : préentraînement small from-scratch → benchmark ASR local (debug par défaut).
# Le préentraînement complet AudioSet est long ; utiliser --smoke pour une passe courte de bout en bout.
#
# Usage (racine snlp) :
#   ./scripts/run_wavjepa_auto_research_small_pipeline.sh --smoke
#   ./scripts/run_wavjepa_auto_research_small_pipeline.sh              # long (50k steps AudioSet)
#   nohup ./scripts/run_wavjepa_auto_research_small_pipeline.sh > logs/auto_small.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE=false
FULL_ASR=false
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=true ;;
    --full-asr) FULL_ASR=true ;;
  esac
done

LOGROOT="${REPO_ROOT}/logs/wavjepa_auto_research_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${LOGROOT}"
MASTER="${LOGROOT}/pipeline.log"

{
  echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) pipeline small from-scratch SMOKE=${SMOKE} ===="
  export SAVE_DIR="${LOGROOT}/pretrain"
  rm -rf "${SAVE_DIR}"
  mkdir -p "${SAVE_DIR}"
  if "${SMOKE}"; then
    "${REPO_ROOT}/scripts/run_wavjepa_research_small_fromscratch.sh" --smoke
  else
    "${REPO_ROOT}/scripts/run_wavjepa_research_small_fromscratch.sh"
  fi

  CKPT=""
  if [[ -d "${SAVE_DIR}/saved_models_jepa_new_masking" ]]; then
    CKPT="$(find "${SAVE_DIR}/saved_models_jepa_new_masking" -name 'step=*.ckpt' 2>/dev/null | sort -V | tail -1 || true)"
  fi
  if [[ -z "${CKPT}" ]]; then
    CKPT="$(find "${SAVE_DIR}/saved_models_jepa_new_masking" -name 'last.ckpt' 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "${CKPT}" ]] || [[ ! -f "${CKPT}" ]]; then
    echo "Aucun checkpoint trouvé sous ${SAVE_DIR}" >&2
    exit 1
  fi
  echo "Using CKPT=${CKPT}"
  export WAVJEPA_LOCAL_CKPT="${CKPT}"
  if "${FULL_ASR}"; then
    "${REPO_ROOT}/scripts/run_asr_wavjepa_small_local_benchmark.sh"
  else
    "${REPO_ROOT}/scripts/run_asr_wavjepa_small_local_benchmark.sh" --debug
  fi
  echo "==== fin pipeline ; logs ${LOGROOT} ===="
} 2>&1 | tee "${MASTER}"

echo "${LOGROOT}"
