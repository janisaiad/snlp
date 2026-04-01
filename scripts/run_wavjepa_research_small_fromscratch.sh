#!/usr/bin/env bash
# Préentraînement WavJEPA *from scratch* : encodeur small (6 couches, d_model=384), nouveau save_dir (pas de last.ckpt).
# Données : AudioSet HF par défaut ; --smoke pour valider le graphe sans gros téléchargement.
#
# Usage (racine snlp) :
#   ./scripts/run_wavjepa_research_small_fromscratch.sh
#   ./scripts/run_wavjepa_research_small_fromscratch.sh --smoke
#   SAVE_DIR=/tmp/jepa_small ./scripts/run_wavjepa_research_small_fromscratch.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date -u +%Y%m%dT%H%MZ)"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/logs/wavjepa_research_small_fromscratch_${TS}}"
SMOKE=false
for a in "$@"; do
  [[ "$a" == "--smoke" ]] && SMOKE=true
done

mkdir -p "${SAVE_DIR}"
echo "[research_small_fromscratch] SAVE_DIR=${SAVE_DIR} SMOKE=${SMOKE}"

# From scratch : refuser un répertoire qui contient déjà un last.ckpt (sinon ce serait un resume).
if find "${SAVE_DIR}" -maxdepth 6 -name "last.ckpt" 2>/dev/null | grep -q .; then
  echo "[research_small_fromscratch] last.ckpt déjà présent sous ${SAVE_DIR} — vide ce dossier ou change SAVE_DIR." >&2
  exit 1
fi

EXTRA_HYDRA=(trainer=research_small_wavjepa)
if "${SMOKE}"; then
  echo "[research_small_fromscratch] Mode smoke (~800 steps)."
  "${REPO_ROOT}/scripts/run_wavjepa_pretrain.sh" --num-gpus 1 --data smoke --save-dir "${SAVE_DIR}" \
    "${EXTRA_HYDRA[@]}" trainer.steps=800
else
  "${REPO_ROOT}/scripts/run_wavjepa_pretrain.sh" --num-gpus 1 --data audioset --save-dir "${SAVE_DIR}" \
    "${EXTRA_HYDRA[@]}"
fi

echo "[research_small_fromscratch] Terminé. Checkpoints : ${SAVE_DIR}/saved_models_jepa_new_masking/"
echo "[research_small_fromscratch] Pour ASR downstream : export WAVJEPA_LOCAL_CKPT=.../last.ckpt && ./scripts/run_asr_wavjepa_small_local_benchmark.sh"
