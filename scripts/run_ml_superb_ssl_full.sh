#!/usr/bin/env bash
# Full SSL grid: 3 languages x 2 durations (10min + 1h), data prep + train + decode + score.
# Use this for the "big run" on full ML-SUPERB data.
#
# Usage (from snlp repo root):
#   ./scripts/run_ml_superb_ssl_full.sh              # data prep + 6 SSL runs
#   ./scripts/run_ml_superb_ssl_full.sh --skip-data # data already prepared, only train+decode
#
# Prerequisites:
#   - Full ML-SUPERB data at data/ml_superb (or MLSUPERB); see refs/REPRODUCTION.md.
#   - uv, and once: uv sync, uv add --editable ./models/espnet, uv add s3prl.
# This script applies the s3prl patch once, then runs the grid.
#
# Jobs run in order: eng1 10min, fra1 10min, deu1 10min, eng1 1h, fra1 1h, deu1 1h.
# Each training is 30 epochs (~1.5 h for 10min, longer for 1h). Run in tmux/screen or nohup.
set -e
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
cd "${REPO_ROOT}"

# detect which languages have data (eng1->mls/eng, fra1->voxforge/fra, deu1->swc/deu)
. "${RECIPE_DIR}/db.sh" || true
AVAILABLE_LANGS=""
for _lang in eng1 fra1 deu1; do
  case "${_lang}" in
    eng1) _dir="${MLSUPERB}/mls/eng" ;;
    fra1) _dir="${MLSUPERB}/voxforge/fra" ;;
    deu1) _dir="${MLSUPERB}/swc/deu" ;;
    *) continue ;;
  esac
  if [ -f "${_dir}/transcript_10min_train.txt" ]; then
    AVAILABLE_LANGS="${AVAILABLE_LANGS} ${_lang}"
  fi
done
AVAILABLE_LANGS=$(echo ${AVAILABLE_LANGS})
if [ -z "${AVAILABLE_LANGS}" ]; then
  echo "[run_ml_superb_ssl_full] No data found for eng1/fra1/deu1 under MLSUPERB=${MLSUPERB}" >&2
  echo "  eng1 needs: mls/eng,  fra1: voxforge/fra,  deu1: swc/deu" >&2
  exit 1
fi
echo "[run_ml_superb_ssl_full] Languages with data: ${AVAILABLE_LANGS}"

echo "[run_ml_superb_ssl_full] Applying s3prl patch (idempotent)..."
uv run python scripts/patch_s3prl_for_ssl.py

echo "[run_ml_superb_ssl_full] Starting grid: ${AVAILABLE_LANGS} x 10min 1h"
exec ./scripts/run_ml_superb_ssl_experiments.sh \
  --langs "${AVAILABLE_LANGS}" \
  --durations "10min 1h" \
  --no-sync \
  "$@"
