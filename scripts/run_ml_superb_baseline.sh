#!/usr/bin/env bash
# Single entrypoint: sync env, set data path, run ML-SUPERB asr1 (data prep + train + decode + score).
# Usage (from repo root, e.g. on a GPU instance):
#   ./scripts/run_ml_superb_baseline.sh
#   ./scripts/run_ml_superb_baseline.sh --single_lang fra1 --duration 10min
#   ./scripts/run_ml_superb_baseline.sh --no-sync --single_lang eng1
# Requires: uv, (optional) GPU. Data: set MLSUPERB or place data in data/ml_superb (see REPRODUCTION.md).
set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASR1="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"

# default data dir (relative to repo root so it works from any cwd when we cd to asr1 later we set MLSUPERB absolute)
if [ -z "${MLSUPERB:-}" ]; then
  export MLSUPERB="${REPO_ROOT}/data/ml_superb"
fi

do_sync=true
run_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-sync)
      do_sync=false
      shift
      ;;
    *)
      run_args+=("$1")
      shift
      ;;
  esac
done

cd "${REPO_ROOT}"
if "${do_sync}"; then
  echo "Syncing environment (uv sync, espnet editable)..."
  uv sync
  uv add --editable ./models/espnet
fi

if [ ! -d "${ASR1}" ]; then
  echo "Error: recipe dir not found: ${ASR1}" 1>&2
  exit 1
fi

echo "Running ML-SUPERB asr1 from ${ASR1} (MLSUPERB=${MLSUPERB})"
cd "${ASR1}"
. ./path.sh
. ./cmd.sh
. ./db.sh

./run_one_lang.sh "${run_args[@]}"
