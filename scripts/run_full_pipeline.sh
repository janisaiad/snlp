#!/usr/bin/env bash
# One-liner pipeline: sync deps → download ML-SUPERB data → data prep → train all frontends → eval → summary.
# For big GPU instances: run from repo root; data goes to data/ml_superb, results to exp/ and logs/.
#
# Usage (from snlp repo root):
#   ./scripts/run_full_pipeline.sh                    # full: download + 30ep train + eval
#   ./scripts/run_full_pipeline.sh --debug             # quick: download + 1ep 2iters (sanity check)
#   ./scripts/run_full_pipeline.sh --skip-download     # use existing data (e.g. data/ml_superb already there)
#   ./scripts/run_full_pipeline.sh --skip-data         # skip data prep (data already prepared)
#   ./scripts/run_full_pipeline.sh --no-sync           # skip uv sync (faster reruns)
#
# Prerequisites: uv, bash, unzip; Hugging Face token if dataset is gated (login: huggingface-cli login).
set -e
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
LOG_DIR="${REPO_ROOT}/logs"
DATA_ROOT="${REPO_ROOT}/data/ml_superb"

SKIP_DOWNLOAD=false
SKIP_DATA=false
DO_SYNC=true
MODE=full

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --skip-data)     SKIP_DATA=true; shift ;;
    --debug)         MODE=debug; shift ;;
    --no-sync)       DO_SYNC=false; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "[run_full_pipeline] REPO_ROOT=${REPO_ROOT}"
echo "[run_full_pipeline] MODE=${MODE} SKIP_DOWNLOAD=${SKIP_DOWNLOAD} SKIP_DATA=${SKIP_DATA} DO_SYNC=${DO_SYNC}"

# 1) Sync dependencies (unless --no-sync)
if "${DO_SYNC}"; then
  echo "[run_full_pipeline] Syncing (uv sync, espnet editable)..."
  (cd "${REPO_ROOT}" && uv sync && uv add --editable ./models/espnet)
fi

# 2) Download ML-SUPERB data (unless --skip-download)
if ! "${SKIP_DOWNLOAD}"; then
  echo "[run_full_pipeline] Downloading ML-SUPERB data to ${DATA_ROOT}..."
  mkdir -p "${DATA_ROOT}"
  if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    set +u
    . "${REPO_ROOT}/.venv/bin/activate"
    set -u
  fi
  export MLSUPERB="${DATA_ROOT}"
  "${REPO_ROOT}/scripts/download_mlsuperb_data.sh" || {
    echo "[run_full_pipeline] Download failed. Use --skip-download if data is already at ${DATA_ROOT}" >&2
    exit 1
  }
  if [ ! -d "${DATA_ROOT}/mls" ]; then
    echo "[run_full_pipeline] Expected ${DATA_ROOT}/mls after download. Aborting." >&2
    exit 1
  fi
  echo "[run_full_pipeline] Data ready at ${DATA_ROOT}"
else
  if [ ! -d "${DATA_ROOT}/mls" ] && [ ! -d "${DATA_ROOT}/voxforge" ]; then
    echo "[run_full_pipeline] --skip-download but no mls/ or voxforge/ in ${DATA_ROOT}. Set MLSUPERB or run without --skip-download." >&2
    exit 1
  fi
  echo "[run_full_pipeline] Using existing data at ${DATA_ROOT}"
fi

# 3) Export so recipe and run_research_full see it
export MLSUPERB="${DATA_ROOT}"

# 4) Run full research pipeline (data prep + train + eval)
_args=("--no-sync")
"${SKIP_DATA}" && _args+=(--skip-data)
[ "${MODE}" = "debug" ] && _args+=(--debug)

echo "[run_full_pipeline] Running research pipeline: ${REPO_ROOT}/scripts/run_research_full.sh ${_args[*]}"
"${REPO_ROOT}/scripts/run_research_full.sh" "${_args[@]}"

echo "[run_full_pipeline] Done. Results: ${RECIPE_DIR}/exp/*/RESULTS.md, summary: ${LOG_DIR}/research_results_*.txt"
