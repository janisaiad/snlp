#!/usr/bin/env bash
# Full pipeline: optional JEPA pretraining → download ML-SUPERB → data prep → train (eng1+fra1+deu1, 30ep) → eval → update report table.
# Default: multi-lang (eng1, fra1, deu1) 10min, 30ep, then refs/rendu1.md table updated from RESULTS.md.
#
# Usage (from snlp repo root):
#   ./scripts/run_full_pipeline.sh                         # full: download + eng1/fra1/deu1 30ep + report update
#   ./scripts/run_full_pipeline.sh --pretrain-gpus 10       # + WavJEPA pretraining (e.g. overnight) then ASR
#   ./scripts/run_full_pipeline.sh --debug                  # quick: eng1 only, 1ep 2iters
#   ./scripts/run_full_pipeline.sh --skip-download           # use existing data
#   ./scripts/run_full_pipeline.sh --skip-data --no-sync    # skip data prep and uv sync
#
# Prerequisites: uv, bash, unzip; for pretraining: AudioSet (or --pretrain-data librispeech) in third_party/wavjepa configs.
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
PRETRAIN_GPUS=0
PRETRAIN_DATA=audioset
PRETRAIN_SAVE_DIR="${REPO_ROOT}/logs/wavjepa_pretrain"

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-download)   SKIP_DOWNLOAD=true; shift ;;
    --skip-data)       SKIP_DATA=true; shift ;;
    --debug)           MODE=debug; shift ;;
    --no-sync)         DO_SYNC=false; shift ;;
    --pretrain-gpus)   PRETRAIN_GPUS="$2"; shift 2 ;;
    --pretrain-data)   PRETRAIN_DATA="$2"; shift 2 ;;
    --pretrain-save-dir) PRETRAIN_SAVE_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "[run_full_pipeline] REPO_ROOT=${REPO_ROOT}"
echo "[run_full_pipeline] MODE=${MODE} SKIP_DOWNLOAD=${SKIP_DOWNLOAD} SKIP_DATA=${SKIP_DATA} DO_SYNC=${DO_SYNC} PRETRAIN_GPUS=${PRETRAIN_GPUS}"

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

# 3b) Optional: WavJEPA SSL pretraining (run first so it can run overnight)
if [ "${PRETRAIN_GPUS}" -gt 0 ]; then
  echo "[run_full_pipeline] Step: WavJEPA pretraining (${PRETRAIN_GPUS} GPUs, data=${PRETRAIN_DATA})..."
  if [ ! -d "${REPO_ROOT}/third_party/wavjepa" ] || [ ! -f "${REPO_ROOT}/third_party/wavjepa/train.py" ]; then
    "${REPO_ROOT}/scripts/setup_wavjepa.sh" || { echo "[run_full_pipeline] WavJEPA setup failed." >&2; exit 1; }
  fi
  [ -f "${REPO_ROOT}/third_party/wavjepa/requirements.txt" ] && (cd "${REPO_ROOT}" && uv pip install -r third_party/wavjepa/requirements.txt 2>/dev/null || true)
  "${REPO_ROOT}/scripts/run_wavjepa_pretrain.sh" --num-gpus "${PRETRAIN_GPUS}" --data "${PRETRAIN_DATA}" --save-dir "${PRETRAIN_SAVE_DIR}" || { echo "[run_full_pipeline] Pretraining failed." >&2; exit 1; }
  echo "[run_full_pipeline] Pretraining done. Checkpoints: ${PRETRAIN_SAVE_DIR}"
fi

# 4) Default: multi-lang (eng1, fra1, deu1) 10min for full run; eng1 only for debug. Auto-detect available langs from data.
if [ "${MODE}" = "debug" ]; then
  export MLSUPERB_LANGS="${MLSUPERB_LANGS:-eng1}"
else
  if [ -z "${MLSUPERB_LANGS:-}" ]; then
    _have=""
    [ -d "${DATA_ROOT}/mls/eng" ] && _have="${_have} eng1"
    [ -d "${DATA_ROOT}/voxforge/fra" ] && _have="${_have} fra1"
    [ -d "${DATA_ROOT}/swc/deu" ] && _have="${_have} deu1"
    _have="${_have# }"
    export MLSUPERB_LANGS="${_have:-eng1}"
    echo "[run_full_pipeline] Auto-detected languages from data: ${MLSUPERB_LANGS}"
  else
    export MLSUPERB_LANGS="${MLSUPERB_LANGS}"
  fi
fi
export MLSUPERB_DURATIONS="${MLSUPERB_DURATIONS:-10min}"
echo "[run_full_pipeline] LANGS=${MLSUPERB_LANGS} DURATIONS=${MLSUPERB_DURATIONS}"

# 5) Run full research pipeline (data prep + train + eval)
_args=("--no-sync")
"${SKIP_DATA}" && _args+=(--skip-data)
[ "${MODE}" = "debug" ] && _args+=(--debug)

echo "[run_full_pipeline] Running research pipeline: ${REPO_ROOT}/scripts/run_research_full.sh ${_args[*]}"
"${REPO_ROOT}/scripts/run_research_full.sh" "${_args[@]}"

# 6) Update report table in refs/rendu1.md from RESULTS.md
if [ -f "${REPO_ROOT}/scripts/update_report_from_results.py" ]; then
  echo "[run_full_pipeline] Updating report table (refs/rendu1.md)..."
  (cd "${REPO_ROOT}" && uv run python scripts/update_report_from_results.py) || true
fi

echo "[run_full_pipeline] Done. Results: ${RECIPE_DIR}/exp/*/RESULTS.md, summary: ${LOG_DIR}/research_results_*.txt"
