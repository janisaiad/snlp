#!/usr/bin/env bash
# Full pipeline at scale: optional JEPA/WavJEPA SSL pretraining (multi-GPU) then ML-SUPERB download + ASR train + eval.
# Use this on a big GPU instance when you want to pretrain a JEPA encoder then run the benchmark.
#
# Usage (from repo root):
#   ./scripts/run_full_pipeline_at_scale.sh
#     → ASR only (same as run_full_pipeline.sh)
#   ./scripts/run_full_pipeline_at_scale.sh --pretrain-gpus 10 --pretrain-data audioset
#     → WavJEPA pretraining on 10 GPUs (AudioSet), then ASR pipeline
#   ./scripts/run_full_pipeline_at_scale.sh --pretrain-gpus 2 --pretrain-data librispeech --skip-download
#     → Quick pretrain test (2 GPUs, LibriSpeech), then ASR with existing data
#
# Prerequisites for pretraining: AudioSet (or LibriSpeech) paths in third_party/wavjepa/configs/data/.
# After pretraining, checkpoints go to logs/wavjepa_pretrain/ (or --pretrain-save-dir). Using them in ASR
# requires loading the encoder into the WavJEPA frontend (see refs/RESEARCH_PLAN.md).
set -e
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRETRAIN_GPUS=0
PRETRAIN_DATA=audioset
PRETRAIN_SAVE_DIR="${REPO_ROOT}/logs/wavjepa_pretrain"
PIPELINE_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --pretrain-gpus)   PRETRAIN_GPUS="$2"; shift 2 ;;
    --pretrain-data)   PRETRAIN_DATA="$2"; shift 2 ;;
    --pretrain-save-dir) PRETRAIN_SAVE_DIR="$2"; shift 2 ;;
    *)                 PIPELINE_ARGS+=("$1"); shift ;;
  esac
done

echo "[run_full_pipeline_at_scale] REPO_ROOT=${REPO_ROOT}"
echo "[run_full_pipeline_at_scale] pretrain_gpus=${PRETRAIN_GPUS} pretrain_data=${PRETRAIN_DATA}"

# use project venv so pretrain and ASR share the same env
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  set +u
  . "${REPO_ROOT}/.venv/bin/activate"
  set -u
fi

# 1) Optional: WavJEPA SSL pretraining (clone repo, then train)
if [ "${PRETRAIN_GPUS}" -gt 0 ]; then
  echo "[run_full_pipeline_at_scale] Step 1: WavJEPA pretraining (${PRETRAIN_GPUS} GPUs, data=${PRETRAIN_DATA})..."
  if [ ! -d "${REPO_ROOT}/third_party/wavjepa" ] || [ ! -f "${REPO_ROOT}/third_party/wavjepa/train.py" ]; then
    "${REPO_ROOT}/scripts/setup_wavjepa.sh"
  fi
  if [ -f "${REPO_ROOT}/third_party/wavjepa/requirements.txt" ]; then
    (cd "${REPO_ROOT}" && uv pip install -r third_party/wavjepa/requirements.txt 2>/dev/null || true)
  fi
  "${REPO_ROOT}/scripts/run_wavjepa_pretrain.sh" \
    --num-gpus "${PRETRAIN_GPUS}" \
    --data "${PRETRAIN_DATA}" \
    --save-dir "${PRETRAIN_SAVE_DIR}" \
    || { echo "[run_full_pipeline_at_scale] Pretraining failed." >&2; exit 1; }
  echo "[run_full_pipeline_at_scale] Pretraining done. Checkpoints: ${PRETRAIN_SAVE_DIR}"
else
  echo "[run_full_pipeline_at_scale] Skipping pretraining (--pretrain-gpus 0 or not set)."
fi

# 2) ASR pipeline: download ML-SUPERB (if needed) + data prep + 30ep train + eval
echo "[run_full_pipeline_at_scale] Step 2: ASR pipeline (download + data prep + train + eval)..."
"${REPO_ROOT}/scripts/run_full_pipeline.sh" --no-sync "${PIPELINE_ARGS[@]}"

echo "[run_full_pipeline_at_scale] Done. ASR results: ${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1/exp/*/RESULTS.md"
[ "${PRETRAIN_GPUS}" -gt 0 ] && echo "[run_full_pipeline_at_scale] Pretrain checkpoints: ${PRETRAIN_SAVE_DIR}"
