#!/usr/bin/env bash
# Run several WavJEPA pretrainings one after the other: smoke first, then one or more data configs.
# Requires: ./scripts/setup_wavjepa.sh already run.
# Real data (audioset, librispeech) need paths in third_party/wavjepa/configs/data/*.yaml;
# default configs point at cluster paths (/gpfs/...) so use --only-smoke if you have no data.
# Usage (from snlp repo root):
#   ./scripts/run_wavjepa_pretrain_sequence.sh --only-smoke
#   ./scripts/run_wavjepa_pretrain_sequence.sh
#   ./scripts/run_wavjepa_pretrain_sequence.sh --num-gpus 2
#   ./scripts/run_wavjepa_pretrain_sequence.sh --skip-smoke --datas audioset
#   ./scripts/run_wavjepa_pretrain_sequence.sh --smoke-steps 1000 --datas "librispeech audioset"
set -e
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SAVE_DIR="${REPO_ROOT}/logs/wavjepa_pretrain"
RUN_SCRIPT="${REPO_ROOT}/scripts/run_wavjepa_pretrain.sh"

NUM_GPUS=1
SMOKE_STEPS=500
SKIP_SMOKE=false
DATAS="audioset librispeech"
ONLY_SMOKE=false

while [ $# -gt 0 ]; do
  case "$1" in
    --num-gpus)    NUM_GPUS="$2"; shift 2 ;;
    --smoke-steps) SMOKE_STEPS="$2"; shift 2 ;;
    --skip-smoke)  SKIP_SMOKE=true; shift ;;
    --only-smoke)  ONLY_SMOKE=true; shift ;;
    --datas)       DATAS="$2"; shift 2 ;;
    --base-save-dir) BASE_SAVE_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

[ "${ONLY_SMOKE}" = true ] && DATAS=""

if [ ! -x "${RUN_SCRIPT}" ]; then
  echo "[run_sequence] Run script not found or not executable: ${RUN_SCRIPT}" >&2
  exit 1
fi

run_one() {
  local data="$1"
  local save_dir="$2"
  echo ""
  echo "[run_sequence] ========== Starting run: data=${data} save_dir=${save_dir} =========="
  if [ "${data}" = "smoke" ]; then
    "${RUN_SCRIPT}" --num-gpus "${NUM_GPUS}" --data smoke --save-dir "${save_dir}" "trainer.steps=${SMOKE_STEPS}"
  else
    "${RUN_SCRIPT}" --num-gpus "${NUM_GPUS}" --data "${data}" --save-dir "${save_dir}"
  fi
  echo "[run_sequence] ========== Finished run: data=${data} =========="
}

echo "[run_sequence] Base save dir: ${BASE_SAVE_DIR}"
echo "[run_sequence] Num GPUs: ${NUM_GPUS} | Smoke steps: ${SMOKE_STEPS} | Skip smoke: ${SKIP_SMOKE} | Datas: ${DATAS}"

if [ "${SKIP_SMOKE}" = false ]; then
  run_one "smoke" "${BASE_SAVE_DIR}/smoke"
fi

for data in ${DATAS}; do
  run_one "${data}" "${BASE_SAVE_DIR}/${data}"
done

echo ""
echo "[run_sequence] All runs done. Checkpoints: ${BASE_SAVE_DIR}/*/saved_models_jepa_new_masking/"
