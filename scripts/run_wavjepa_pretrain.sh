#!/usr/bin/env bash
# Run WavJEPA SSL pretraining using the cloned third_party/wavjepa repo.
# Requires: ./scripts/setup_wavjepa.sh already run; data config set in configs/data/ (see README in wavjepa).
# Usage (from snlp repo root):
#   ./scripts/run_wavjepa_pretrain.sh --num-gpus 10 --data audioset
#   ./scripts/run_wavjepa_pretrain.sh --num-gpus 2 --data librispeech   # quick test
#   ./scripts/run_wavjepa_pretrain.sh --num-gpus 10 --data audioset --save-dir /path/to/checkpoints
set -e
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAVJEPA_DIR="${REPO_ROOT}/third_party/wavjepa"
NUM_GPUS=2
DATA=audioset
SAVE_DIR="${REPO_ROOT}/logs/wavjepa_pretrain"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --data)     DATA="$2"; shift 2 ;;
    --save-dir) SAVE_DIR="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [ ! -d "${WAVJEPA_DIR}" ] || [ ! -f "${WAVJEPA_DIR}/train.py" ]; then
  echo "[run_wavjepa_pretrain] WavJEPA not found at ${WAVJEPA_DIR}. Run: ./scripts/setup_wavjepa.sh" >&2
  exit 1
fi

# WavJEPA uses Hydra; config_path is relative to cwd so we must run from WAVJEPA_DIR
# Override trainer.num_gpus and data=...; optional save_dir in config/base.yaml
echo "[run_wavjepa_pretrain] Running from ${WAVJEPA_DIR} with num_gpus=${NUM_GPUS} data=${DATA} save_dir=${SAVE_DIR}"
mkdir -p "${SAVE_DIR}"

(
  cd "${WAVJEPA_DIR}"
  export HYDRA_FULL_ERROR=1
  # use current venv so that wavjepa (editable) and deps are available
  python train.py \
    data="${DATA}" \
    trainer.num_gpus="${NUM_GPUS}" \
    save_dir="$(cd "${REPO_ROOT}" && realpath "${SAVE_DIR}")" \
    "${EXTRA_ARGS[@]}"
)

echo "[run_wavjepa_pretrain] Done. Checkpoints: ${SAVE_DIR}"
