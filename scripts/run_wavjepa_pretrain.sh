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

# Show who is using the GPU so you don't OOM when another process already holds most of it
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[run_wavjepa_pretrain] GPU memory before run:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv,noheader 2>/dev/null | while read -r line; do echo "  GPU $line"; done
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | while IFS=, read -r pid mem name; do echo "  PID $pid: ${mem} (${name})"; done
fi

# Cap num_gpus to available GPUs so it runs on machines with fewer devices
AVAILABLE_GPUS=1
if command -v nvidia-smi >/dev/null 2>&1; then
  AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  [ "${AVAILABLE_GPUS}" -lt 1 ] && AVAILABLE_GPUS=1
fi
if [ "${NUM_GPUS}" -gt "${AVAILABLE_GPUS}" ]; then
  echo "[run_wavjepa_pretrain] Capping num_gpus ${NUM_GPUS} -> ${AVAILABLE_GPUS} (only ${AVAILABLE_GPUS} GPU(s) available)"
  NUM_GPUS="${AVAILABLE_GPUS}"
fi

# For data=smoke use a small batch size and no compile; pick a GPU with free memory or fall back to CPU
SMOKE_OVERRIDES=()
SMOKE_USE_CPU=false
if [ "${DATA}" = "smoke" ]; then
  SMOKE_OVERRIDES=(trainer.batch_size=4 trainer.compile_modules=False)
  echo "[run_wavjepa_pretrain] Smoke run: using trainer.batch_size=4, trainer.compile_modules=False to save GPU memory"
  # Pick a GPU with at least 4 GiB free so smoke fits; if none, run on CPU so the run completes
  MIN_FREE_MB=4000
  if command -v nvidia-smi >/dev/null 2>&1; then
    BEST_GPU=""
    BEST_FREE=0
    while IFS=, read -r gpu_idx free_mb; do
      gpu_idx=$(echo "$gpu_idx" | tr -d ' ')
      free_mb=$(echo "$free_mb" | tr -d ' ')
      [[ ! "${free_mb}" =~ ^[0-9]+$ ]] && continue
      if [ "${free_mb}" -ge "${MIN_FREE_MB}" ] && [ "${free_mb}" -gt "${BEST_FREE}" ]; then
        BEST_FREE=${free_mb}
        BEST_GPU=${gpu_idx}
      fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "${BEST_GPU}" ]; then
      export CUDA_VISIBLE_DEVICES="${BEST_GPU}"
      NUM_GPUS=1
      echo "[run_wavjepa_pretrain] Smoke run: using GPU ${BEST_GPU} (${BEST_FREE} MiB free)"
    else
      SMOKE_USE_CPU=true
      SMOKE_OVERRIDES=(trainer.batch_size=4 trainer.compile_modules=False trainer.accelerator=cpu trainer.num_gpus=1)
      echo "[run_wavjepa_pretrain] Smoke run: no GPU with >= ${MIN_FREE_MB} MiB free; running on CPU"
    fi
  else
    SMOKE_USE_CPU=true
    SMOKE_OVERRIDES=(trainer.batch_size=4 trainer.compile_modules=False trainer.accelerator=cpu trainer.num_gpus=1)
    echo "[run_wavjepa_pretrain] Smoke run: nvidia-smi not found; running on CPU"
  fi
fi

# WavJEPA uses Hydra; config_path is relative to cwd so we must run from WAVJEPA_DIR.
# Prefer snlp venv (has wavjepa editable + deps e.g. transformers); else wavjepa .venv.
echo "[run_wavjepa_pretrain] Running from ${WAVJEPA_DIR} with num_gpus=${NUM_GPUS} data=${DATA} save_dir=${SAVE_DIR}"
mkdir -p "${SAVE_DIR}"
# Mark run start so we have a record even if the log is lost
printf "[run_wavjepa_pretrain] started at %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${SAVE_DIR}/pretrain_started.txt"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
# Optional: reduce CUDA fragmentation if OOM persists (e.g. when GPU is shared)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -f "${REPO_ROOT}/.venv/bin/python" ] && command -v uv >/dev/null 2>&1; then
  (cd "${WAVJEPA_DIR}" && uv run --project "${REPO_ROOT}" python train.py \
    data="${DATA}" \
    trainer.num_gpus="${NUM_GPUS}" \
    save_dir="$(cd "${REPO_ROOT}" && realpath "${SAVE_DIR}")" \
    "${SMOKE_OVERRIDES[@]}" \
    "${EXTRA_ARGS[@]}")
elif [ -f "${WAVJEPA_DIR}/.venv/bin/activate" ]; then
  (
    cd "${WAVJEPA_DIR}"
    set +u
    . "${WAVJEPA_DIR}/.venv/bin/activate"
    set -u
    python train.py \
      data="${DATA}" \
      trainer.num_gpus="${NUM_GPUS}" \
      save_dir="$(cd "${REPO_ROOT}" && realpath "${SAVE_DIR}")" \
      "${SMOKE_OVERRIDES[@]}" \
      "${EXTRA_ARGS[@]}"
  )
else
  (cd "${WAVJEPA_DIR}" && python train.py \
    data="${DATA}" \
    trainer.num_gpus="${NUM_GPUS}" \
    save_dir="$(cd "${REPO_ROOT}" && realpath "${SAVE_DIR}")" \
    "${SMOKE_OVERRIDES[@]}" \
    "${EXTRA_ARGS[@]}")
fi

printf "[run_wavjepa_pretrain] finished at %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${SAVE_DIR}/pretrain_finished.txt"
echo "[run_wavjepa_pretrain] Done. Checkpoints: ${SAVE_DIR}"
echo "[run_wavjepa_pretrain] Progress trail: ${SAVE_DIR}/pretrain_progress.txt (step and loss every 200 steps)"
