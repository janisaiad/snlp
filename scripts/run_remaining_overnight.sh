#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
RECIPE_EXP="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1/exp"
LOCAL_RESULTS="${RECIPE_EXP}/asr_train_asr_wavjepa_local_ckpt_10min_eng1_10min/RESULTS.md"

mkdir -p "${LOG_DIR}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export MLSUPERB="${MLSUPERB:-${REPO_ROOT}/data/ml_superb}"

echo "[overnight] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[overnight] repo=${REPO_ROOT}"
echo "[overnight] MLSUPERB=${MLSUPERB}"

# 1) Wait for local WavJEPA-ASR 30ep to finish (or start it if absent), then collect table.
if [ ! -f "${LOCAL_RESULTS}" ]; then
  if ! pgrep -af "run_asr_wavjepa_local_benchmark|asr_train_asr_wavjepa_local_ckpt_10min_eng1_10min" >/dev/null 2>&1; then
    echo "[overnight] launching local WavJEPA ASR benchmark"
    nohup "${REPO_ROOT}/scripts/run_asr_wavjepa_local_benchmark.sh" > "${LOG_DIR}/asr_wavjepa_local_full.log" 2>&1 &
  else
    echo "[overnight] local WavJEPA ASR already running"
  fi

  echo "[overnight] waiting for local ASR RESULTS.md ..."
  for _ in $(seq 1 240); do
    if [ -f "${LOCAL_RESULTS}" ]; then
      break
    fi
    sleep 120
  done
fi

if [ -f "${LOCAL_RESULTS}" ]; then
  echo "[overnight] collecting ASR table"
  (cd "${REPO_ROOT}" && uv run python scripts/collect_asr_results.py) | tee "${LOG_DIR}/collect_asr_results_overnight.log"
else
  echo "[overnight] local ASR did not finish within wait window; continuing"
fi

# 2) Try extra-language runs only if required corpora exist.
if [ -d "${MLSUPERB}/voxforge/fra" ] || [ -d "${MLSUPERB}/swc/deu" ]; then
  echo "[overnight] running extra languages (fra1/deu1 if present)"
  "${REPO_ROOT}/scripts/run_ml_superb_extra_langs.sh" fra1 deu1 > "${LOG_DIR}/ml_superb_extra_langs_overnight.log" 2>&1 || true
else
  echo "[overnight] skipping extra languages: fra/deu corpora missing in MLSUPERB"
fi

# 3) Start / resume full JEPA pretraining (375k default steps in config).
if ! pgrep -af "third_party/wavjepa/train.py|run_wavjepa_pretrain.sh" >/dev/null 2>&1; then
  echo "[overnight] launching JEPA pretraining (auto-resume from last.ckpt if found)"
  nohup "${REPO_ROOT}/scripts/run_wavjepa_pretrain.sh" --num-gpus 1 --data audioset > "${LOG_DIR}/wavjepa_pretrain_overnight.log" 2>&1 &
else
  echo "[overnight] JEPA pretraining already running"
fi

echo "[overnight] done setup $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[overnight] monitor:"
echo "  - ${LOG_DIR}/asr_wavjepa_local_full.log"
echo "  - ${LOG_DIR}/collect_asr_results_overnight.log"
echo "  - ${LOG_DIR}/ml_superb_extra_langs_overnight.log"
echo "  - ${LOG_DIR}/wavjepa_pretrain_overnight.log"
