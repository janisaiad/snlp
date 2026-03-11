#!/usr/bin/env bash
# Minimal ML-SUPERB pipeline test: data prep + a few training steps to verify hardware and stack.
# Run from repo root: uv sync --extra espnet && bash scripts/test_mlsuperb_pipeline.sh
# Requires: snlp env with espnet + s3prl + datasets (uv sync --extra espnet).
# Note: data prep downloads espnet/ml_superb_hf from HuggingFace; ensure enough disk quota.
# Quick stack check without download: uv run python scripts/test_mlsuperb_stack.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/models/espnet/egs2/ml_superb2/asr1"

if [ ! -f "${RECIPE_DIR}/run.sh" ]; then
    echo "Recipe not found at ${RECIPE_DIR}. Ensure models/espnet is present." >&2
    exit 1
fi

cd "${RECIPE_DIR}"
# use repo venv so python/python3 have espnet, s3prl, datasets
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    set +u
    source "${REPO_ROOT}/.venv/bin/activate"
    set -u
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

echo "[test_mlsuperb_pipeline] Stage 1-4: data prep (download HF dataset)..."
./run.sh --stop_stage 4

echo "[test_mlsuperb_pipeline] Stage 5-11: minimal training (1 epoch, 2 iters)..."
./run.sh --stage 5 --stop_stage 11 --asr_args "--max_epoch 1 --num_iters_per_epoch 2"

echo "[test_mlsuperb_pipeline] Done. Pipeline (data + SSL frontend + CTC train) ran successfully."
