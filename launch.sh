#!/usr/bin/env bash
# One script in root: setup env (uv, venv, deps) then download ML-SUPERB data and run full training + eval (30 ep).
# Run from repo root: ./launch.sh
# Optional: ./launch.sh --debug   for quick 1-ep sanity check; ./launch.sh --skip-download   if data already at data/ml_superb.
set -e
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# 1) Ensure uv is available
if ! command -v uv &>/dev/null; then
  echo "[launch] uv not in PATH; installing via curl..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH:-}"
  if ! command -v uv &>/dev/null; then
    echo "[launch] Install failed or uv not in PATH. Add ~/.local/bin to PATH or: pip install uv" >&2
    exit 1
  fi
fi

# 2) Create venv (if missing) and install deps (sync + espnet editable)
echo "[launch] Syncing dependencies (uv sync, espnet editable)..."
if [ ! -d "${REPO_ROOT}/.venv" ]; then
  uv venv
fi
set +u
. "${REPO_ROOT}/.venv/bin/activate"
set -u
uv sync
uv add --editable ./models/espnet

# 3) Optional: quick env check
if [ -f "${REPO_ROOT}/tests/test_env.py" ]; then
  echo "[launch] Running tests/test_env.py..."
  uv run pytest tests/test_env.py -q || true
fi

# 4) Run full pipeline: download data (if needed) + data prep + 30ep train + eval
echo "[launch] Running full pipeline (download + data prep + 30ep train + eval)..."
"${REPO_ROOT}/scripts/run_full_pipeline.sh" --no-sync "$@"

echo "[launch] Done. Results: ${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1/exp/*/RESULTS.md and ${REPO_ROOT}/logs/research_results_*.txt"
