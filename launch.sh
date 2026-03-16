#!/usr/bin/env bash
# One script in root: setup env then full pipeline (download → eng1+fra1+deu1 10min → 30ep train → eval → update refs/rendu1.md).
# Run from repo root: ./launch.sh
# Optional: ./launch.sh --pretrain-gpus 10   to run WavJEPA pretraining first (e.g. overnight); ./launch.sh --debug   for 1-ep check; ./launch.sh --skip-download   if data exists.
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

# 4) Run full pipeline: (optional pretrain) + download + data prep + eng1/fra1/deu1 30ep train + eval + update report table
echo "[launch] Running full pipeline (multi-lang 30ep + report update; pass --pretrain-gpus N for JEPA pretraining)..."
"${REPO_ROOT}/scripts/run_full_pipeline.sh" --no-sync "$@"

echo "[launch] Done. Results: ${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1/exp/*/RESULTS.md and ${REPO_ROOT}/logs/research_results_*.txt"
