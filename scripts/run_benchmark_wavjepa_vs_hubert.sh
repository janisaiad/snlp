#!/usr/bin/env bash
# Run ML-SUPERB-style benchmark: compare WavJEPA (HuggingFace or local ckpt) vs HuBERT vs JEPA minimal.
# Requires: MLSUPERB set to ML-SUPERB data root; data prepared (or run without --skip-data).
# Usage (from snlp repo root):
#   ./scripts/run_benchmark_wavjepa_vs_hubert.sh              # full 30 ep (long)
#   ./scripts/run_benchmark_wavjepa_vs_hubert.sh --debug      # quick: 1 ep, 2 iters
#   ./scripts/run_benchmark_wavjepa_vs_hubert.sh --skip-data --no-sync
set -e
set -u
set -o pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [ -z "${MLSUPERB:-}" ] || [ ! -d "${MLSUPERB}" ]; then
  echo "Set MLSUPERB to ML-SUPERB data root. Example: export MLSUPERB=/path/to/ml_superb" >&2
  exit 1
fi
exec ./scripts/run_ml_superb_train_eval_all.sh "$@"
