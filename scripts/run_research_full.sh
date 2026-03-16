#!/usr/bin/env bash
# Run the full research pipeline: data (optional) → train all frontends → eval → summary.
# Encoder pretraining is NOT run here; we use pretrained HuBERT/WavJEPA or minimal JEPA.
# Usage (from snlp repo root):
#   ./scripts/run_research_full.sh                # data + full 30ep train + eval
#   ./scripts/run_research_full.sh --skip-data    # skip data prep
#   ./scripts/run_research_full.sh --debug        # 1 ep, 2 iters (quick check)
#   ./scripts/run_research_full.sh --no-sync      # skip uv sync
set -e
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
LOG_DIR="${REPO_ROOT}/logs"
SKIP_DATA=false
DO_SYNC=true
MODE=full

while [ $# -gt 0 ]; do
  case "$1" in
    --debug)      MODE=debug; shift ;;
    --skip-data)  SKIP_DATA=true; shift ;;
    --no-sync)    DO_SYNC=false; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${LOG_DIR}"
echo "[run_research_full] MODE=${MODE} SKIP_DATA=${SKIP_DATA} DO_SYNC=${DO_SYNC}"
echo "[run_research_full] REPO_ROOT=${REPO_ROOT}"

if "${DO_SYNC}"; then
  echo "[run_research_full] Syncing (uv sync, espnet editable)..."
  (cd "${REPO_ROOT}" && uv sync && uv add --editable ./models/espnet)
fi

_args=(--no-sync)
"${SKIP_DATA}" && _args+=(--skip-data)
[ "${MODE}" = "debug" ] && _args+=(--debug) || _args+=(--full)

echo "[run_research_full] Calling run_ml_superb_train_eval_all.sh ${_args[*]}"
"${REPO_ROOT}/scripts/run_ml_superb_train_eval_all.sh" "${_args[@]}"

# Write a short results summary
SUMMARY="${LOG_DIR}/research_results_$(date +%Y%m%d_%H%M).txt"
{
  echo "Research run: $(date -Iseconds)"
  echo "MODE=${MODE} SKIP_DATA=${SKIP_DATA}"
  echo ""
  echo "--- CER/WER (test_10min_eng1) from RESULTS.md ---"
  for f in "${RECIPE_DIR}"/exp/asr_train_*/RESULTS.md; do
    [ -f "$f" ] || continue
    tag=$(basename "$(dirname "$f")")
    echo "## ${tag}"
    grep -A1 "decode_asr.*test_10min" "$f" 2>/dev/null | head -6 || true
    echo ""
  done
  echo "--- End ---"
} > "${SUMMARY}" 2>/dev/null || true

echo "[run_research_full] Done. Summary: ${SUMMARY}"
echo "[run_research_full] Full results: ${RECIPE_DIR}/exp/*/RESULTS.md"
echo "[run_research_full] Research plan: refs/RESEARCH_PLAN.md"
