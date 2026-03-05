#!/usr/bin/env bash
# Download ML-SUPERB data for asr1 recipe (mono/multi tracks).
# Usage: from repo root, ./scripts/download_mlsuperb_data.sh
# Or: MLSUPERB=/path/to/dir ./scripts/download_mlsuperb_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${MLSUPERB:-${REPO_ROOT}/data/ml_superb}"

echo "ML-SUPERB data directory: ${DATA_DIR}"
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

# Option 1: try Huggingface Hub (repo has eighth_version.zip; we download then extract)
if command -v huggingface-cli &>/dev/null; then
  echo "Download start: $(date -Iseconds 2>/dev/null || date)"
  _start=$(date +%s 2>/dev/null || true)
  echo "Running: huggingface-cli download ftshijt/mlsuperb_8th --repo-type dataset --local-dir ."
  if huggingface-cli download ftshijt/mlsuperb_8th --repo-type dataset --local-dir .; then
    _end=$(date +%s 2>/dev/null || true)
    echo "Download finished: $(date -Iseconds 2>/dev/null || date)"
    if [ -n "${_start:-}" ] && [ -n "${_end:-}" ]; then
      _elapsed=$((_end - _start))
      echo "Elapsed time: ${_elapsed} s ($((${_elapsed} / 60))m $((${_elapsed} % 60))s)"
    fi
    # HF repo is just eighth_version.zip (30GB); recipe needs $MLSUPERB/mls/eng/ etc.
    if [ -f "eighth_version.zip" ] && [ ! -d "mls" ]; then
      echo "Extracting eighth_version.zip (this may take a few minutes)..."
      unzip -o -q eighth_version.zip
      # zip may extract to a top-level dir that contains mls/ (e.g. eighth_version/mls or just mls)
      if [ -d "mls" ]; then
        echo "Extract done. Data layout: mls/ present."
      else
        _sub=$(find . -maxdepth 1 -type d -name "eighth_version" 2>/dev/null | head -1)
        [ -z "${_sub}" ] && _sub=$(find . -maxdepth 1 -type d ! -name "." 2>/dev/null | head -1)
        if [ -n "${_sub}" ] && [ -d "${_sub}/mls" ]; then
          echo "Moving ${_sub}/* into ${DATA_DIR} so mls/ is under MLSUPERB..."
          mv "${_sub}"/* . 2>/dev/null || true
          rmdir "${_sub}" 2>/dev/null || true
        fi
      fi
      if [ ! -d "mls" ]; then
        echo "Contents after unzip:"
        ls -la
        echo "If a subdir here contains mls/, set MLSUPERB to that path, e.g.:"
        echo "  export MLSUPERB=${DATA_DIR}/<subdir>"
      fi
    fi
    if [ -d "mls" ]; then
      echo "Data ready. Run: ./scripts/run_ml_superb_baseline.sh"
      exit 0
    fi
    echo "If mls/ is present above, run: ./scripts/run_ml_superb_baseline.sh"
    exit 0
  fi
fi

# Option 2: manual instructions
echo "Automatic download did not complete or huggingface-cli not found."
echo ""
echo "Manual setup:"
echo "  1. Download ML-SUPERB 8th from: https://huggingface.co/datasets/ftshijt/mlsuperb_8th"
echo "     or Google Drive: https://drive.google.com/file/d/1vQ5NksmGl-lY7I4mlU4Kde3EhrEYGii2/view"
echo "  2. Extract the archive so that ${DATA_DIR} contains dataset folders (e.g. mls, voxforge, commonvoice)"
echo "     and under each: lang codes (e.g. eng), with transcript_10min_train.txt, transcript_10min_dev.txt,"
echo "     transcript_10min_test.txt, and wav/<utt_id>.wav"
echo "  3. Export and run: export MLSUPERB=${DATA_DIR}"
echo "  4. From models/espnet/egs2/ml_superb/asr1 run: ./run_one_lang.sh --single_lang eng1 --duration 10min"
exit 1
