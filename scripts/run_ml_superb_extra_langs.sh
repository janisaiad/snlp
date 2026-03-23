#!/usr/bin/env bash
# Run data prep + ASR for extra langs (fra1, deu1, ...).
# Usage: export MLSUPERB=/path/to/ml_superb && ./scripts/run_ml_superb_extra_langs.sh fra1 deu1
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
[ $# -ge 1 ] || { echo "Usage: $0 <lang> ..." >&2; exit 1; }
cd "${RECIPE}"
. ./path.sh
. ./cmd.sh
. ./db.sh
[ -n "${MLSUPERB:-}" ] && [ -d "${MLSUPERB}" ] || { echo "Set MLSUPERB" >&2; exit 1; }
for lang in "$@"; do
  echo "=== ${lang}: stage 1..4 data prep ==="
  ./run_one_lang.sh --single_lang "${lang}" --duration 10min --stage 1 --stop_stage 4 || {
    echo "${lang}: data prep failed (likely missing corpus in MLSUPERB), skipping" >&2
    continue
  }
  if python3 -c "import s3prl" 2>/dev/null; then
    ./run_one_lang.sh --single_lang "${lang}" --duration 10min --asr_config conf/tuning/train_asr_s3prl_10min.yaml --stage 5 --stop_stage 13 || true
  fi
  ./run_one_lang.sh --single_lang "${lang}" --duration 10min --asr_config conf/tuning/train_asr_jepa_10min.yaml --stage 5 --stop_stage 13 || true
  ./run_one_lang.sh --single_lang "${lang}" --duration 10min --asr_config conf/tuning/train_asr_wavjepa_10min.yaml --stage 5 --stop_stage 13 || true
done
