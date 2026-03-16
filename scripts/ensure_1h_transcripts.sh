#!/usr/bin/env bash
# Create transcript_1h_train.txt from transcript_10min_train.txt where 1h is missing.
# Lets the recipe run "1h" data prep when you only have 10min ML-SUPERB data.
# From repo root: ./scripts/ensure_1h_transcripts.sh
# Optional: MLSUPERB=/path/to/data ./scripts/ensure_1h_transcripts.sh
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLSUPERB="${MLSUPERB:-${REPO_ROOT}/data/ml_superb}"

# lang -> dataset dir (same as single_lang_data_prep.py LANG_TO_SELECTED_DATASET)
for _lang_code in eng fra deu; do
  case "${_lang_code}" in
    eng) _ds=mls ;;
    fra) _ds=voxforge ;;
    deu) _ds=swc ;;
    *) continue ;;
  esac
  _dir="${MLSUPERB}/${_ds}/${_lang_code}"
  _train_1h="${_dir}/transcript_1h_train.txt"
  _train_10="${_dir}/transcript_10min_train.txt"
  if [ ! -f "${_train_1h}" ] && [ -f "${_train_10}" ]; then
    cp "${_train_10}" "${_train_1h}"
    echo "[ensure_1h] Created ${_train_1h} from 10min"
  fi
done
echo "[ensure_1h] Done. MLSUPERB=${MLSUPERB}"
