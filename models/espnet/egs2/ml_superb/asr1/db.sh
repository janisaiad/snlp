# ML-SUPERB data root (unzipped tarball or output of scripts/download_mlsuperb_data.sh).
# Override with: export MLSUPERB=/path/to/ml_superb_data
# Default: snlp repo root data/ml_superb (asr1 -> ml_superb -> egs2 -> espnet -> models -> snlp = 5 levels)
_mls_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
export MLSUPERB="${MLSUPERB:-${_mls_root}/data/ml_superb}"
