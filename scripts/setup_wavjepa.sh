#!/usr/bin/env bash
# Clone WavJEPA repo into third_party/wavjepa and add as editable dep so we can run pretraining.
# Usage (from snlp repo root):
#   ./scripts/setup_wavjepa.sh              # clone and uv add -e third_party/wavjepa
#   ./scripts/setup_wavjepa.sh --no-uv       # clone only (you install deps yourself)
set -e
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAVJEPA_DIR="${REPO_ROOT}/third_party/wavjepa"
WAVJEPA_REPO="${WAVJEPA_REPO:-https://github.com/labhamlet/wavjepa.git}"
WAVJEPA_BRANCH="${WAVJEPA_BRANCH:-master}"
DO_UV=true

while [ $# -gt 0 ]; do
  case "$1" in
    --no-uv) DO_UV=false; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "[setup_wavjepa] REPO_ROOT=${REPO_ROOT} WAVJEPA_DIR=${WAVJEPA_DIR}"

if [ -d "${WAVJEPA_DIR}/.git" ]; then
  echo "[setup_wavjepa] Already cloned. Pulling latest..."
  (cd "${WAVJEPA_DIR}" && git fetch origin && git checkout "${WAVJEPA_BRANCH}" && git pull --rebase origin "${WAVJEPA_BRANCH}" || true)
else
  mkdir -p "${REPO_ROOT}/third_party"
  echo "[setup_wavjepa] Cloning ${WAVJEPA_REPO} (branch ${WAVJEPA_BRANCH}) into ${WAVJEPA_DIR}..."
  git clone --branch "${WAVJEPA_BRANCH}" --depth 1 "${WAVJEPA_REPO}" "${WAVJEPA_DIR}"
fi

if "${DO_UV}"; then
  echo "[setup_wavjepa] Adding wavjepa as editable dependency (uv add -e)..."
  (cd "${REPO_ROOT}" && uv add --editable "${WAVJEPA_DIR}")
  echo "[setup_wavjepa] Install WavJEPA training deps: cd third_party/wavjepa && uv pip install -r requirements.txt (or use run_wavjepa_pretrain.sh which uses the snlp venv)"
fi

echo "[setup_wavjepa] Done. Pretrain with: ./scripts/run_wavjepa_pretrain.sh --num-gpus 10 --data audioset"
