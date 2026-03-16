#!/usr/bin/env bash
# Tail SSL run logs live. From repo root.
#
#   ./scripts/tail_ssl_logs.sh           # full grid log (full_ssl_run.log)
#   ./scripts/tail_ssl_logs.sh train     # latest experiment's train.log
#   ./scripts/tail_ssl_logs.sh train_asr_s3prl_10min_eng1_10min  # that exp's train.log
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${REPO_ROOT}/models/espnet/egs2/ml_superb/asr1/exp"
MAIN_LOG="${REPO_ROOT}/full_ssl_run.log"

if [ $# -eq 0 ]; then
  if [ -f "${MAIN_LOG}" ]; then
    exec tail -f "${MAIN_LOG}"
  else
    echo "No ${MAIN_LOG}. Run: nohup ./scripts/run_ml_superb_ssl_full.sh > full_ssl_run.log 2>&1 &" >&2
    exit 1
  fi
fi

if [ "$1" = "train" ] && [ $# -eq 1 ]; then
  # latest train.log by mtime
  _latest=$(find "${EXP_DIR}" -maxdepth 2 -name "train.log" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
  if [ -n "${_latest}" ]; then
    exec tail -f "${_latest}"
  else
    echo "No train.log under ${EXP_DIR}" >&2
    exit 1
  fi
fi

# specific experiment
_tag="$1"
_log="${EXP_DIR}/${_tag}/train.log"
if [ -f "${_log}" ]; then
  exec tail -f "${_log}"
else
  echo "Not found: ${_log}" >&2
  echo "Available:" >&2
  ls -1 "${EXP_DIR}"/train_asr_s3prl_*/train.log 2>/dev/null | sed "s|${EXP_DIR}/||;s|/train.log||" >&2
  exit 1
fi
