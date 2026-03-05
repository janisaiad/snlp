# activate snlp project venv and espnet (from asr1: ../../../../../ = snlp repo root)
_snlp_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
if [ -f "${_snlp_root}/.venv/bin/activate" ]; then
  . "${_snlp_root}/.venv/bin/activate"
fi
export PYTHONPATH="${_snlp_root}/models/espnet:${PYTHONPATH:-}"
# prepend local bin so sclite wrapper is used when system sclite is not installed (set in asr1/path.sh)
