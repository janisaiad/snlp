curl -LsSf https://astral.sh/uv/install.sh
pip install uv
uv venv --clear
source .venv/bin/activate
uv pip install -e .
uv cache prune
source .venv/bin/activate
uv run pytest tests/test_env.py