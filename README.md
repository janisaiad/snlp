## Installation

**One-liner (Unix):** `chmod +x launch.sh && ./launch.sh` — we make the script executable, then run it to create the venv, install deps, and run tests.

To install dependencies using uv manually, follow these steps:

1. Install uv:
   
   **macOS/Linux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or using wget:
   ```bash
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

   **Windows:**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Alternatively, you can install uv using:
   - pipx (recommended): `pipx install uv`
   - pip: `pip install uv`
   - Homebrew: `brew install uv`
   - WinGet: `winget install --id=astral-sh.uv -e`
   - Scoop: `scoop install main/uv`

2. Using uv in this project:

   - Initialize a new virtual environment:
   ```bash
   uv venv
   ```

   - Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Unix
   .venv\Scripts\activate     # On Windows
   ```

   - Install dependencies from requirements.txt:
   ```bash
   uv add -r requirements.txt
   ```


   - Add a new package:
   ```bash
   uv add package_name
   ```

   - Remove a package:
   ```bash
   uv remove package_name
   ```

   - Update a package:
   ```bash
   uv pip install --upgrade package_name
   ```

   - Generate requirements.txt:
   ```bash
   uv pip freeze > requirements.txt
   ```

   - List installed packages:
     ```bash
     uv pip list
     ```

## Launch

`launch.sh` is the project bootstrap script. On Unix/macOS, make it executable once with `chmod +x launch.sh`, then run `./launch.sh` to create the virtual environment, install the project in editable mode, and run the test suite.

## Warning

If you're using macOS or Python 3, replace `pip` with `pip3` in line 1 of ```launch.sh```

Replace with your project folder name (which means the name of the library you are deving) in :```tests/test_env.py: ```