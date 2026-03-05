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

`launch.sh` is the project bootstrap script: it creates the virtual environment, installs the project in editable mode, and runs the test suite. On Unix/macOS you must make it executable once, then run it:

- **`chmod +x launch.sh`** — mark the script as executable (only needed once per clone).
- **`./launch.sh`** — run it; it will use `pip` (or `pip3` on some systems) to set up the venv and install this package so you can import it and run tests.

If your system only has `pip3` or you use Python 3 explicitly, edit the first line of `launch.sh` and replace `pip` with `pip3`. If tests fail with an import error, open `tests/test_env.py` and set the project folder name (the importable package name you are developing) to match your project.

## Warning

- **macOS / Python 3:** The script may call `pip` by default. If that fails or points to Python 2, replace `pip` with `pip3` in the first line of `launch.sh`.
- **Test env:** In `tests/test_env.py`, replace the project folder name with the actual name of the library you are developing (the package you `import` in Python).