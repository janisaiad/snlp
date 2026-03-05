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

## ML-SUPERB ASR training (optional)

This repo can run an **ML-SUPERB**-style ASR baseline: train a small transformer encoder + CTC on 10 minutes of English (or other languages), then decode and report CER/WER. It is a **baseline / proof-of-concept** run: the model trains and decoding works; for strong numbers you’d use more data, more epochs, or SSL (e.g. HuBERT) as in the real ML-SUPERB benchmark.

**What’s been done vs project goal:** The project abstract is “pretrained SSL (HuBERT, wav2vec 2.0) **frozen** + CTC with 10 min/1h in a low-resource language (ML-SUPERB)”. So far we have run the **FBANK baseline** only: 10 min English, CTC on top of hand-crafted log-mel features (no SSL, nothing frozen). Next step for the real goal: use the recipe’s SSL config (e.g. HuBERT), freeze the SSL backbone, train only the CTC head on 10 min/1h; then you can add comparisons (other SSL models, languages, LoRA/PEFT, multilingual vs monolingual).

### What the training is

- **Task:** Automatic speech recognition (ASR).
- **Data:** ML-SUPERB format under `data/ml_superb/` (e.g. `mls/eng/` with `transcript_10min_*.txt` and `wav/*.wav`). Default run: English, 10 min subset (`eng1`, 10min).
- **Model:** ESPnet ASR — 80-dim log-mel frontend, 2-layer transformer encoder, CTC only (no attention decoder in the default config).
- **Outputs:** Trained checkpoint (e.g. `valid.loss.ave.pth`), decode hypotheses, and CER/WER in the experiment dir.

So: **yes, it is working** — the pipeline trains, decodes, and scores. The file `data/ml_superb/mls/eng/transcript_10min_test.txt` is **not** the model output; it is the **ground-truth labels** for the test set (one line per utterance: `utt_id duration reference_text`). The model’s predictions are written under the decode directories (see below); CER/WER compare those predictions to these references.

### Where the logs and results are

All paths below are relative to **`models/espnet/egs2/ml_superb/asr1/`** (recipe directory). Default experiment name: `asr_train_asr_fbank_single_eng1_10min`.

| What | Path |
|------|------|
| **Training log** (epochs, loss, CER) | `exp/asr_train_asr_fbank_single_eng1_10min/train.log` |
| **Config** | `exp/asr_train_asr_fbank_single_eng1_10min/config.yaml` |
| **Checkpoint used for decoding** | `exp/asr_train_asr_fbank_single_eng1_10min/valid.loss.ave.pth` |
| **CER/WER summary** | `exp/asr_train_asr_fbank_single_eng1_10min/RESULTS.md` |
| **Decode hypotheses (model output)** | `exp/.../decode_asr_asr_model_valid.loss.ave/test_10min_eng1/text` and same for `org/dev_10min_eng1` |
| **Score details** | `exp/.../decode_asr_asr_model_valid.loss.ave/test_10min_eng1/score_cer/result.txt` (and `score_wer/result.txt`) |

To watch training live: from the `asr1` dir run `tail -f exp/asr_train_asr_fbank_single_eng1_10min/train.log`. Full run and one-liner are documented in **REPRODUCTION.md** (in `refs/` or repo root).