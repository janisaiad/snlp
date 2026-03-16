# ML-SUPERB reproduction (incremental)

Reproduce ML-SUPERB baseline results for the snlp project (Janis track): one language, 10 min, then extend.

**This project uses [uv](https://docs.astral.sh/uv/).** Do not use `pip` inside the venv; use `uv add` / `uv sync` only.

---

## Full reproduction (data + SSL, eng1 10min)

End-to-end steps to obtain the reported SSL results (HuBERT frozen + CTC, eng1 10min). All commands from the **snlp repo root** unless noted.

### Step 1: Clone and enter repo

```bash
git clone <repo_url> snlp && cd snlp
```

### Step 2: Get the data

You need the **ML-SUPERB 8th release** (one large zip, ~30 GB). The recipe expects the **unzipped** root to contain dataset folders (e.g. `mls`, `voxforge`, `commonvoice`) with per-language subdirs and files: `transcript_10min_train.txt`, `transcript_10min_dev.txt`, `transcript_10min_test.txt`, and `wav/<utt_id>.wav`.

#### How to get the zip

- **Huggingface (browser):**  
  1. Open [https://huggingface.co/datasets/ftshijt/mlsuperb_8th](https://huggingface.co/datasets/ftshijt/mlsuperb_8th).  
  2. Click the **"Files and versions"** tab.  
  3. Download the file **`eighth_version.zip`** (use the download button next to it). You may need to log in or accept the dataset terms.

- **Huggingface (CLI):**  
  Install the CLI (`pip install huggingface_hub` or `uv add huggingface_hub`), then from any directory:
  ```bash
  huggingface-cli download ftshijt/mlsuperb_8th eighth_version.zip --repo-type dataset --local-dir .
  ```
  This creates `eighth_version.zip` in the current directory.

- **Google Drive:**  
  1. Open [https://drive.google.com/file/d/1vQ5NksmGl-lY7I4mlU4Kde3EhrEYGii2/view](https://drive.google.com/file/d/1vQ5NksmGl-lY7I4mlU4Kde3EhrEYGii2/view).  
  2. Click **"Download"** (top right). The downloaded file may be named `eighth_version.zip` or similar.  
  3. If the link asks for access, request it from the dataset owners or use the Huggingface source instead.

#### After you have the zip

**Option A — Automatic (recommended if `huggingface-cli` is available):**

From the **snlp repo root**:

```bash
./scripts/download_mlsuperb_data.sh
```

- If the script finds `huggingface-cli`, it downloads into `data/ml_superb` (or `$MLSUPERB`), then extracts the zip so `data/ml_superb/mls/eng/` etc. exist.  
- If you already have `eighth_version.zip` elsewhere, you can put it in `data/ml_superb/` and run the script; it will only extract (or use Option B below).

**Option B — Manual (you already have `eighth_version.zip`):**

1. Create the data directory and put the zip there (or extract anywhere):
   ```bash
   mkdir -p data/ml_superb
   # move or copy eighth_version.zip into data/ml_superb/
   cd data/ml_superb
   unzip eighth_version.zip
   ```
2. Ensure the **unzipped root** (the directory that contains `mls/`, `voxforge/`, etc.) is what the recipe sees. Sometimes the zip extracts into a subdir (e.g. `eighth_version/mls/`). In that case either:
   - set `MLSUPERB` to that subdir: `export MLSUPERB=/path/to/data/ml_superb/eighth_version`, or  
   - move contents up: `mv eighth_version/* . && rmdir eighth_version`.
3. Check layout: `ls mls/eng/` should show `transcript_10min_train.txt`, `transcript_10min_dev.txt`, `transcript_10min_test.txt`, and a `wav/` folder.  
4. From the repo root, the recipe uses `data/ml_superb` by default. If you extracted elsewhere, set:
   ```bash
   export MLSUPERB=/path/to/unzipped/root
   ```
   or symlink: `mkdir -p data && ln -snf /path/to/unzipped/root data/ml_superb`.

### Step 3: Environment

```bash
uv sync
uv add --editable ./models/espnet
uv add s3prl
uv run python scripts/patch_s3prl_for_ssl.py
```

- `patch_s3prl_for_ssl.py` fixes s3prl for Python 3.12+ and torchaudio 2.1+ (see § SSL env below). Run it once per venv after adding s3prl.

### Step 4: Run SSL pipeline (eng1 10min)

**Quick sanity check (1 epoch, 2 iters):**

```bash
./scripts/run_ml_superb_ssl_experiments.sh --langs "eng1" --durations "10min" --no-sync --debug
```

**Full training (30 epochs, ~1.5 h on GPU):**

```bash
./scripts/run_ml_superb_ssl_experiments.sh --langs "eng1" --durations "10min" --no-sync
```

- Data prep (stages 1–4) runs first; if data is already prepared, add `--skip-data`.
- Results: `models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_10min_eng1_10min/` (decode logs, `RESULTS.md`, CER/WER in `decode_*/score_cer/result.txt`).

**More languages and 1 h:**

```bash
./scripts/run_ml_superb_ssl_experiments.sh --langs "eng1 fra1 deu1" --durations "10min 1h" --skip-data --no-sync
```

### Step 5: Big run (full grid on full data)

For the full reproduction grid (3 languages × 2 durations = 6 jobs: data prep + train + decode + score), use the dedicated script. **Ensure full ML-SUPERB data is at `data/ml_superb`** (or `$MLSUPERB`).

**One command (from repo root):**

```bash
./scripts/run_ml_superb_ssl_full.sh
```

- Applies the s3prl patch, then runs data prep for all (lang, duration) pairs, then runs SSL training + decode + score for each. Order: eng1 10min → fra1 10min → deu1 10min → eng1 1h → fra1 1h → deu1 1h.
- If data is already prepared (e.g. you ran data prep before), skip it: `./scripts/run_ml_superb_ssl_full.sh --skip-data`.
- Each 10min training is ~1.5–2 h on GPU; 1h training is longer. Total runtime is many hours; run in a persistent session.

**Run in background (recommended):**

```bash
# Option A: nohup (log to file)
nohup ./scripts/run_ml_superb_ssl_full.sh > full_ssl_run.log 2>&1 &

# Option B: tmux (detach with Ctrl+B then D, reattach with tmux attach)
tmux new -s ssl
./scripts/run_ml_superb_ssl_full.sh
# Ctrl+B D to detach
```

**Monitor:** Watch the current experiment’s training log:

```bash
tail -f models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_10min_eng1_10min/train.log
# (replace with the current asr_tag, e.g. train_asr_s3prl_1h_fra1_1h)
```

**Results:** Under `models/espnet/egs2/ml_superb/asr1/exp/`, one dir per run. **Index of all training runs:** [refs/SSL_TRAINING_LINKS.md](refs/SSL_TRAINING_LINKS.md) — paths and RESULTS.md links for eng1/fra1/deu1 × 10min/1h.

---

## One-liner (GPU instance, FBANK baseline)

From the **snlp repo root** (clone then run; no sox or sclite required):

```bash
./scripts/run_ml_superb_baseline.sh
```

- **Preconditions:** `uv` installed; optional GPU (CPU works with tiny/small configs). Data: set `MLSUPERB` or use default `data/ml_superb` (see Data below).
- The script runs `uv sync`, `uv add --editable ./models/espnet`, then the full pipeline (data prep → train → decode → score) for `eng1` / `10min` by default.
- **Skip sync:** `./scripts/run_ml_superb_baseline.sh --no-sync`
- **Other lang/duration/config:** `./scripts/run_ml_superb_baseline.sh --single_lang fra1 --duration 10min` or `--asr_config conf/tuning/train_asr_fbank_single_tiny.yaml`

Scoring uses a built-in sclite-compatible CER/WER script when system sclite is not installed.

### Monitor live (training progress)

Training logs go to a file, so the main terminal may not stream updates. To watch progress in real time:

1. **Second terminal** – from the **asr1** dir (`models/espnet/egs2/ml_superb/asr1`):
   ```bash
   tail -f exp/asr_train_asr_fbank_single_eng1_10min/train.log
   ```
   (Replace the `exp/...` path with your `asr_tag` if you changed config or lang.)

2. **Unbuffered Python** – `path.sh` sets `PYTHONUNBUFFERED=1` so log updates appear as they are written when you `tail -f`.

3. **Single terminal** – run the pipeline in the background and follow the log:
   ```bash
   cd models/espnet/egs2/ml_superb/asr1
   . ./path.sh && . ./cmd.sh && . ./db.sh
   ./run_one_lang.sh --single_lang eng1 --duration 10min &
   tail -f exp/asr_train_asr_fbank_single_eng1_10min/train.log
   ```

## 1. Environment

From the **snlp repo root**:

```bash
# install project deps (creates/updates .venv)
uv sync

# add local ESPnet so the recipe has espnet2 + all ASR deps (kaldiio, torch_complex, editdistance, etc.)
uv add --editable ./models/espnet
```

- The recipe is run from `models/espnet/egs2/ml_superb/asr1`; `local/path.sh` activates the snlp `.venv` and sets `PYTHONPATH` to `models/espnet`, so `python` in the recipe uses the uv-managed env.
- Optional (PyPI espnet only, no local clone): `uv sync --extra espnet` (see `pyproject.toml`).

## 2. Data

- **Default data dir:** `data/ml_superb` (see `models/espnet/egs2/ml_superb/asr1/db.sh`). Override with `export MLSUPERB=/path/to/unzipped/root`.
- **Expected layout (e.g. eng1):**  
  `$MLSUPERB/mls/eng/` must contain `transcript_10min_train.txt`, `transcript_10min_dev.txt`, `transcript_10min_test.txt`, and `wav/<utt_id>.wav`. Same idea for other datasets (e.g. `voxforge`, `commonvoice`) and language codes. If you see "0 languages processed" or "no utterances remained", the data dir is wrong or empty.
- **How to get data:** see **Full reproduction** above: run `./scripts/download_mlsuperb_data.sh` (Option A) or follow the manual download/extract steps (Option B).

## 3. Run one (lang, duration)

From **`models/espnet/egs2/ml_superb/asr1`**:

```bash
. ./path.sh && . ./cmd.sh && . ./db.sh

# Data prep + train + decode (full run)
./run_one_lang.sh --single_lang eng1 --duration 10min

# Only data prep (stages 1–2)
./run_one_lang.sh --single_lang eng1 --duration 10min --stage 1 --stop_stage 2

# FBANK baseline (default). For HuBERT/SSL use:
./run_one_lang.sh --single_lang eng1 --duration 10min --asr_config conf/tuning/train_asr_s3prl_10min.yaml
```

Results go to `exp/<asr_tag>/` and decode logs; CER is in the scoring output.

## 4. SSL experiments (HuBERT frozen + CTC) — one script

**Debug run:** Use `--debug` for a quick pipeline check (1 epoch, 2 iters). Full training: run without `--debug`.

From the **snlp repo root**, one script runs data prep + SSL training + decode for multiple languages and durations (project requirement: *pretrained SSL, freeze parameters, CTC, 10 min / 1 h*):

```bash
# Default: eng1, 10min only (quick validation)
./scripts/run_ml_superb_ssl_experiments.sh

# Multiple languages and 10min + 1h
./scripts/run_ml_superb_ssl_experiments.sh --langs "eng1 fra1 deu1" --durations "10min 1h"

# Data already prepared: only train + decode
./scripts/run_ml_superb_ssl_experiments.sh --skip-data --langs "eng1 fra1 deu1" --durations "10min 1h"

# Preview without running
./scripts/run_ml_superb_ssl_experiments.sh --dry-run --langs "eng1 fra1" --durations "10min 1h"
```

- Uses `conf/tuning/train_asr_s3prl_10min.yaml` and `train_asr_s3prl_1h.yaml`: **freeze_param: [frontend.upstream]**, **ctc_weight: 1.0**, upstream **hubert_large_ll60k**.
- Experiments go to `models/espnet/egs2/ml_superb/asr1/exp/<asr_tag>/` (e.g. `exp/train_asr_s3prl_10min_eng1_10min/`). Watch: `tail -f exp/<asr_tag>/train.log`.
- Optional: `--no-sync` to skip `uv sync` and `uv add --editable ./models/espnet`.

### SSL env (Python 3.12+, torchaudio 2.10)

`pyproject.toml` pins `setuptools>=69,<82` and optional deps `soxr`, `tensorboard`. If you hit S3PRL import errors (e.g. `AttributeError: module 'torchaudio' has no attribute 'set_audio_backend'`) after a fresh `uv add s3prl` or `uv sync`, run from repo root:

```bash
uv run python scripts/patch_s3prl_for_ssl.py
```

That script patches the installed `s3prl` in the current venv. Alternatively, apply by hand under `.venv/lib/python3.*/site-packages/s3prl/`:

1. **`upstream/byol_s/byol_a/common.py`** – wrap `torchaudio.set_audio_backend("sox_io")` in `if hasattr(torchaudio, "set_audio_backend"): ...` (removed in torchaudio 2.1+).
2. **`upstream/roberta/roberta_model.py`** – use `field(default_factory=...)` for any dataclass field whose default is a mutable type (e.g. `encoder`, `decoder`, `quant_noise`).
3. **`upstream/mos_prediction/expert.py`** – `from torchaudio.sox_effects import apply_effects_tensor` in a try/except, set to `None` on ImportError.

## 5. Incremental reproduction plan

1. **One language, 10 min, FBANK:**  
   `./run_one_lang.sh --single_lang eng1 --duration 10min`  
   Compare CER to ML-SUPERB (2023) paper.

2. **Same with HuBERT (SSL):**  
   `./run_one_lang.sh --single_lang eng1 --duration 10min --asr_config conf/tuning/train_asr_s3prl_10min.yaml`  
   Or use `./scripts/run_ml_superb_ssl_experiments.sh` (default = eng1 10min).

3. **Add 2–3 more languages:**  
   `./scripts/run_ml_superb_ssl_experiments.sh --langs "eng1 fra1 deu1" --durations "10min 1h"`

4. **Document** commands and CER in a short table for Vadim/Bruny and for the report.

## Files touched for this setup

- `scripts/run_ml_superb_ssl_experiments.sh` — SSL (HuBERT frozen + CTC) experiments: data prep + train + decode for multiple langs/durations; run from repo root.
- `scripts/run_ml_superb_baseline.sh` — one-liner entrypoint (sync + run from repo root), if present.
- `models/espnet/egs2/ml_superb/asr1/db.sh` — sets `MLSUPERB` (default: `data/ml_superb`).
- `models/espnet/egs2/ml_superb/asr1/path.sh` — real file so `MAIN_ROOT` points to espnet root; adds `local/bin` for sclite wrapper.
- `models/espnet/egs2/ml_superb/asr1/local/path.sh` — activates snlp `.venv` and sets `PYTHONPATH` to `models/espnet`.
- `models/espnet/egs2/ml_superb/asr1/local/bin/sclite` — wrapper that runs Python CER scoring when sclite is not installed.
- `models/espnet/egs2/ml_superb/asr1/local/score_cer_sclite.py` — sclite-format CER from ref.trn / hyp.trn.
- `models/espnet/egs2/ml_superb/asr1/local/single_lang_data_prep.py`, `local/data_prep.py` — wav.scp uses direct wav paths (no sox).
- `models/espnet/egs2/ml_superb/asr1/run_one_lang.sh` — single (lang, duration) run.
- `scripts/download_mlsuperb_data.sh` — download/data dir instructions.
- `data/ml_superb` — default data directory (create with script or manual extract).
