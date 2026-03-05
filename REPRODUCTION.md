# ML-SUPERB reproduction (incremental)

Reproduce ML-SUPERB baseline results for the snlp project (Janis track): one language, 10 min, then extend.

**This project uses [uv](https://docs.astral.sh/uv/).** Do not use `pip` inside the venv; use `uv add` / `uv sync` only.

## One-liner (GPU instance)

From the **snlp repo root** (clone then run; no sox or sclite required):

```bash
./scripts/run_ml_superb_baseline.sh
```

- **Preconditions:** `uv` installed; optional GPU (CPU works with tiny/small configs). Data: set `MLSUPERB` or use default `data/ml_superb` (see Data below).
- The script runs `uv sync`, `uv add --editable ./models/espnet`, then the full pipeline (data prep → train → decode → score) for `eng1` / `10min` by default.
- **Skip sync:** `./scripts/run_ml_superb_baseline.sh --no-sync`
- **Other lang/duration/config:** `./scripts/run_ml_superb_baseline.sh --single_lang fra1 --duration 10min` or `--asr_config conf/tuning/train_asr_fbank_single_tiny.yaml`

Scoring uses a built-in sclite-compatible CER/WER script when system sclite is not installed.

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

- **Default data dir:** `data/ml_superb` (set in `models/espnet/egs2/ml_superb/asr1/db.sh`). Override with:
  ```bash
  export MLSUPERB=/path/to/unzipped/ml_superb
  ```

- **Download:**  
  - Run: `./scripts/download_mlsuperb_data.sh`  
  - Or manually: download ML-SUPERB 8th from [Huggingface](https://huggingface.co/datasets/ftshijt/mlsuperb_8th) or [Google Drive](https://drive.google.com/file/d/1vQ5NksmGl-lY7I4mlU4Kde3EhrEYGii2/view), then extract so that `$MLSUPERB` contains dataset folders (e.g. `mls`, `voxforge`, `commonvoice`) with per-language subdirs and files: `transcript_10min_train.txt`, `transcript_10min_dev.txt`, `transcript_10min_test.txt`, `wav/<utt_id>.wav`.

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

## 4. Incremental reproduction plan

1. **One language, 10 min, FBANK:**  
   `./run_one_lang.sh --single_lang eng1 --duration 10min`  
   Compare CER to ML-SUPERB (2023) paper.

2. **Same with HuBERT (SSL):**  
   `./run_one_lang.sh --single_lang eng1 --duration 10min --asr_config conf/tuning/train_asr_s3prl_10min.yaml`

3. **Add 2–3 more languages:** e.g. `fra1`, `deu1` with 10 min (and optionally 1 h) each.

4. **Document** commands and CER in a short table for Vadim/Bruny and for the report.

## Files touched for this setup

- `scripts/run_ml_superb_baseline.sh` — one-liner entrypoint (sync + run from repo root).
- `models/espnet/egs2/ml_superb/asr1/db.sh` — sets `MLSUPERB` (default: `data/ml_superb`).
- `models/espnet/egs2/ml_superb/asr1/path.sh` — real file so `MAIN_ROOT` points to espnet root; adds `local/bin` for sclite wrapper.
- `models/espnet/egs2/ml_superb/asr1/local/path.sh` — activates snlp `.venv` and sets `PYTHONPATH` to `models/espnet`.
- `models/espnet/egs2/ml_superb/asr1/local/bin/sclite` — wrapper that runs Python CER scoring when sclite is not installed.
- `models/espnet/egs2/ml_superb/asr1/local/score_cer_sclite.py` — sclite-format CER from ref.trn / hyp.trn.
- `models/espnet/egs2/ml_superb/asr1/local/single_lang_data_prep.py`, `local/data_prep.py` — wav.scp uses direct wav paths (no sox).
- `models/espnet/egs2/ml_superb/asr1/run_one_lang.sh` — single (lang, duration) run.
- `scripts/download_mlsuperb_data.sh` — download/data dir instructions.
- `data/ml_superb` — default data directory (create with script or manual extract).
