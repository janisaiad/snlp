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

- **Default data dir:** `data/ml_superb` (set in `models/espnet/egs2/ml_superb/asr1/db.sh`). Override with:
  ```bash
  export MLSUPERB=/path/to/unzipped/ml_superb
  ```

- **Expected layout (e.g. for default eng1):**  
  `$MLSUPERB/mls/eng/` must exist and contain `transcript_10min_train.txt`, `transcript_10min_dev.txt`, `transcript_10min_test.txt`, and `wav/<utt_id>.wav`. If you see "0 languages processed" or "no utterances remained", the data dir is missing or empty on that machine.
- **Download:**  
  - Run: `./scripts/download_mlsuperb_data.sh`  
  - Or manually: download ML-SUPERB 8th from [Huggingface](https://huggingface.co/datasets/ftshijt/mlsuperb_8th) or [Google Drive](https://drive.google.com/file/d/1vQ5NksmGl-lY7I4mlU4Kde3EhrEYGii2/view), then extract so that `$MLSUPERB` contains dataset folders (e.g. `mls`, `voxforge`, `commonvoice`) with per-language subdirs and the transcript + wav files above.

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

`pyproject.toml` pins `setuptools>=69,<82` and optional deps `soxr`, `tensorboard`. If you hit S3PRL import errors after a fresh `uv sync`, apply these patches under `.venv/lib/python3.*/site-packages/s3prl/` (or re-use a venv where they were applied):

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
