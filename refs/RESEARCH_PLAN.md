# Full research pipeline: ML-SUPERB + JEPA

This doc describes the **whole research** pipeline (data → optional encoder pretraining → ASR → eval → report) and what is implemented.

---

## Pipeline overview

1. **Data** — ML-SUPERB 10 min / 1 h per language (e.g. eng1, fra1).
2. **Encoder (frontend)** — Either **use a pretrained encoder** (HuBERT, WavJEPA) or **pretrain an encoder** yourself.
3. **ASR** — Train CTC head on frozen (or loaded) encoder with 10 min/1 h labeled data.
4. **Eval** — Decode, score CER/WER (and optionally ABX, LID).
5. **Report** — Aggregate results (e.g. `refs/rendu1.md`, RESULTS.md).

---

## Can encoder pretraining be done so far?

### HuBERT encoder pretraining — **yes, in ESPnet**

- **Where:** `espnet2/tasks/hubert.py` (HubertTask), `espnet2/tasks/ssl.py` (SSLTask with HuBERT loss).
- **How:** Use the **HuBERT pretraining** recipe (e.g. `egs2/mini_an4/ssl1` with `hubert.sh`, or a custom recipe that calls `espnet2.bin.hubert_train` / `ssl_train`). You get a checkpoint; then you can load it as encoder for ASR (e.g. FairseqHubertPretrainEncoder with `finetuning=True` or load weights into a frontend).
- **In this repo:** We do **not** run HuBERT pretraining in the ML-SUPERB egs; we use **already pretrained** HuBERT via S3PRL (`train_asr_s3prl_10min.yaml`). So “encoder pretraining” for HuBERT is **possible** in the toolkit but not wired in this project — you’d add a stage that runs `hubert_train` (or ssl_train) on your data and then points ASR to that checkpoint.

### JEPA encoder pretraining — **not in this repo**

- **Current setup:** We use JEPA only as a **frozen frontend** in ASR:
  - **Minimal encoder** (`jepa`): small mel-patch encoder, **no pretraining** (random init).
  - **WavJEPA-Nat** (`wavjepa`): **pretrained** encoder loaded from Hugging Face (`labhamlet/wavjepa-nat-base`).
- **To pretrain a JEPA encoder yourself:** There is **no** JEPA pretraining task in ESPnet (no JEPA loss, no predictor head training). You would:
  1. **Option A:** Use an external codebase (e.g. [Audio-JEPA](https://arxiv.org/abs/2507.02915) when released, or [Sony audio-representations](https://github.com/SonyCSLParis/audio-representations)) to pretrain a JEPA encoder on audio, then load that checkpoint into our JEPA frontend via `frontend_conf.checkpoint_path` (and optionally `jepa_repo_path` for Sony).
  2. **Option B:** Implement a JEPA SSL task in ESPnet (encoder + predictor, joint-embedding loss) and a small recipe — non-trivial and not done so far.

**Summary:** Encoder pretraining is **done so far** only in the sense that we **use** pretrained encoders (HuBERT via S3PRL, WavJEPA via HF). **Running** encoder pretraining in this repo: **HuBERT yes** (via ESPnet hubert/ssl tasks, not yet in ml_superb); **JEPA no** (use external pretraining + load checkpoint, or add a new task later).

---

## Pretraining a full JEPA model: can we? GPU? Time?

**Yes, you can pretrain a whole JEPA** — but not in this repo. You use an external codebase (e.g. [WavJEPA](https://github.com/labhamlet/wavjepa), [Audio-JEPA](https://arxiv.org/abs/2507.02915) when released) and then load the checkpoint into our frontend.

### WavJEPA-Nat (paper / HF card) — reference numbers

| Item | Value |
|------|--------|
| **Model size** | ~0.2B params (200M) |
| **GPUs** | **2 × NVIDIA H100 80GB/94GB** |
| **Batch size** | 16 per GPU → effective 256 (8× in-batch sampling) |
| **Steps** | **375,000** |
| **Precision** | Mixed (fp16/bf16), torch.compile, Flash Attention |
| **Data** | AudioSet unbalanced (1.74M × 10 s clips) + 70k Matterport3D naturalistic scenes |
| **Rough wall‑clock** | ~**4–8 days** on 2× H100 (assuming ~1–2 s/step; paper does not report exact hours) |

So at **paper scale**: you need **2× H100-class GPUs**, **AudioSet-scale data**, and on the order of **a week** of training.

### If you have less compute

- **1× H100:** Same 375k steps; expect ~**8–16 days** (steps take ~2× longer with half the batch, or you halve batch and train longer).
- **1× A100 40GB:** Possible with smaller batch (e.g. 8); 375k steps can be **~2–3 weeks**; you may need gradient checkpointing and/or a slightly smaller model to fit.
- **1× V100 32GB / RTX 3090:** Smaller batch (4–8), longer time (**weeks to a month+**); consider fewer steps (e.g. 100k) for a “small JEPA” experiment, with lower expected quality.
- **Academic clusters:** Check if your lab has H100/A100 multi-GPU nodes; 2× A100 80GB for 375k steps is a plausible “medium” setup (~1–2 weeks).

### Data

- **Full replication:** AudioSet (1.74M clips) + Matterport3D/RIR simulation for WavJEPA-Nat. AudioSet is public but large; preparation (download, 16 kHz, manifest) is a separate step.
- **Smaller experiment:** You could pretrain on a subset (e.g. 100k–500k clips) for fewer steps to test the pipeline; results will be weaker than the paper.

### Summary

| Question | Answer |
|----------|--------|
| **Can we pretrain a whole JEPA?** | Yes, using external code (WavJEPA repo or Audio-JEPA when available). Not implemented in this snlp repo. |
| **GPU capabilities** | **Paper setup:** 2× H100 80/94 GB. **Minimum sensible:** 1× A100 40GB (smaller batch, longer time). **Tight budget:** 1× V100/3090 with small batch and possibly fewer steps. |

### Using the WavJEPA repo from this project (multi-GPU, e.g. 10× H100)

We **clone the [WavJEPA](https://github.com/labhamlet/wavjepa) repo** and add it as an editable library so you can run pretraining from here with many GPUs.

1. **Clone and add as library**
   ```bash
   ./scripts/setup_wavjepa.sh
   ```
   This clones `labhamlet/wavjepa` into `third_party/wavjepa` and runs `uv add --editable third_party/wavjepa`. Optional: `./scripts/setup_wavjepa.sh --no-uv` to only clone.

2. **Install WavJEPA training deps** (in the same venv or in `third_party/wavjepa`)
   ```bash
   cd third_party/wavjepa && uv pip install -r requirements.txt
   ```
   (Or use the venv where you did `uv sync --extra wavjepa-pretrain` after cloning.)

3. **Run pretraining** (e.g. 10× H100, AudioSet)
   ```bash
   ./scripts/run_wavjepa_pretrain.sh --num-gpus 10 --data audioset --save-dir /path/to/checkpoints
   ```
   Quick test on 2 GPUs with LibriSpeech: `./scripts/run_wavjepa_pretrain.sh --num-gpus 2 --data librispeech`.

4. **Data:** Set `configs/data/audioset.yaml` (or the data config you use) with `base_data_dir` / `val_data_dir`; for AudioSet you can use e.g. HuggingFace `agkphysics/AudioSet` and point the config to the extracted paths.

With **10× H100** you can increase batch size (e.g. `trainer.batch_size=64`) and/or run 375k steps; wall-clock will be shorter than the paper’s 2× H100 setup.

### Full pipeline at scale (pretrain + ASR in one go)

One script runs **optional WavJEPA pretraining** then the **ASR pipeline** (download ML-SUPERB, data prep, 30ep train, eval):

```bash
# ASR only (same as run_full_pipeline.sh)
./scripts/run_full_pipeline_at_scale.sh

# WavJEPA pretraining on 10 GPUs (AudioSet) then ASR
./scripts/run_full_pipeline_at_scale.sh --pretrain-gpus 10 --pretrain-data audioset

# Quick pretrain test (2 GPUs, LibriSpeech) then ASR with existing data
./scripts/run_full_pipeline_at_scale.sh --pretrain-gpus 2 --pretrain-data librispeech --skip-download
```

Pretraining checkpoints go to `logs/wavjepa_pretrain/` (or `--pretrain-save-dir`). To **use your pretrained encoder in ASR**: the current WavJEPA frontend loads from HuggingFace only; loading from a local checkpoint would require adding a `checkpoint_path` (or similar) in `frontend_conf` and mapping the saved state dict to the encoder (WavJEPA’s training saves the full JEPA; you need the encoder subset). That is optional follow-up work.

---

## What the scripts do (full research run)

- **Data:** `scripts/download_mlsuperb_data.sh`, then `run_one_lang.sh --stage 1 --stop_stage 4` (or `run_ml_superb_*.sh` without `--skip-data`).
- **ASR (no pretraining):** Train ASR with one of:
  - HuBERT (S3PRL): `train_asr_s3prl_10min.yaml`
  - JEPA minimal: `train_asr_jepa_10min.yaml`
  - WavJEPA (HF): `train_asr_wavjepa_10min.yaml`
  - All three: `scripts/run_ml_superb_train_eval_all.sh --full`
- **Eval:** Decode + score are part of the same run (stages 12–13). Results in `exp/<asr_tag>/RESULTS.md`.
- **Report:** Update `refs/rendu1.md` (and optionally a table in `refs/report.md`) with CER/WER from each exp.

A single script that chains data → train (all configs) → report is `scripts/run_research_full.sh` (see below).

---

## Single command: run the full pipeline

**One-liner (download + data prep + train + eval)** — for big GPU instances, run from repo root:

```bash
# Full: download ML-SUPERB → sync deps → data prep → 30ep train (HuBERT/JEPA/WavJEPA) → eval → summary
./scripts/run_full_pipeline.sh

# Quick sanity check: download + 1 epoch, 2 iters per config
./scripts/run_full_pipeline.sh --debug

# Data already present (e.g. data/ml_superb/mls)
./scripts/run_full_pipeline.sh --skip-download

# Data already prepared (skip stages 1–4)
./scripts/run_full_pipeline.sh --skip-download --skip-data

# Faster reruns (skip uv sync)
./scripts/run_full_pipeline.sh --skip-download --no-sync
```

Data is written to `data/ml_superb`, results to `models/espnet/egs2/ml_superb/asr1/exp/*/RESULTS.md` and `logs/research_results_*.txt`. Prerequisites: `uv`, `unzip`; for gated datasets run `huggingface-cli login` first.

**Without download** (you set `MLSUPERB` yourself):

```bash
# Full pipeline: data prep + train (JEPA + WavJEPA) + eval; no HuBERT if S3PRL missing
./scripts/run_research_full.sh

# Skip data prep (already done)
./scripts/run_research_full.sh --skip-data

# Quick run (1 epoch, 2 iters per config) to verify
./scripts/run_research_full.sh --debug --skip-data
```

These do **not** run encoder pretraining; they use existing pretrained frontends (and the minimal JEPA encoder without pretraining). To add HuBERT pretraining as a stage you’d add a call to the ESPnet HuBERT/SSL recipe and pass the resulting checkpoint to the ASR config.
