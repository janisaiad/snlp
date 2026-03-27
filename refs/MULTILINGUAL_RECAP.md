# Recap: multilingual ML-SUPERB, LID, and LoRA (HuBERT)

This document summarizes what was **implemented**, what **finished**, and what is **pending or stopped**. It complements the monolingual `eng1` benchmark in `ASR_RESULTS_TABLE.md`.

## Data scope (important)

The multilingual recipe expects the full ML-SUPERB corpus tree under `MLSUPERB` (default: `data/ml_superb`). On the machine used for this work, only a **partial** extract was present (**mls**, **swc**, **voxforge**), so data prep produced **three languages** (eng, deu, fra) rather than the full 143-language paper setup. **Multilingual numbers are reproducible only with the same data layout.**

## Recipe and code changes

| Item | Location | Purpose |
|------|----------|---------|
| Distinct experiment dirs per track | `models/espnet/egs2/ml_superb/asr1/run_multi.sh` | `asr_tag` and `asr_stats_dir` use suffixes `""`, `_only_lid`, `_lid` so ASR-only / LID-only / ASR+LID runs do not overwrite each other. |
| Partial corpus tree | `models/espnet/egs2/ml_superb/asr1/local/data_prep.py` | Skip missing top-level corpus folders with a warning. |
| LoRA configs (HuBERT large) | `conf/tuning/train_asr_s3prl_lora_10min.yaml`, `train_asr_s3prl_lora_1h.yaml` | Multilingual ASR with LoRA on selected attention projections. |
| **LoRA dependency** | `pyproject.toml` → `loralib` | ESPnet `create_lora_adapter` requires `loralib` at import time. |
| Full queue (sequential, 1 GPU) | `scripts/run_ml_superb_multilingual_peft_queue.sh` | Order: ASR-only 10m/1h → LID-only 10m/1h → ASR+LID 10m/1h → LoRA 10m/1h; then `collect_asr_results.py` (eng1-only autogen). |
| Resume after reboot | `scripts/run_ml_superb_multilingual_peft_resume.sh` | Same order; **skips** steps that already have `exp/asr_${asr_tag}/RESULTS.md`. |
| Documentation | `refs/MULTILINGUAL_PEFT_QUEUE.md` | Launch and monitor commands. |

Logs: `logs/ml_superb_multilingual_peft/` (`master.log`, `multi_*.log`, `collect.log`, `resume.nohup.log`).

## Queue status (summary, March 2026)

**Completed** (each has `RESULTS.md` where ASR WER/CER apply):

1. Multilingual **ASR only** — 10 min, 1 h  
2. **LID only** — 10 min, 1 h (use LID metrics, not ASR word tables)  
3. **ASR + LID** — **10 min only** (`test_10min_lid`)  
4. **LoRA multilingual ASR** — **10 min only** (`test_10min`)

**Stopped / incomplete:**

- **ASR+LID 1 h** — training was **stopped** before a final `RESULTS.md` for `asr_train_asr_s3prl_1h_multilingual_1h_lid`.

**Pending:**

- **LoRA 1 h** — run when `exp/asr_train_asr_s3prl_lora_1h_multilingual_1h/RESULTS.md` exists (check `master.log` for `MANUAL2 DONE lora_asr_only 1h`).

**Collector:** `uv run python scripts/collect_asr_results.py` writes **`refs/ASR_RESULTS_TABLE_eng1.md`** (monolingual). Full multilingual tables are **`refs/ASR_RESULTS_TABLE.md`** (hand-maintained).

## Published metrics (test sets)

### Multilingual ASR-only (HuBERT frozen)

| Duration | Test CER (%) | Test WER (%) |
|----------|--------------|--------------|
| 10 min | 24.96 | 23.48 |
| 1 h | 20.76 | 18.30 |

### ASR+LID (joint)

| Duration | Test CER (%) | Test WER (%) | Test set |
|----------|--------------|--------------|----------|
| 10 min | 26.33 | 25.49 | `test_10min_lid` |

### LoRA / PEFT (multilingual ASR-only)

| Duration | Test CER (%) | Test WER (%) | Test set |
|----------|--------------|--------------|----------|
| 10 min | 24.95 | 23.66 | `test_10min` |

Compared to **frozen** multilingual ASR 10 min: **CER** is effectively tied (24.95 vs 24.96); **WER** is slightly higher with LoRA (23.66 vs 23.48).

**LID-only** runs: use task-specific LID scores, not the default ASR WER/CER rows when word counts are zero.

## Checkpoints

Training directories store checkpoints under each `exp/asr_train_*/` folder; ESPnet uses `--resume true` when continuing.

## GPU and wall-clock time

Runs used **one GPU** (`--ngpu 1`) on host **`janis-gpuL4-48`** (NVIDIA **L4** class). See **`refs/GPU_RUNTIME_TABLE.md`** and **`logs/ml_superb_multilingual_peft/master.log`** (`START` / `DONE`).

## How to reproduce (after clone)

```bash
cd /path/to/snlp
export PATH="$(pwd)/.venv/bin:$PATH"
export MLSUPERB="${MLSUPERB:-$(pwd)/data/ml_superb}"
uv sync
nohup ./scripts/run_ml_superb_multilingual_peft_resume.sh >> logs/ml_superb_multilingual_peft/resume.nohup.log 2>&1 &
```

Full queue from scratch:

```bash
nohup ./scripts/run_ml_superb_multilingual_peft_queue.sh >> logs/ml_superb_multilingual_peft.nohup.log 2>&1 &
```
