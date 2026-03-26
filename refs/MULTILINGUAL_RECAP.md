# Recap: multilingual ML-SUPERB, LID, and LoRA (HuBERT)

This document summarizes what was **implemented**, what **finished**, and what was **still running** at the time of the consolidated report (March 2026). It complements the monolingual `eng1` benchmark in `ASR_RESULTS_TABLE.md`.

## Data scope (important)

The multilingual recipe expects the full ML-SUPERB corpus tree under `MLSUPERB` (default: `data/ml_superb`). On the machine used for this work, only a **partial** extract was present (**mls**, **swc**, **voxforge**), so data prep produced **three languages** (eng, deu, fra) rather than the full 143-language paper setup. **Multilingual numbers are reproducible only with the same data layout.**

## Recipe and code changes

| Item | Location | Purpose |
|------|----------|---------|
| Distinct experiment dirs per track | `models/espnet/egs2/ml_superb/asr1/run_multi.sh` | `asr_tag` and `asr_stats_dir` use suffixes `""`, `_only_lid`, `_lid` so ASR-only / LID-only / ASR+LID runs do not overwrite each other. |
| Partial corpus tree | `models/espnet/egs2/ml_superb/asr1/local/data_prep.py` | Skip missing top-level corpus folders with a warning. |
| LoRA configs (HuBERT large) | `conf/tuning/train_asr_s3prl_lora_10min.yaml`, `train_asr_s3prl_lora_1h.yaml` | Multilingual ASR with LoRA on selected attention projections. |
| Full queue (sequential, 1 GPU) | `scripts/run_ml_superb_multilingual_peft_queue.sh` | Order: ASR-only 10m/1h → LID-only 10m/1h → ASR+LID 10m/1h → LoRA 10m/1h; then `collect_asr_results.py`. |
| Resume after reboot | `scripts/run_ml_superb_multilingual_peft_resume.sh` | Same order; **skips** steps that already have `exp/asr_${asr_tag}/RESULTS.md`. |
| Documentation | `refs/MULTILINGUAL_PEFT_QUEUE.md` | Launch and monitor commands. |

Logs: `logs/ml_superb_multilingual_peft/` (`master.log`, `multi_*.log`, `collect.log`, `resume.nohup.log`).

## Queue status (from `master.log`)

Completed and logged with **DONE**:

1. Multilingual **ASR only** — 10 min, 1 h  
2. **LID only** — 10 min, 1 h  

Started (check `master.log` for **DONE**):

3. **ASR + LID** — 10 min (**DONE**), 1 h (**RUNNING**)  
4. **LoRA multilingual** — 10 min, 1 h (**pending**)  
5. **`uv run python scripts/collect_asr_results.py`** — runs once at the **end** of the queue script (**pending**).

## Published metrics (multilingual ASR, test sets)

From `exp/asr_train_asr_s3prl_*_multilingual_*/RESULTS.md` (ESPnet `show_asr_result` tables):

| Setting | Duration | Test CER (%) | Test WER (%) |
|---------|----------|--------------|----------------|
| Multilingual ASR (HuBERT frozen, S3PRL) | 10 min | 24.96 | 23.48 |
| Multilingual ASR (HuBERT frozen, S3PRL) | 1 h | 20.76 | 18.30 |

**LID-only** runs produce `RESULTS.md` with **no word-level ASR reference** in the standard table (word count 0); interpret LID using task-specific scores / decode logs, not WER/CER from that file.

**ASR+LID 10 min** result is now available:

| Setting | Duration | Test CER (%) | Test WER (%) |
|---------|----------|--------------|--------------|
| Multilingual ASR+LID (HuBERT frozen, S3PRL) | 10 min | 26.33 | 25.49 |

**Pending rows** should be added once `RESULTS.md` exists under:

- `exp/asr_train_asr_s3prl_10min_multilingual_10min_lid/`
- `exp/asr_train_asr_s3prl_1h_multilingual_1h_lid/`
- `exp/asr_train_asr_s3prl_lora_10min_multilingual_10min/`
- `exp/asr_train_asr_s3prl_lora_1h_multilingual_1h/`

## Checkpoints

Training directories store `epoch.pth`, `latest.pth`, `checkpoint.pth`, and `valid.loss.best.pth` under each `exp/asr_train_*/` folder; ESPnet is run with `--resume true` when continuing.

## GPU and wall-clock time

Runs used **one GPU per job** (`--ngpu 1`, queue `ML_SUPERB_NGPU=1`) on host **`janis-gpuL4-48`** (NVIDIA **L4** class — confirm with `nvidia-smi`).

- **Full queue step duration** (prep + train + decode): see **`logs/ml_superb_multilingual_peft/master.log`** between each `START` and `DONE`.
- **Training subprocess only**: footer of each **`exp/.../train.log`** (`elapsed time … seconds`).

Consolidated tables (multilingual queue, monolingual, fra1/deu1 matrix): **`refs/GPU_RUNTIME_TABLE.md`**.

## How to reproduce (after clone)

```bash
cd /path/to/snlp
export PATH="$(pwd)/.venv/bin:$PATH"
export MLSUPERB="${MLSUPERB:-$(pwd)/data/ml_superb}"
nohup ./scripts/run_ml_superb_multilingual_peft_resume.sh >> logs/ml_superb_multilingual_peft/resume.nohup.log 2>&1 &
```

Full queue from scratch (re-runs finished jobs too):

```bash
nohup ./scripts/run_ml_superb_multilingual_peft_queue.sh >> logs/ml_superb_multilingual_peft.nohup.log 2>&1 &
```
