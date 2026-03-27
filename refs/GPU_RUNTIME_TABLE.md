# GPU / wall-clock runtime documentation

This file records **how long jobs occupied the GPU** on the development machine, and **how the numbers were measured**.

## Hardware (this workspace)

- **Host:** `janis-gpuL4-48` (from ESPnet logs).
- **GPU:** **NVIDIA L4** (inferred from hostname; verify with `nvidia-smi` on the machine).
- **Typical job shape:** ESPnet `asr_train` with **`--ngpu 1`** (one process group on one GPU). The multilingual queue script sets **`ML_SUPERB_NGPU=1`**.

## What “time on GPU” means here

| Source | What it measures |
|--------|------------------|
| **`exp/.../train.log` footer** | **Wall time of the training subprocess** only: lines like `# Ended (code 0) at ..., elapsed time NNNN seconds` (from ESPnet `run.pl`). This is the main GPU-bound phase. |
| **`logs/ml_superb_multilingual_peft/master.log`** | **Wall time of the full recipe** for that queue step (stages 1–13: data prep if rerun, stats, train, decode, score). Longer than `train.log` alone. |
| **Multi-GPU** | Not used for these runs (`ngpu 1`). If you use more GPUs, **GPU·hours** = wall × number of GPUs actually utilized. |

**Note:** For a **single fully utilized GPU**, wall-clock time of the training step is a practical proxy for **GPU time** (not “SM-hours” from a profiler).

---

## Multilingual + PEFT queue (`master.log`)

Wall time between **START** and **DONE** (UTC), full pipeline per step:

| Step | Start (UTC) | End (UTC) | Wall time |
|------|-------------|-----------|-----------|
| Multilingual ASR only, 10 min | 2026-03-21 23:35:52 | 2026-03-22 05:43:19 | **~6 h 07 min** |
| Multilingual ASR only, 1 h | 2026-03-22 05:43:19 | 2026-03-22 17:19:59 | **~11 h 37 min** |
| LID-only, 10 min (resume start) | 2026-03-22 20:09:22 | 2026-03-23 01:13:12 | **~5 h 04 min** |
| LID-only, 1 h | 2026-03-23 01:13:12 | 2026-03-23 12:42:41 | **~11 h 30 min** |
| ASR+LID, 10 min | 2026-03-23 12:42:41 | (see `master.log` when **DONE** appears) | *in progress or pending at last export* |

**Training-only** comparison (from `train.log` **elapsed time**, successful `code 0`):

| Experiment directory | Training elapsed | Approx. |
|----------------------|------------------|---------|
| `asr_train_asr_s3prl_10min_multilingual_10min` | 21350 s | **~5.93 h** |
| `asr_train_asr_s3prl_1h_multilingual_1h` | 40408 s | **~11.22 h** |
| `asr_train_asr_s3prl_10min_multilingual_10min_only_lid` | 17744 s | **~4.93 h** |
| `asr_train_asr_s3prl_1h_multilingual_1h_only_lid` | 40206 s | **~11.17 h** |

---

## Monolingual / other ASR runs (`train.log` elapsed time, `code 0`)

| Experiment directory | Training elapsed | Approx. |
|----------------------|------------------|---------|
| `asr_train_asr_s3prl_1h_eng1_1h` | 14237 s | **~3.95 h** |
| `asr_train_asr_wavjepa_10min_eng1_10min` | 7166 s | **~1.99 h** |
| `asr_train_asr_wavjepa_local_ckpt_10min_eng1_10min` | 5824 s | **~1.62 h** |
| `asr_train_asr_jepa_10min_eng1_10min` | 6 s | **not meaningful** (log shows a trivial run; use RESULTS timestamp vs. an earlier full run if you retrain) |
| `asr_train_asr_s3prl_10min_eng1_10min` | 13 s, `code 1` | **failed** in current `train.log` (checkpoint mismatch); **do not** use as duration for the reported `RESULTS.md` |

## Monolingual HuBERT matrix (fra1 / deu1, `train.log`)

| Experiment directory | Training elapsed | Approx. |
|----------------------|------------------|---------|
| `asr_train_asr_s3prl_10min_fra1_10min` | 29127 s | **~8.09 h** |
| `asr_train_asr_s3prl_1h_fra1_1h` | 53074 s | **~14.74 h** |
| `asr_train_asr_s3prl_10min_deu1_10min` | 18751 s | **~5.21 h** |
| `asr_train_asr_s3prl_1h_deu1_1h` | 33993 s | **~9.44 h** |

---

## How to refresh these numbers after new runs

1. **Full queue step duration:**  
   `grep -E 'START|DONE' logs/ml_superb_multilingual_peft/master.log`

2. **Training-only duration:**  
   `grep 'elapsed time' models/espnet/egs2/ml_superb/asr1/exp/<expdir>/train.log | tail -1`

3. **GPU model:**  
   `nvidia-smi --query-gpu=name --format=csv,noheader`

---

## Linked documents

- Multilingual recap: [`MULTILINGUAL_RECAP.md`](MULTILINGUAL_RECAP.md)
- PDF report: [`report.tex`](report.tex)
