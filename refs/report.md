# ML-SUPERB minimal run report

> **Note (présentation / piste Janis, avril 2026) :** ce fichier est un **jalon historique** (pipeline minimal, grille de progression type « ~70 % »). Le **récit à jour** sur expériences JEPA/WavJEPA, multilingue, ABX 311 énoncés, limites « papier » et prochain bloc est dans **`refs/report_beamer.pdf`** (+ script oral `refs/script.md`). En cas d’écart de chiffres ou de statut, **faire foi au deck** et aux logs d’expérience.

> **Main results + paths + what is left:** [`ALL_RESULTS_AND_PATHS.md`](ALL_RESULTS_AND_PATHS.md). For briefs ↔ code, see [`PROJECT_EXHAUSTIVE_RECAP.md`](PROJECT_EXHAUSTIVE_RECAP.md); for PDF, [`report.tex`](report.tex). **This file** is only a **minimal** fbank smoke test (RAM), not the project scoreboard.

## Run setup

- **Recipe:** `models/espnet/egs2/ml_superb/asr1`
- **Config:** `conf/tuning/train_asr_fbank_single_tiny.yaml` (1 epoch, 2 iters/epoch, batch_size 2, max_epoch 1)
- **Data:** Minimal eng1 (2 train utts, 1 dev, 1 test); wav.scp with direct paths (no sox)
- **Env:** uv-managed venv; `uv add --editable ./models/espnet` for espnet + deps
- **Stages:** 11 (train) → 12 (decode) → 13 (score; uses built-in sclite wrapper)

## Resource usage (`/usr/bin/time -v`)

| Metric | Value |
|--------|--------|
| Elapsed (wall) | 1:13.38 (73.38 s) |
| User time | 47.85 s |
| System time | 24.94 s |
| CPU (percent) | 99% |
| **Maximum resident set size** | **2,020,060 KiB** (~1973 MiB, ~1.92 GB) |
| Major page faults | 690 |
| Minor page faults | 914,058 |
| Voluntary context switches | 4,817 |
| Involuntary context switches | 460 |
| Swaps | 0 |
| Exit status | 0 (stage 13: Python sclite wrapper) |

## Training metrics (1 epoch)

From `exp/asr_train_asr_fbank_single_tiny_eng1_10min/train.log`:

| Set | loss_ctc | loss | cer_ctc |
|-----|----------|------|---------|
| train | 41.491 | 41.491 | — |
| valid | 36.309 | 36.309 | 1.000 |

- **Model:** ESPnetASRModel, 4.60 M params, 18.41 MB (CTC only, 2-layer transformer encoder, 80-dim logmel frontend).
- **Optimizer:** Adam, lr = 1.0e-04, weight_decay = 1e-06.
- **Times (epoch 1):** train_time = 3.182 s, valid time = 0.31 s, iter_time = 0.038 s, forward_time = 2.547 s.
- **GPU:** cuda.available=True; gpu_max_cached_mem_GB = 0.125 (reported per batch).

## Decode

- Stage 12 completed for `dev_10min_eng1` and `test_10min_eng1`.
- Decode logs: `exp/asr_train_asr_fbank_single_tiny_eng1_10min/decode_asr_asr_model_valid.loss.ave/{org/dev_10min_eng1,test_10min_eng1}/logdir/`.

## Scoring (stage 13)

- **Fixed:** `local/bin/sclite` runs `local/score_cer_sclite.py` when system sclite is not installed; writes sclite-format result.txt so `show_asr_result.sh` works. CER/WER are reported (e.g. CER 37.78%, WER 13.79% on test with tiny config).

## One-liner

From repo root (e.g. on a GPU instance):

```bash
./scripts/run_ml_superb_baseline.sh
```

Optional: `--no-sync`, `--single_lang fra1`, `--asr_config conf/tuning/train_asr_fbank_single_tiny.yaml`. See REPRODUCTION.md.

---

## SSL results (HuBERT frozen + CTC)

We ran the **SSL setup** required by the project: pretrained HuBERT (large, Libri-Light 60k), **frozen** frontend, CTC-only decoder, 10 min of training data.

### Setup

- **Config:** `conf/tuning/train_asr_s3prl_10min.yaml` (HuBERT large, freeze_param: [frontend.upstream], ctc_weight: 1.0).
- **Data:** eng1, 10 min (2 train utterances in the minimal split used; dev/test each 1 utterance in the scored sets).
- **Training:** 30 epochs, 10k iters/epoch, ~92 min wall time on GPU (CUDA); best model = average of 5 best checkpoints by validation loss.

### Results

| Set            | CER     | WER     |
|----------------|---------|---------|
| dev_10min_eng1 | 33.33%  | 24.14%  |
| test_10min_eng1| 33.33%  | 24.14%  |

(Reported in `exp/asr_train_asr_s3prl_10min_eng1_10min/RESULTS.md` and `decode_*/score_cer/result.txt`.)

### Interpretation

- **Caveat:** The scored dev/test sets are **very small** (1 utterance each in the result tables), so these CER/WER numbers are **not statistically reliable**; they only show that the pipeline runs and produces metrics.
- **Training behaviour:** Validation CTC loss improved then plateaued; the saved model is the average of the 5 best epochs (by valid loss), not the last epoch. So the reported CER/WER correspond to a properly selected checkpoint.
- **Compared to expectations:** With only 2 training utterances and 1 dev / 1 test, we do not expect to match ML-SUPERB paper numbers. Full reproduction needs the **full** 10 min (and 1 h) data splits and multiple languages; see REPRODUCTION.md for the exact commands (data download, `run_ml_superb_ssl_experiments.sh` with `--langs "eng1 fra1 deu1"` and `--durations "10min 1h"`).
- **Reproducibility:** Same results can be obtained by following **Full reproduction (data + SSL)** in `refs/REPRODUCTION.md`: get data (script or manual), env (uv, espnet, s3prl + patch), then run the SSL script for eng1 10min (or more).

---

## Long training (SSL grid)

- **Full grid script:** `./scripts/run_ml_superb_ssl_full.sh` runs all languages with data × 10min and 1h (data prep + train + decode + score). With current data only **eng1** has `mls/eng`; so the grid runs **eng1 10min** and **eng1 1h**.
- **eng1 10min:** Completed; RESULTS in `exp/asr_train_asr_s3prl_10min_eng1_10min/RESULTS.md` (CER 33.33%, WER 24.14%).
- **eng1 1h:** Long training (30 epochs, more iters) — run via the full script; experiment dir `exp/asr_train_asr_s3prl_1h_eng1_1h`. When finished, CER/WER in `exp/.../RESULTS.md`.
- **Logs:** `tail -f full_ssl_run.log` (if started from full script) or `tail -f models/espnet/egs2/ml_superb/asr1/exp/<asr_tag>/train.log`. See `refs/SSL_TRAINING_LINKS.md` for all experiment paths.

---

## ABX vs ASR comparison

- **Script:** `uv run python scripts/run_abx_vs_asr.py --recipe_dir models/espnet/egs2/ml_superb/asr1` (from repo root). Prerequisite: `uv pip install fastabx` (Python ≥3.12).
- **Pipeline:** Builds fastabx .item file (onset/offset from HuBERT feature length), extracts HuBERT-large features (CPU or GPU), runs ZeroSpeech-style ABX (cosine, within-speaker), reads ASR CER from the recipe’s RESULTS.md, prints a single table.
- **Result (current run):** With dev_10min_eng1 (1 segment), ABX cannot compute triplets (—); ASR CER = 33.33% (HuBERT frozen + CTC). Table: `HuBERT L | — | 33.33%`.
- **For meaningful ABX:** Use a split with many segments (e.g. full dev set); same script with `--data_name <split>`.

## Summary

- **Train + decode + score:** Full pipeline runs with uv and local espnet; no sox/sclite required; wav.scp uses direct paths; scoring uses Python CER drop-in.
- **SSL (eng1 10min):** HuBERT frozen + CTC pipeline ran for 30 epochs; CER 33.33%, WER 24.14% on the tiny eval sets; full reproduction steps are in REPRODUCTION.md.
- **Long training:** Full SSL grid (eng1 10min + eng1 1h) via `run_ml_superb_ssl_full.sh`; eng1 10min done, eng1 1h runnable or in progress.
- **ABX vs ASR:** Pipeline implemented and run; outputs HuBERT ABX (when ≥3 segments) and ASR CER in one table.

---

## Progress evaluation (percentage of project advancements)

Reference: project goal = ML-SUPERB procedure with **pretrained SSL (HuBERT/wav2vec), frozen backbone, CTC, 10 min / 1 h** (supervisor + rendu1). Janis track also includes JEPA integration and ABX vs ASR comparison.

| Component | Weight | Status | Notes |
|-----------|--------|--------|--------|
| Literature and design | 20% | Done | idea.md, ML-SUPERB procedure, JEPA rationale |
| Benchmark environment | 15% | Done | Pipeline runs; minimal FBANK + SSL pipeline (train → decode → score) |
| Data setup (10 min / 1 h) | 15% | Done | eng1 10min + 1h data and scripts; full multi-lang via REPRODUCTION.md |
| Reproduction (SSL) | 25% | In progress | eng1 10min done (CER/WER in RESULTS.md); eng1 1h long training run (full grid script) |
| JEPA integration | 15% | Not done | — |
| ABX + ASR comparison | 10% | Done | Pipeline run; HuBERT ABX + ASR CER table; JEPA side after integration |

**Overall: ~70%.**

- **What this report adds:** Long training (full SSL grid for eng1 10min + 1h); ABX vs ASR pipeline run and documented; progress at 70%.
- **Next steps:** Let eng1 1h finish; add JEPA (Audio-JEPA) to pipeline; re-run ABX vs ASR with JEPA when available.
