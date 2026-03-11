# ML-SUPERB minimal run report

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

## Summary

- **Train + decode + score:** Full pipeline runs with uv and local espnet; no sox/sclite required; wav.scp uses direct paths; scoring uses Python CER drop-in.

---

## Progress evaluation (percentage of project advancements)

Reference: project goal = ML-SUPERB procedure with **pretrained SSL (HuBERT/wav2vec), frozen backbone, CTC, 10 min / 1 h** (supervisor + rendu1). Janis track also includes JEPA integration and ABX vs ASR comparison.

| Component | Weight | Status | Notes |
|-----------|--------|--------|--------|
| Literature and design | 20% | Done | idea.md, ML-SUPERB procedure, JEPA rationale |
| Benchmark environment | 15% | Done | Pipeline runs; this report proves minimal run (train → decode → score) and documents resources |
| Data setup (10 min / 1 h) | 15% | Partial | eng1 10 min ready; multi-lang and 1 h optional |
| Reproduction (SSL) | 25% | Not done | This run is **FBANK** (tiny), not HuBERT/wav2vec frozen + CTC |
| JEPA integration | 15% | Not done | — |
| ABX + ASR comparison | 10% | Not done | Depends on SSL reproduction and JEPA |

**Overall: ~35–40%.**

- **What this report adds:** A minimal, reproducible run (~73 s, ~2 GB RAM) that validates the full pipeline and scoring; no change to the SSL/reproduction gap.
- **Next step to raise the percentage:** Run SSL (HuBERT frozen + CTC) with `train_asr_s3prl_10min.yaml` for at least eng1 10 min (e.g. `./scripts/run_ml_superb_ssl_experiments.sh`), then document CER. That would move “Reproduction (SSL)” from 0% to a large fraction and overall to ~55–65%.
