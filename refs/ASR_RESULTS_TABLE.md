# ASR results

Canonical numbers come from `models/espnet/egs2/ml_superb/asr1/exp/<exp>/RESULTS.md` (ESPnet `show_asr_result`). **Do not** overwrite this file by running `scripts/collect_asr_results.py` alone — that script only auto-fills monolingual `eng1` rows; multilingual sections below are maintained here.

## 1) Monolingual (eng1, 10 min, test_10min_eng1)

| Frontend | CER | WER | Note |
|----------|-----|-----|------|
| HuBERT (frozen) | 33.33% | 24.14% | S3PRL |
| JEPA minimal | 62.22% | 44.83% |  |
| WavJEPA (HF) | 33.33% | 24.14% |  |
| WavJEPA (local pretrain ckpt) | 33.33% | 24.14% |  |

## 2) Multilingual HuBERT ASR-only (partial MLSUPERB tree)

Three languages in prep when only mls/swc/voxforge are present; not the full 143-lang paper setup. See `refs/MULTILINGUAL_RECAP.md`.

| Setting | Test CER | Test WER | Exp tag (under `exp/`) |
|---------|----------|------------|-------------------------|
| Multilingual ASR, 10 min | 24.96% | 23.48% | `asr_train_asr_s3prl_10min_multilingual_10min` |
| Multilingual ASR, 1 h | 20.76% | 18.30% | `asr_train_asr_s3prl_1h_multilingual_1h` |

## 3) Multilingual ASR+LID (partial MLSUPERB tree)

| Setting | Test CER | Test WER | Exp tag | Status |
|---------|----------|----------|---------|--------|
| ASR+LID, 10 min (`test_10min_lid`) | 26.33% | 25.49% | `asr_train_asr_s3prl_10min_multilingual_10min_lid` | DONE |
| ASR+LID, 1 h | — | — | `asr_train_asr_s3prl_1h_multilingual_1h_lid` | **Not completed** (run stopped; no `RESULTS.md`) |

## 4) Multilingual LoRA / PEFT (HuBERT + LoRA adapters, ASR-only)

Requires `loralib` in the environment (`pyproject.toml` includes it). Same partial-3-lang data as section 2.

| Setting | Test CER | Test WER | Exp tag | Status |
|---------|----------|----------|---------|--------|
| LoRA, 10 min (`test_10min`) | 24.95% | 23.66% | `asr_train_asr_s3prl_lora_10min_multilingual_10min` | DONE |
| LoRA, 1 h | — | — | `asr_train_asr_s3prl_lora_1h_multilingual_1h` | Pending until `RESULTS.md` exists |

**Versus frozen multilingual ASR 10 min (section 2):** CER is effectively tied (24.95% vs 24.96%); WER is marginally higher with LoRA (23.66% vs 23.48%).

**Runtimes (GPU wall time):** see `refs/GPU_RUNTIME_TABLE.md`.
