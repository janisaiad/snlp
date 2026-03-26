# ASR results

## 1) Monolingual (eng1, 10 min, test_10min_eng1)

| Frontend | CER | WER | Note |
|----------|-----|-----|------|
| HuBERT (frozen) | 33.33% | 24.14% | S3PRL |
| JEPA minimal | 62.22% | 44.83% |  |
| WavJEPA (HF) | 33.33% | 24.14% |  |
| WavJEPA (local pretrain ckpt) | 33.33% | 24.14% |  |

Generated from `exp/*/RESULTS.md` under `models/espnet/egs2/ml_superb/asr1/exp`.

## 2) Multilingual HuBERT ASR-only (partial MLSUPERB tree)

Three languages in prep when only mls/swc/voxforge are present; not the full 143-lang paper setup. See `refs/MULTILINGUAL_RECAP.md`.

| Setting | Test CER | Test WER | Exp tag (under `exp/`) |
|---------|----------|------------|-------------------------|
| Multilingual ASR, 10 min | 24.96% | 23.48% | `asr_train_asr_s3prl_10min_multilingual_10min` |
| Multilingual ASR, 1 h | 20.76% | 18.30% | `asr_train_asr_s3prl_1h_multilingual_1h` |

## 3) Multilingual ASR+LID (partial MLSUPERB tree)

| Setting | Test CER | Test WER | Exp tag (under `exp/`) | Status |
|---------|----------|----------|-------------------------|--------|
| Multilingual ASR+LID, 10 min | 26.33% | 25.49% | `asr_train_asr_s3prl_10min_multilingual_10min_lid` | DONE |
| Multilingual ASR+LID, 1 h | pending | pending | `asr_train_asr_s3prl_1h_multilingual_1h_lid` | RUNNING |

LoRA multilingual rows will be added when `RESULTS.md` exists for:
`asr_train_asr_s3prl_lora_10min_multilingual_10min` and
`asr_train_asr_s3prl_lora_1h_multilingual_1h`.

**Runtimes (GPU wall time):** see `refs/GPU_RUNTIME_TABLE.md`.
