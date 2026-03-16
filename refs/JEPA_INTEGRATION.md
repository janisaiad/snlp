# JEPA integration (ML-SUPERB pipeline)

JEPA (Joint Embedding Predictive Architecture) is integrated as an optional **frontend** in the ML-SUPERB ASR recipe. You can train with a frozen JEPA encoder + CTC in the same way as HuBERT (S3PRL). All commands use **uv** and the snlp venv.

## One-liner (from snlp repo root)

```bash
# Full training (30 epochs, ~1–2 h on GPU)
./scripts/run_ml_superb_jepa.sh --skip-data --no-sync

# Quick check (1 epoch, 2 iters)
./scripts/run_ml_superb_jepa.sh --skip-data --no-sync --debug
```

- **Log:** `tail -f logs/jepa_full_run.log` if you ran with `nohup ... > logs/jepa_full_run.log &`
- **Training log:** `tail -f models/espnet/egs2/ml_superb/asr1/exp/asr_train_asr_jepa_10min_eng1_10min/train.log`
- **Results:** `models/espnet/egs2/ml_superb/asr1/exp/asr_train_asr_jepa_10min_eng1_10min/RESULTS.md`

## Config and manual run

- **Config:** `models/espnet/egs2/ml_superb/asr1/conf/tuning/train_asr_jepa_10min.yaml`
- **Frontend:** `jepa` (ESPnet class `JEPAFrontend` in `espnet2/asr/frontend/jepa.py`)

From the **asr1** directory (after data prep and env):

```bash
cd models/espnet/egs2/ml_superb/asr1
. ./path.sh && . ./cmd.sh && . ./db.sh
./run_one_lang.sh --single_lang eng1 --duration 10min \
  --asr_config conf/tuning/train_asr_jepa_10min.yaml \
  --stage 5 --stop_stage 13
```

From **snlp repo root** (legacy, use run_ml_superb_jepa.sh instead):

```bash
./scripts/run_ml_superb_ssl_experiments.sh --langs "eng1" --durations "10min" --no-sync
# then override config: edit the script to use train_asr_jepa_10min.yaml, or run run_one_lang.sh manually as above
```

## Two modes

1. **Minimal encoder (default)**  
   No extra install. The frontend uses a built-in mel-patch encoder (80-bin log-mel, patch size 16, 768-dim output). Good for testing the pipeline and for loading your own JEPA checkpoints via `frontend_conf.checkpoint_path`.

2. **Sony audio-representations ViT**  
   Optional: use the JEPA ViT from [SonyCSLParis/audio-representations](https://github.com/SonyCSLParis/audio-representations).

   - Clone the repo (e.g. into `third_party/audio-representations` or any path).
   - Install deps: `pip install timm` (and optionally the repo’s `requirements.txt`).
   - In the recipe config, set:
     ```yaml
     frontend_conf:
       jepa_repo_path: /path/to/audio-representations
       # checkpoint_path: /path/to/sony_jepa.ckpt  # optional
     ```
   - The frontend will import their `ViTEncoder`, run it on 80×208 mel chunks, and concatenate outputs. Note: Sony do not publish checkpoints; the repo suggests [MATPAC](https://github.com/aurianworld/matpac) for a powerful audio encoder.

## Pretrained JEPA: WavJEPA-Nat (Hugging Face)

Use the **pretrained** WavJEPA-Nat model from Hugging Face ([labhamlet/wavjepa-nat-base](https://huggingface.co/labhamlet/wavjepa-nat-base)) for better ASR than the minimal (untrained) encoder.

```bash
# Full training (30 epochs)
./scripts/run_ml_superb_wavjepa.sh --skip-data --no-sync

# Quick check
./scripts/run_ml_superb_wavjepa.sh --skip-data --no-sync --debug
```

- **Config:** `conf/tuning/train_asr_wavjepa_10min.yaml` — frontend `wavjepa`, frozen encoder, 768→80 preencoder.
- **Frontend:** `espnet2/asr/frontend/wavjepa.py` — loads `labhamlet/wavjepa-nat-base` with `trust_remote_code=True`; mono input is duplicated to stereo by the HF feature extractor.
- **Requires:** `transformers>=4.45,<5` (5.x has API changes that break this model), `editdistance`.

## Checkpoint

- **Minimal encoder:** You can train JEPA elsewhere (e.g. Sony repo or Audio-JEPA) and load the encoder weights via `checkpoint_path`. Keys should match the minimal encoder’s `patch_embed` and any other layers you add.
- **Sony ViT:** If you have a checkpoint from the Sony training (e.g. `M2DModule`), set `checkpoint_path` and `jepa_repo_path`; the frontend loads `encoder.*` state into their ViTEncoder.

## ABX vs ASR with JEPA

The current ABX script (`scripts/run_abx_vs_asr.py`) uses HuBERT features. To compare JEPA vs HuBERT:

1. Add a JEPA feature extractor (same interface as `extract_hubert_for_abx.py`) that loads the JEPA frontend or encoder and saves `.npy` per utterance.
2. Run ABX on those features and optionally print a second row (JEPA vs HuBERT) in the table.

This is left as a follow-up; the pipeline (train ASR with JEPA frontend, decode, score) is in place.
