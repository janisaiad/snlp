# Reproduction report (ASR + JEPA/WavJEPA + multilingual)

## Monolingual benchmark (completed)

- ML-SUPERB ASR for **eng1**, **10 min** (and 1 h where run): HuBERT (s3prl), JEPA minimal, WavJEPA (HF), WavJEPA (local ckpt).
- Result collection: `uv run python scripts/collect_asr_results.py` → `refs/ASR_RESULTS_TABLE.md` (section 1).
- WavJEPA local checkpoint: `lightning_checkpoint_path` / `checkpoint_path` in ESPnet frontend; example config `conf/tuning/train_asr_wavjepa_local_10min.yaml`; runner `scripts/run_asr_wavjepa_local_benchmark.sh`.

### Numbers (test_10min_eng1)

- HuBERT: CER 33.33%, WER 24.14%
- JEPA minimal: CER 62.22%, WER 44.83%
- WavJEPA (HF): CER 33.33%, WER 24.14%
- WavJEPA (local ckpt): CER 33.33%, WER 24.14%

## Multilingual + LID + LoRA (partial MLSUPERB)

- Orchestration: `scripts/run_ml_superb_multilingual_peft_queue.sh`; resume: `scripts/run_ml_superb_multilingual_peft_resume.sh`.
- Recipe tweaks: `run_multi.sh` (per-track `asr_tag` / stats); `local/data_prep.py` (skip missing corpora).
- LoRA configs: `conf/tuning/train_asr_s3prl_lora_{10min,1h}.yaml`.
- **Completed** with standard `RESULTS.md` for multilingual **ASR-only** (10 min, 1 h) and **LID-only** (10 min, 1 h). Metrics for ASR-only: see `refs/ASR_RESULTS_TABLE.md` section 2 and `refs/MULTILINGUAL_RECAP.md`.
- **ASR+LID**: 10 min run completed (`exp/asr_train_asr_s3prl_10min_multilingual_10min_lid/RESULTS.md`), 1 h currently running in resume queue.
- **LoRA**: pending; will run after ASR+LID 1 h in the same queue, then `collect_asr_results.py`.

## ABX

- Scripts: `scripts/run_abx_all_frontends.py`, `scripts/run_abx_vs_asr.py` (see each for `--data_name`).

## Extra languages (monolingual)

- `scripts/run_ml_superb_extra_langs.sh fra1 deu1` when corpora exist under `MLSUPERB`.

## Full write-up

- LaTeX: `refs/report.tex`
- Multilingual recap: `refs/MULTILINGUAL_RECAP.md`
- GPU wall times (L4, `ngpu 1`, `master.log` vs `train.log`): `refs/GPU_RUNTIME_TABLE.md`
