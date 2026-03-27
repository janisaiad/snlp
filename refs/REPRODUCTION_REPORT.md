# Reproduction report (ASR + JEPA/WavJEPA + multilingual)

## Monolingual benchmark (completed)

- ML-SUPERB ASR for **eng1**, **10 min** (and 1 h where run): HuBERT (s3prl), JEPA minimal, WavJEPA (HF), WavJEPA (local ckpt).
- Auto-generated monolingual table: `uv run python scripts/collect_asr_results.py` → **`refs/ASR_RESULTS_TABLE_eng1.md`**. The **full** table (multilingual, ASR+LID, LoRA) is **`refs/ASR_RESULTS_TABLE.md`**.
- WavJEPA local checkpoint: `lightning_checkpoint_path` / `checkpoint_path` in ESPnet frontend; example config `conf/tuning/train_asr_wavjepa_local_10min.yaml`; runner `scripts/run_asr_wavjepa_local_benchmark.sh`.

### Numbers (test_10min_eng1)

- HuBERT: CER 33.33%, WER 24.14%
- JEPA minimal: CER 62.22%, WER 44.83%
- WavJEPA (HF): CER 33.33%, WER 24.14%
- WavJEPA (local ckpt): CER 33.33%, WER 24.14%

## Multilingual + LID + LoRA (partial MLSUPERB)

- Orchestration: `scripts/run_ml_superb_multilingual_peft_queue.sh`; resume: `scripts/run_ml_superb_multilingual_peft_resume.sh`.
- Recipe tweaks: `run_multi.sh` (per-track `asr_tag` / stats); `local/data_prep.py` (skip missing corpora).
- LoRA configs: `conf/tuning/train_asr_s3prl_lora_{10min,1h}.yaml`; dependency **`loralib`** in `pyproject.toml`.
- **Completed** with `RESULTS.md`: multilingual **ASR-only** (10 min, 1 h), **LID-only** (10 min, 1 h), **ASR+LID 10 min**, **LoRA multilingual 10 min**.
- **ASR+LID 1 h** — no final `RESULTS.md` (run stopped).
- **LoRA 1 h** — pending until `exp/asr_train_asr_s3prl_lora_1h_multilingual_1h/RESULTS.md` exists.

**Metrics:** see **`refs/ASR_RESULTS_TABLE.md`** and **`refs/MULTILINGUAL_RECAP.md`**.

## ABX

- Scripts: `scripts/run_abx_all_frontends.py`, `scripts/run_abx_vs_asr.py` (see each for `--data_name`).

## Extra languages (monolingual)

- `scripts/run_ml_superb_extra_langs.sh fra1 deu1` when corpora exist under `MLSUPERB`.

## Full write-up

- LaTeX: `refs/report.tex`
- Multilingual recap: `refs/MULTILINGUAL_RECAP.md`
- GPU wall times (L4, `ngpu 1`, `master.log` vs `train.log`): `refs/GPU_RUNTIME_TABLE.md`
