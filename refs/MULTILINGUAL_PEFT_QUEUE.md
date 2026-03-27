# Multilingual ASR / LID / LoRA queue

**Recap (results, data caveats, status):** [`MULTILINGUAL_RECAP.md`](MULTILINGUAL_RECAP.md) · **Briefs ↔ code (exhaustive):** [`PROJECT_EXHAUSTIVE_RECAP.md`](PROJECT_EXHAUSTIVE_RECAP.md) · **PDF report:** [`report.tex`](report.tex) · **GPU / wall times:** [`GPU_RUNTIME_TABLE.md`](GPU_RUNTIME_TABLE.md)

## Dependency (LoRA / PEFT)

**LoRA** steps require **`loralib`** (ESPnet adapter). It is listed in the repo root **`pyproject.toml`**; after `uv sync`, `import loralib` must succeed before `train_asr_s3prl_lora_*.yaml` runs.

## What runs (sequential, one GPU)

Script: `scripts/run_ml_superb_multilingual_peft_queue.sh`

1. **Multilingual ASR** — `run_multi.sh`, `lid=false`, `only_lid=false` — `10min` + `1h`, `train_asr_s3prl_{10min,1h}.yaml`
2. **LID only** — `only_lid=true` — same durations / configs
3. **ASR + LID** — `lid=true`, `only_lid=false` — same durations / configs
4. **Multilingual ASR + LoRA** — same `run_multi.sh` flags, configs `train_asr_s3prl_lora_{10min,1h}.yaml` (HuBERT **large** + LoRA on `q_proj` / `k_proj`)

Logs: `logs/ml_superb_multilingual_peft/` (`master.log`, `multi_*.log`).  
End of queue runs `collect_asr_results.py` → updates **`refs/ASR_RESULTS_TABLE_eng1.md`** only; full multilingual tables are **`refs/ASR_RESULTS_TABLE.md`**.

## Recipe fixes applied

- **`run_multi.sh`**: distinct `asr_tag` and `asr_stats_dir` per track (`""`, `_only_lid`, `_lid`) so experiments do not overwrite each other.
- **`local/data_prep.py`**: skips missing top-level corpus folders so a **partial** `MLSUPERB` tree still works (warnings in log).

## Partial data caveat (your machine)

Under `data/ml_superb` you currently have **`mls`**, **`swc`**, **`voxforge`** extracted; other corpora from `eighth_version.zip` are skipped. Multilingual prep logged **3 languages** (eng, deu, fra). This is **not** the full 143-language ML-SUPERB paper setup — for that, extract the full archive (or all `DATA` folders) into `MLSUPERB`.

## Launch

```bash
cd /root/snlp
nohup ./scripts/run_ml_superb_multilingual_peft_queue.sh >> logs/ml_superb_multilingual_peft.nohup.log 2>&1 &
```

Optional: `ML_SUPERB_NGPU=1 ML_SUPERB_NJ=8 ML_SUPERB_INF_NJ=4` (defaults shown).

## After a reboot (resume without redoing finished jobs)

Script: `scripts/run_ml_superb_multilingual_peft_resume.sh` — same order as the main queue, but **skips** any step whose `exp/asr_${asr_tag}/RESULTS.md` already exists (so completed multilingual ASR runs are not repeated; in-progress LID/LoRA runs continue from existing checkpoints when you rerun from stage 1).

```bash
cd /root/snlp
nohup ./scripts/run_ml_superb_multilingual_peft_resume.sh >> logs/ml_superb_multilingual_peft/resume.nohup.log 2>&1 &
```

## Monitor

```bash
tail -f logs/ml_superb_multilingual_peft/master.log
tail -f logs/ml_superb_multilingual_peft/multi_asr_only_10min.log
```
