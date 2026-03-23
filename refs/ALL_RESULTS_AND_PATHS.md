# All results, paths, and what is left (single place)

**Repo root** in paths below: the directory that contains `models/`, `refs/`, `scripts/` (e.g. `/path/to/snlp` or `~/snlp`).

**ESPnet recipe (ML-SUPERB):** `models/espnet/egs2/ml_superb/asr1/`  
**Experiments (checkpoints, RESULTS):** `models/espnet/egs2/ml_superb/asr1/exp/`  
**Data:** `MLSUPERB` env or default `data/ml_superb/`

---

## 0) Which document is which (stop the confusion)

| File | What it is | Use it for |
|------|------------|------------|
| **`ALL_RESULTS_AND_PATHS.md`** (this file) | **Master list**: numbers + paths + remaining work | Day-to-day truth; share with collaborators |
| **`ASR_RESULTS_TABLE.md`** | Compact **tables only** (CER/WER) | Quick copy-paste into slides |
| **`report.tex`** | **PDF/LaTeX** narrative for a formal report | Compile with `pdflatex`; not the live status dashboard |
| **`report.md`** | **Tiny** smoke test (fbank tiny config, RAM usage) | **Not** the main results; historical minimal run |
| **`PRESENTATION_SUMMARY.md`** | Short talk / slide bullets | Overview, not full paths |
| **`MULTILINGUAL_RECAP.md`** | Multilingual + LID + queue story | Context and data caveats |
| **`GPU_RUNTIME_TABLE.md`** | Wall times, GPU notes | Runtime only |
| **`PROJECT_EXHAUSTIVE_RECAP.md`** | Links `idea.md` / `rendu1.md` to code | Thesis / group alignment |
| **`GIT_PUSH_AND_HANDOFF.md`** | How to `git push` / fix 403 | Git only |

If something disagrees, prefer **this file + `exp/.../RESULTS.md`** over older notes in `rendu1.md`.

---

## 1) Monolingual ASR (`eng1`, test `test_10min_eng1`)

Frozen SSL + CTC head; **same protocol** as ML-SUPERB monolingual track.

| Frontend | Test CER | Test WER | Exp directory under `exp/` |
|----------|----------|------------|------------------------------|
| HuBERT (frozen, S3PRL) | 33.33% | 24.14% | `asr_train_asr_s3prl_10min_eng1_10min` |
| JEPA minimal | 62.22% | 44.83% | `asr_train_asr_jepa_10min_eng1_10min` |
| WavJEPA (HF) | 33.33% | 24.14% | `asr_train_asr_wavjepa_10min_eng1_10min` |
| WavJEPA (local pretrain ckpt) | 33.33% | 24.14% | `asr_train_asr_wavjepa_local_ckpt_10min_eng1_10min` |

**Full metrics file per run:**  
`models/espnet/egs2/ml_superb/asr1/exp/<expdir>/RESULTS.md`

---

## 2) Multilingual HuBERT (partial MLSUPERB tree)

When only **mls / swc / voxforge** (or similar) are present, prep uses **~3 languages** (e.g. eng, deu, fra) — **not** the full 143-language paper setup.

### 2a) Multilingual ASR only (HuBERT frozen)

| Duration | Test CER | Test WER | Exp directory |
|----------|----------|------------|---------------|
| 10 min | 24.96% | 23.48% | `asr_train_asr_s3prl_10min_multilingual_10min` |
| 1 h | 20.76% | 18.30% | `asr_train_asr_s3prl_1h_multilingual_1h` |

### 2b) LID-only (completed; ASR table not meaningful)

Runs finished with **`RESULTS.md`**, but ESPnet’s ASR **WER/CER rows show 0 words** (LID task — do **not** interpret as ASR quality).

| Duration | Status | Exp directory |
|----------|--------|----------------|
| 10 min | DONE | `asr_train_asr_s3prl_10min_multilingual_10min_only_lid` |
| 1 h | DONE | `asr_train_asr_s3prl_1h_multilingual_1h_only_lid` |

Use **LID accuracy / decode logs** for LID metrics if needed.

### 2c) ASR + LID (joint) — **not finished**

| Duration | `RESULTS.md` | Exp directory | Notes |
|----------|--------------|---------------|--------|
| 10 min | **pending** | `asr_train_asr_s3prl_10min_multilingual_10min_lid` | Training was **stopped by user**; **checkpoints kept** under this folder |
| 1 h | **pending** | `asr_train_asr_s3prl_1h_multilingual_1h_lid` | Not started |

### 2d) Multilingual LoRA (HuBERT large + LoRA) — **not finished**

| Duration | `RESULTS.md` | Exp directory |
|----------|--------------|---------------|
| 10 min | **pending** | `asr_train_asr_s3prl_lora_10min_multilingual_10min` |
| 1 h | **pending** | `asr_train_asr_s3prl_lora_1h_multilingual_1h` |

**Configs:**  
`models/espnet/egs2/ml_superb/asr1/conf/tuning/train_asr_s3prl_lora_10min.yaml`  
`models/espnet/egs2/ml_superb/asr1/conf/tuning/train_asr_s3prl_lora_1h.yaml`

---

## 3) Monolingual matrix (fra1 / deu1) — ran separately

Numbers are in each run’s **`RESULTS.md`** (not duplicated here line-by-line). **Training wall times** (from `train.log`) are in `refs/GPU_RUNTIME_TABLE.md`.

| Lang | Durations | Example exp tag pattern |
|------|-----------|-------------------------|
| fra1 | 10 min, 1 h | `asr_train_asr_s3prl_*_fra1_*` |
| deu1 | 10 min, 1 h | `asr_train_asr_s3prl_*_deu1_*` |

---

## 4) Queue / resume status (multilingual + PEFT)

**Scripts (from repo root):**

- Full queue: `scripts/run_ml_superb_multilingual_peft_queue.sh`
- **Resume** (skips steps that already have `RESULTS.md`): `scripts/run_ml_superb_multilingual_peft_resume.sh`

**Logs:**

- Master timeline: `logs/ml_superb_multilingual_peft/master.log`
- Per-job logs: `logs/ml_superb_multilingual_peft/multi_*.log`
- Last known line: **ASR+LID 10 min** was **START**’d; then **`STOPPED_BY_USER`** (checkpoints preserved).

**Remaining order (after you rerun resume):**

1. Finish **ASR+LID 10 min** (resume from `exp/.../asr_train_asr_s3prl_10min_multilingual_10min_lid/` checkpoints).
2. **ASR+LID 1 h**
3. **LoRA 10 min**
4. **LoRA 1 h**
5. `uv run python scripts/collect_asr_results.py` (end of script; output in `logs/ml_superb_multilingual_peft/collect.log` if run)

---

## 5) Path cheat sheet (copy-paste)

| What | Path |
|------|------|
| Recipe | `models/espnet/egs2/ml_superb/asr1/` |
| All experiment dirs | `models/espnet/egs2/ml_superb/asr1/exp/` |
| Checkpoints (gitignored) | `exp/<asr_train_*>/*.pth` etc. |
| Multilingual queue logs | `logs/ml_superb_multilingual_peft/` |
| Aggregate results script | `scripts/collect_asr_results.py` |
| WavJEPA pretrain (optional) | `scripts/run_wavjepa_pretrain.sh`; local clone ignored: `third_party/wavjepa/` |

**Environment:**

```bash
export PATH="/path/to/snlp/.venv/bin:$PATH"
export MLSUPERB="/path/to/snlp/data/ml_superb"   # or your MLSUPERB root
```

---

## 6) Runtimes (summary)

See **`refs/GPU_RUNTIME_TABLE.md`** for full tables. Short version for multilingual queue steps that **completed**:

| Step | ~Wall (full pipeline) |
|------|------------------------|
| Multilingual ASR 10 min | ~6 h 07 min |
| Multilingual ASR 1 h | ~11 h 37 min |
| LID-only 10 min | ~5 h 04 min |
| LID-only 1 h | ~11 h 30 min |

---

## 7) How to refresh this file after new runs

1. Check each **`exp/asr_train_*/RESULTS.md`** for new CER/WER.
2. Update **`refs/ASR_RESULTS_TABLE.md`** (tables).
3. Update **this file** (sections 1–2 and “remaining”).
4. Optionally run **`uv run python scripts/collect_asr_results.py`** and paste into `refs/` if your collector writes a summary.

---

*Last consolidated for clarity across `report.tex`, `report.md`, and scattered `refs/*.md`.*
