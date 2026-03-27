# ML-SUPERB Reproduction and JEPA Extension: Linear Summary

## 1) Project Goal

The project follows ML-SUPERB (Shi et al., 2023): use frozen SSL speech representations and train an ASR head with CTC in low-resource settings (10 min and 1 h).  
The extension adds JEPA-based frontends and compares them to HuBERT.

## 2) Scope Implemented

- Baseline recipe: ESPnet `egs2/ml_superb/asr1`.
- Frontends evaluated (monolingual `eng1`):
  - HuBERT (frozen, S3PRL)
  - JEPA minimal
  - WavJEPA (Hugging Face pretrained)
  - WavJEPA (local pretrained checkpoint)
- Multilingual tracks (HuBERT / S3PRL): ASR-only, LID-only, ASR+LID, LoRA — orchestrated by `scripts/run_ml_superb_multilingual_peft_queue.sh` with resume script `scripts/run_ml_superb_multilingual_peft_resume.sh`.
- Primary monolingual benchmark slice verified: `eng1`, `10min`, `test_10min_eng1`.

## 3) Infrastructure and Pipeline Status

- Data, recipe, and scoring pipeline run end-to-end.
- Automated **monolingual eng1** table: `scripts/collect_asr_results.py` → `refs/ASR_RESULTS_TABLE_eng1.md`. Full multilingual tables: `refs/ASR_RESULTS_TABLE.md`.
- Multilingual recipe fixes: distinct `asr_tag` / stats dirs per track in `run_multi.sh`; partial `MLSUPERB` tree handled in `local/data_prep.py`.
- Long-run workflow: background queue, checkpoints under each `exp/asr_train_*/`, resume after reboot.

## 4) Key Engineering Work Completed

1. Integrated JEPA/WavJEPA into the ASR workflow.
2. Local WavJEPA checkpoint loading path in ESPnet frontend.
3. Scripts for local-checkpoint ASR benchmark and result collection.
4. Long-run reliability (resume, checkpoints, logging).
5. ABX extraction scripts for HuBERT, WavJEPA, and JEPA minimal.
6. Multilingual + LID + LoRA queue, LoRA YAMLs, and resume-by-`RESULTS.md` script.
7. Consolidated documentation: `refs/report.tex`, `refs/MULTILINGUAL_RECAP.md`, this file.

## 5) Verified ASR Results

### Monolingual (eng1, 10 min, test)

From `refs/ASR_RESULTS_TABLE.md` (section 1):

- HuBERT (frozen): CER 33.33, WER 24.14  
- JEPA minimal: CER 62.22, WER 44.83  
- WavJEPA (HF): CER 33.33, WER 24.14  
- WavJEPA (local ckpt): CER 33.33, WER 24.14  

### Multilingual HuBERT ASR-only (partial data; see recap)

From `refs/ASR_RESULTS_TABLE.md` (section 2):

- 10 min: CER 24.96, WER 23.48  
- 1 h: CER 20.76, WER 18.30  

### Multilingual ASR+LID and LoRA (partial data)

From `refs/ASR_RESULTS_TABLE.md` (sections 3–4):

- ASR+LID 10 min (`test_10min_lid`): CER 26.33, WER 25.49  
- ASR+LID 1 h: not completed (no `RESULTS.md`)  
- LoRA 10 min (`test_10min`): CER 24.95, WER 23.66 — vs frozen multilingual ASR 10 min, CER ~tie; WER slightly higher with LoRA  
- LoRA 1 h: pending  

## 6) Interpretation

- Pretraining matters: JEPA minimal is clearly weaker than pretrained frontends on `eng1`.
- On the monolingual slice, HuBERT and WavJEPA (HF/local) are tied.
- Multilingual ASR numbers are **not** comparable to monolingual `eng1` without qualification: different test sets and **partial** MLSUPERB (typically three langs when only mls/swc/voxforge are present).

## 7) Queue Status (multilingual + PEFT)

**Done:** multilingual ASR-only (10 min, 1 h), LID-only (10 min, 1 h), ASR+LID 10 min, LoRA 10 min (all have `RESULTS.md` where applicable).  
**Stopped:** ASR+LID 1 h (no final table).  
**Pending:** LoRA 1 h when training/decoding finishes.  
**Collector:** `collect_asr_results.py` refreshes **`refs/ASR_RESULTS_TABLE_eng1.md`** only.

## 8) Remaining Work

1. Finish **LoRA 1 h** and add its row to `refs/ASR_RESULTS_TABLE.md` when `RESULTS.md` exists.
2. Optionally rerun **ASR+LID 1 h** if a joint 1 h score is required.
3. Optionally expand `MLSUPERB` to full corpora for paper-comparable multilingual ranges.
4. Finalize ABX numeric scoring if required (toolchain constraints may apply).
5. Optional long-horizon WavJEPA pretraining and downstream re-check.

## 9) Reproducibility Artifacts

- **All numbers + paths + remaining work (read this first):** `refs/ALL_RESULTS_AND_PATHS.md`
- **Exhaustive recap** (maps `idea.md`, `rendu1.md`, roles Vadim/Janis/Bruny → code): `refs/PROJECT_EXHAUSTIVE_RECAP.md`
- Main tables: `refs/ASR_RESULTS_TABLE.md`
- Multilingual recap: `refs/MULTILINGUAL_RECAP.md`
- **GPU / wall-clock timings:** `refs/GPU_RUNTIME_TABLE.md` (L4, `ngpu 1`; queue vs `train.log` elapsed)
- PDF-ready report: `refs/report.tex` (includes alignment comments with briefs)
- Progress/history: `refs/PROJECT_STATUS.md`, `refs/REPRODUCTION_REPORT.md`
- Queue how-to: `refs/MULTILINGUAL_PEFT_QUEUE.md`
- **Push to GitHub / auth 403 fix:** `refs/GIT_PUSH_AND_HANDOFF.md`
- Scripts: `scripts/collect_asr_results.py`, `scripts/run_ml_superb_multilingual_peft_queue.sh`, `scripts/run_ml_superb_multilingual_peft_resume.sh`, `scripts/run_wavjepa_pretrain.sh`

## 10) One-Slide Conclusion

Monolingual `eng1` benchmarks are fixed across HuBERT / JEPA / WavJEPA. Multilingual tracks include frozen ASR, ASR+LID (10 min scored), and LoRA 10 min (near-tie CER vs frozen; WER slightly worse). ASR+LID 1 h was stopped; LoRA 1 h pending. Full write-up: `refs/report.tex`.
