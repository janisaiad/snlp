# Exhaustive project recap: briefs, research notes, and repository mapping

This document **ties together** the written project briefs (`rendu1.md`, `idea.md`, `paper.md` excerpts), the **research plan** (`RESEARCH_PLAN.md`, `EXTENSIONS_ROADMAP.md`), and **what exists in the codebase**. Use it when writing the final report, thesis chapter, or handoff to collaborators (Vadim: layer-wise analysis; Bruny: paralinguistic / neural alignment).

**How to read the map:** each subsection cites a **source file** in `refs/` and states **what was implemented**, **what is partial**, and **where to look** (paths, scripts).

---

## 1. Group vision and roles (`rendu1.md` — “Project Context”)

| Stakeholder | Stated goal (summary) | Status in this repo | Notes / next steps |
|-------------|----------------------|---------------------|----------------------|
| **Vadim** | Core ML-SUPERB pipeline; **layer-wise** analysis; classify “hard” languages; relate difficulty to phonetics vs linguistic distance; enrich with **JEP**A to see if language hierarchy shifts | **Pipeline:** yes (ESPnet `egs2/ml_superb/asr1`, data prep, scoring). **Layer-wise language difficulty:** not automated here — needs analysis scripts on per-layer features or per-language CER tables. | Same `exp/` and `RESULTS.md` can feed external analysis; multilingual partial data (see `MULTILINGUAL_RECAP.md`) limits comparability to full ML-SUPERB. |
| **Janis** | **JEPA** in speech; **ABX** (phonetic) vs **ASR** (lexical); complementarity with HuBERT/wav2vec; share metrics with Bruny | **Implemented:** JEPA minimal frontend, WavJEPA (HF + local ckpt), ABX scripts (`run_abx_all_frontends.py`, `run_abx_vs_asr.py`), multilingual + **LoRA** queue. **Partial:** ABX numeric pipeline may need env fixes; fusion JEPA+HuBERT (`idea.md` § fusion) **not** implemented as a combined encoder. | See §3–4 below and `JEPA_INTEGRATION.md`. |
| **Bruny** | **SER**, **Brain Score**; layers aligning with cortex vs emotion; cross-lingual logic from Vadim; Janis architectures | **Not implemented** in this repository (no SER/Brain Score scripts in the tracked workflow). | Requires separate datasets and evaluation code; ASR `exp/` paths are a possible feature source for future alignment work. |

**Reference:** `refs/rendu1.md` (lines 1–41: group proposal; 54–137: Janis progress table and commands).

---

## 2. Janis deliverables vs `rendu1.md` checklist

The **“What Janis should do right now”** block in `rendu1.md` asks for: ML-SUPERB data → one baseline → 2–3 languages → document commands and CER.

| Checklist item | Evidence in repo |
|----------------|------------------|
| ML-SUPERB data + `db.sh` / `MLSUPERB` | `data/ml_superb` (partial tree), `models/espnet/egs2/ml_superb/asr1/db.sh` |
| Single baseline (HuBERT, one lang, 10 min) | `run_one_lang.sh`, `scripts/run_ml_superb_baseline.sh`; `exp/asr_train_asr_s3prl_10min_eng1_10min/RESULTS.md` (see caveats in `GPU_RUNTIME_TABLE.md` for failed short `train.log` vs final metrics) |
| 2–3 languages (e.g. eng1, fra1, deu1) | Monolingual exps under `exp/asr_train_*_{eng1,fra1,deu1}_*`; `scripts/run_ml_superb_extra_langs.sh` |
| Document commands + CER | `refs/ASR_RESULTS_TABLE.md`, `refs/REPRODUCTION_REPORT.md`, `refs/report.tex`, `refs/GPU_RUNTIME_TABLE.md` |
| JEPA/WavJEPA in pipeline | `conf/tuning/train_asr_jepa_10min.yaml`, `train_asr_wavjepa_*.yaml`, `train_asr_wavjepa_local_ckpt_10min.yaml` |
| **Multilingual ASR + LID + LoRA** | `run_multi.sh` suffix fix, `run_ml_superb_multilingual_peft_queue.sh`, `run_ml_superb_multilingual_peft_resume.sh`, LoRA YAMLs |

**Reference:** `refs/rendu1.md` (§ “What Janis should do right now”, § “What’s done now”).

---

## 3. Conceptual alignment (`idea.md` — transfer learning, universal latent space)

`idea.md` frames the study as: **frozen SSL + 10 min/1 h** tests whether **latent space** carries **language-independent** structure; **mono vs multilingual** SSL; **frozen vs LoRA** (PEFT).

| Concept in `idea.md` | Where it appears in the project |
|----------------------|----------------------------------|
| Frozen SSL + CTC, 10 min / 1 h | Default ML-SUPERB recipe; `train_asr_s3prl_{10min,1h}.yaml` |
| Monolingual vs multilingual ASR | `run_one_lang.sh` / `run_mono.sh` vs `run_multi.sh`; tables in `ASR_RESULTS_TABLE.md` |
| **Frozen vs LoRA** (2026 expectation) | **Implemented** for multilingual track: `train_asr_s3prl_lora_{10min,1h}.yaml` + queue script (runs after ASR+LID). Monolingual frozen-vs-LoRA grid is **optional** / not fully tabulated in one MD file. |
| JEPA vs masked-prediction SSL; **ABX vs ASR** | `idea.md` § research question; **ASR:** `RESULTS.md` for JEPA minimal vs HuBERT vs WavJEPA. **ABX:** scripts exist; **layer-wise** ranking ABX vs ASR still a **research to-do** (Poli / fastabx story). |
| **Fusion** JEPA + HuBERT features | **Not implemented** as a fused frontend in ESPnet (would need two-stream or concat + downstream). Listed as central “deep question” in `idea.md` § “Conclusion travaillée”. |
| Phonetic vs **lexical** (avoid “semantic” for ASR) | Clarified in `idea.md` § “Clarification”; **our reporting** uses CER/WER = lexical transcription; **LID** = language ID; **not** semantic SLU. |
| SUPERBs aggregate score | Defined in `idea.md`; **not** recomputed in-repo for all four tasks; partial multilingual metrics only. |

**Reference:** `refs/idea.md` (full file: goals, SpidR-Adapt, ML-SUPERB procedure, JEPA hypotheses, frozen vs LoRA, MMS, publishable directions).

---

## 4. Research questions → code / results (traceability)

| Question (from `idea.md` / `rendu1.md`) | Artifact or result |
|----------------------------------------|---------------------|
| Does **JEPA minimal** (no pretrain) match **HuBERT** on ASR? | **No** — JEPA minimal CER/WER much worse on `eng1` 10 min (`ASR_RESULTS_TABLE.md`). |
| Does **WavJEPA (pretrained)** match HuBERT on this slice? | **Yes** (tied CER/WER in reported table). |
| **Multilingual** HuBERT ASR vs monolingual `eng1` | **Different** test sets and data coverage — **do not** compare numerically without qualification (`MULTILINGUAL_RECAP.md`). |
| **LID-only** / **ASR+LID** | Queue runs `run_multi.sh` with `--only_lid` / `--lid`; LID-only `RESULTS.md` ASR WER/CER rows are **not** interpretable as word error; use LID accuracy from scoring logs. |
| **LoRA** multilingual HuBERT | Configs + queue; fill **GPU_RUNTIME_TABLE.md** and `ASR_RESULTS_TABLE.md` when `RESULTS.md` exists. |
| **Audio-JEPA** (Tuncay et al.) as citation | **WavJEPA** integration uses the public **WavJEPA** / **Audio-JEPA** line of work; see `rendu1.md` ref [1], `paper.md` / `README` for exact checkpoint names. |
| **SpidR-Adapt** / **MMS** as extra SSL | **Not** run as baselines in the current tables (`idea.md` suggests them as extensions). |

---

## 5. ML-SUPERB paper vs `paper.md` (abridged)

`refs/paper.md` is a **paste of the ML-SUPERB (2023) PDF** (arXiv:2305.10615): frozen SSL, 143 languages, monolingual + multilingual tracks, ASR + LID.

**Relevance:** confirms **protocol** (10 min / 1 h, CER, LID accuracy). Our **partial data** subset means we are **not** reproducing the full Table 1–style numbers from the paper.

---

## 6. Supporting documents (index)

| File | Role |
|------|------|
| `idea.md` | Deep research narrative: JEPA vs HuBERT, ABX, fusion, PEFT, MMS, SUPERBs, publishable directions. |
| `rendu1.md` | Group context + Janis progress + commands + results snapshot (some numbers may differ slightly from `ASR_RESULTS_TABLE.md` — **prefer** `ASR_RESULTS_TABLE.md` + `exp/.../RESULTS.md` as canonical). |
| `RESEARCH_PLAN.md` | Full pipeline: data → optional WavJEPA pretrain → ASR; GPU/time estimates for pretraining. |
| `EXTENSIONS_ROADMAP.md` | Maps **every** brief extension to ESPnet entry points (`run_multi.sh`, LID flags, phases). |
| `JEPA_INTEGRATION.md` | Technical JEPA / WavJEPA frontend wiring. |
| `MULTILINGUAL_RECAP.md` | Multilingual + LID + LoRA recap; data caveat. |
| `MULTILINGUAL_PEFT_QUEUE.md` | Launch/resume commands; link to GPU times. |
| `GPU_RUNTIME_TABLE.md` | Wall times: `master.log` vs `train.log` elapsed. |
| `ASR_RESULTS_TABLE.md` | Canonical CER/WER tables. |
| `PRESENTATION_SUMMARY.md` | Short slide-style summary. |
| `PROJECT_STATUS.md` | High-level % status. |
| `REPRODUCTION_REPORT.md` | Reproduction + pointers. |
| `report.tex` | LaTeX report (includes alignment with briefs). |
| `SSL_TRAINING_LINKS.md` | (if present) pointers to SSL training recipes. |

---

## 7. Conflicts and consistency rules

1. **Numbers:** If `rendu1.md` § ASR table disagrees with `ASR_RESULTS_TABLE.md` (e.g. WavJEPA 37.78% vs 33.33%), treat **`ASR_RESULTS_TABLE.md` + fresh `RESULTS.md`** as canonical unless you re-justify an older run.
2. **“Semantic” vs “lexical”:** Follow `idea.md` — ASR metrics are **lexical** (orthography); **phonetic** is ABX/PER.
3. **Full 143 languages:** Not achieved with partial data; **always** state which corpora are present under `MLSUPERB`.

---

## 8. Suggested next steps (aligned with `idea.md` + `rendu1.md`)

1. **Finish** multilingual queue (ASR+LID, LoRA) → update `ASR_RESULTS_TABLE.md`, `GPU_RUNTIME_TABLE.md`, `collect_asr_results.py` output.
2. **Vadim:** export per-layer features or per-language CER from `exp/` for layer-wise analysis (same protocol as `rendu1.md`).
3. **Janis:** optional **fusion** experiment (JEPA + HuBERT) as in `idea.md` — requires new frontend or two-pass feature concat.
4. **Bruny:** plug SER / Brain Score **outside** this repo or add a new `scripts/` eval harness.
5. **Poli line:** strengthen **ABX vs ASR** table with **layer-wise** fastabx if environment allows.

---

## 9. References cited in briefs (quick list)

- ML-SUPERB: Shi et al., 2023 (`paper.md`, arXiv:2305.10615).
- Audio-JEPA: Tuncay et al., arXiv:2507.02915 (`rendu1.md` [1], `idea.md`).
- Neural alignment / Brain Score: Raugel et al. (`rendu1.md` [2]) — not implemented.
- SpidR-Adapt / Poli (`idea.md`) — conceptual related work, not a baseline in current tables.

---

*This file is the exhaustive markdown hub for **reporting** and **cross-team** alignment. For PDF output, see `refs/report.tex` (section on alignment with `idea.md` and `rendu1.md`).*
