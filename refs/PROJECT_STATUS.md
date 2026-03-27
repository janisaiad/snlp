# Project status: recap (updated)

**Traceability:** [`PROJECT_EXHAUSTIVE_RECAP.md`](PROJECT_EXHAUSTIVE_RECAP.md) (briefs ↔ code, Vadim/Janis/Bruny).

## Summary

| Area | Status | Notes |
|------|--------|--------|
| Monolingual ASR (eng1, 10 min/1 h protocol) | **Done** | HuBERT, JEPA minimal, WavJEPA HF/local; table in `refs/ASR_RESULTS_TABLE.md` |
| JEPA/WavJEPA integration in ESPnet | **Done** | Frontends + local WavJEPA ckpt path |
| Multilingual HuBERT **ASR-only** (partial MLSUPERB) | **Done** | CER/WER in `ASR_RESULTS_TABLE.md` §2; three langs typical with partial data |
| Multilingual **LID-only** | **Done** | `RESULTS.md` exists; WER/CER rows not meaningful for LID-only (see recap) |
| Multilingual **ASR+LID** + **LoRA** | **In queue or pending** | See `logs/ml_superb_multilingual_peft/master.log` |
| `collect_asr_results.py` after full queue | **When queue ends** | Appends `collect.log` |
| ABX (numeric) | **Partial** | Extraction scripts exist; scoring env may vary |
| Long WavJEPA pretrain | **Optional** | `scripts/run_wavjepa_pretrain.sh` |

**Overall:** core benchmark and multilingual pipeline **implemented**; remaining items are **final multilingual table rows** (ASR+LID, LoRA), optional full data, and optional ABX/pretrain depth.

---

## What was built (reproducibility)

1. **One-liner / launch:** `./launch.sh` (see `REPRODUCTION.md`) for env + benchmark path on a fresh machine.
2. **Multilingual queue:** `scripts/run_ml_superb_multilingual_peft_queue.sh` and **resume:** `scripts/run_ml_superb_multilingual_peft_resume.sh`.
3. **Documentation:** `refs/report.tex`, `refs/MULTILINGUAL_RECAP.md`, `refs/PRESENTATION_SUMMARY.md`.

---

## Multilingual caveat

Results labeled **multilingual** use whatever languages exist under `MLSUPERB` after `data_prep`. With only **mls / swc / voxforge**, expect **~3 languages**, not the full ML-SUPERB paper scale.

---

## On another cluster (benchmark only)

```bash
git clone <this-repo> snlp && cd snlp
chmod +x launch.sh && ./launch.sh
```

Multilingual: set `MLSUPERB`, then run the resume or full queue script from `refs/MULTILINGUAL_PEFT_QUEUE.md`.
