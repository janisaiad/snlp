Project Context

This project is based on ML-SUPERB (Shi et al., 2023), which benchmarks frozen SSL represen-
tations for multilingual ASR and LID under low-resource conditions (10 min / 1 h). Our group

proposes a cohesive extension by linking linguistic interpretability (Vadim), architectural in-
novation (Janis), and paralinguistic/neural validation (Bruny) to explore the universal capa-
bilities of speech models.

Vadim: Linguistic Interpretability and Cross-Lingual Benchmarking
I will implement the core ML-SUPERB pipeline to identify why certain languages underperform
through a layer-wise analysis, interpreting how learned weights prioritize acoustic, phonetic, or
semantic information across language families. By classifying "hard" languages, I will determine
if their difficulty stems from phonetic ambiguity or linguistic distance from the pre-training data.
This baseline study will be further enriched by incorporating Janis’s JEPA model to observe if this
new architecture shifts the hierarchy of difficult languages or alters the layer-wise contribution to
the final representation.
Janis: Joint Embedding Predictive Architectures (JEPA)

I propose to extend the benchmark to recent architectures, specifically Joint Embedding Pre-
dictive Architectures (JEPA), which remain underexplored in speech (e.g., Audio-JEPA [1]).

I will evaluate the trade-off between phonetic/spectral representations (via ABX discrimina-
tion) and semantic/lexical representations (ASR with CTC objectives). Specifically, I will

test whether JEPA and HuBERT/wav2vec are complementary and if fine-tuning Audio-JEPA with
a CTC objective produces representations as linguistically expressive as wav2vec 2.0. These results
will be shared with Bruny to quantify their robustness on paralinguistic tasks and their biological
plausibility.
Bruny: Paralinguistic Robustness and Neural Alignment

I will extend the evaluation framework with two additional metrics: Speech Emotion Recogni-
tion (SER) for low-resource emotion classification, and Brain Score to quantify the alignment

between SSL representations and time-resolved neural responses to speech [2]. I will inves-
tigate whether the layers that best align with cortical speech processing are also those that

most strongly support emotion recognition. This study will apply the cross-lingual logic developed
by Vadim to see if ASR difficulty correlates with paralinguistic difficulty, while evaluating if the
architectural shifts introduced by Janis improve the alignment with human auditory processing.

1

References
[1] Ludovic Tuncay, Etienne Labbé, Emmanouil Benetos, and Thomas Pellegrini. Audio-jepa:
Self-supervised learning for audio via joint embedding predictive architectures. arXiv preprint
arXiv:2507.02915, 2025.

[2] Alex Raugel, Jean-Rémi King, et al. Hierarchical and temporal alignment between neural ac-
tivity and language models during speech processing. arXiv preprint arXiv:2512.01591,

---

## Janis work: progress and next steps

### Progress evaluation (approx.)

| Component | Weight | Status | Notes |
|-----------|--------|--------|--------|
| Literature and design | 20% | Done | refs/idea.md: ML-SUPERB procedure, JEPA rationale, ABX vs ASR, fusion idea |
| Benchmark environment | 15% | Partial | ESPnet + ml_superb recipe under models/espnet/egs2/ml_superb; no run yet |
| Data setup (ML-SUPERB 10 min/1 h) | 15% | Not done | MLSUPERB not set; data not downloaded or prepared |
| Reproduction (paper results) | 25% | Not done | No runs; no HuBERT/wav2vec CER for any language |
| JEPA integration (Audio-JEPA in pipeline) | 15% | Not done | No S3PRL upstream or config for JEPA |
| ABX + ASR comparison (JEPA vs HuBERT) | 10% | Not done | Depends on reproduction and JEPA |

**Overall: ~25–30%.** Strong on design and spec; execution (data, runs, JEPA) not started. Reproduction is the critical path for the rest.

### What Janis should do right now: reproduce the benchmark first

**Goal:** Reprendre le benchmark et répliquer quelques résultats pour des modèles et langues donnés (take up the benchmark and replicate some results for given models and languages).

1. **Set up ML-SUPERB data**
   - Download ML-SUPERB (e.g. [Huggingface ftshijt/mlsuperb_8th](https://huggingface.co/datasets/ftshijt/mlsuperb_8th) or 7th) and unzip.
   - In `models/espnet/egs2/ml_superb/asr1/`, create or edit `db.sh` and set `MLSUPERB` to the unzipped data path.
   - Run data preparation (stages 1–10) so train/dev/test splits for 10 min and 1 h exist for at least a few languages.

2. **Run one baseline to validate the pipeline**
   - From `models/espnet/egs2/ml_superb/asr1/`, run a **single** setup: e.g. HuBERT (e.g. `train_asr_s3prl_10min.yaml` or a mono config), **one language** (e.g. `eng1` or `fra1`), **10 min** only.
   - Use `run_mono.sh` with the right `--asr_config` and ensure `path.sh` / `cmd.sh` point to the ESPnet env. Check that training and decoding finish and that CER is written to `exp/` (or the log file).

3. **Replicate a small set of paper results**
   - Pick **2–3 languages** (e.g. eng1, fra1, deu1) and **one SSL** (HuBERT-base or the recipe’s default).
   - Run **10 min** (and optionally 1 h) for each. Record CER per language and compare to ML-SUPERB (2023) tables (e.g. HuBERT-base rows) to confirm reproduction is in the right ballpark.

4. **Document and hand off**
   - Note exact commands, data paths, and CER in a short report or table. Once this is stable, Vadim can plug in layer-wise analysis and Bruny can use the same pipeline; JEPA integration (S3PRL upstream or custom frontend) and ABX come next after reproduction.