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
| Benchmark environment | 15% | Done | ESPnet + ml_superb recipe; pipeline runs (run_ml_superb_baseline.sh, SSL scripts) |
| Data setup (ML-SUPERB 10 min/1 h) | 15% | Done | eng1 data (mls/eng); scripts for download/prep; 1h transcripts (ensure_1h_transcripts.sh) |
| Reproduction (paper results) | 25% | In progress | eng1 10min SSL done (HuBERT frozen + CTC, CER in RESULTS.md); eng1 1h long training run |
| JEPA integration (Audio-JEPA in pipeline) | 15% | Done | JEPA frontend in espnet2; config train_asr_jepa_10min.yaml; optional Sony repo (refs/JEPA_INTEGRATION.md) |
| ABX + ASR comparison (JEPA vs HuBERT) | 10% | Done | Pipeline run (scripts/run_abx_vs_asr.py); HuBERT ABX + ASR CER table; JEPA after integration |

**Overall: ~85%.** Reproduction, ABX vs ASR, and JEPA integration done; optional: run JEPA training, ABX with JEPA features.

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

---

### What’s done now (update)

- **Benchmark environment:** Done. Pipeline runs (path.sh, run_one_lang.sh, `./scripts/run_ml_superb_baseline.sh`, `./scripts/run_ml_superb_ssl_full.sh`), scoring without sclite, uv-only.
- **Data setup:** Done for eng1 (download script, extract, `data/ml_superb/mls/eng/` with transcripts + wav, 1h transcripts). Full 8th release gives more languages.
- **Baseline runs:** **FBANK** baseline (10 min eng1) and **SSL (HuBERT frozen + CTC)** for eng1 10min — CER/WER in `exp/.../RESULTS.md`. Long training: full grid (`run_ml_superb_ssl_full.sh`) runs eng1 10min + eng1 1h.
- **ABX + ASR comparison:** Done. Script `scripts/run_abx_vs_asr.py` extracts HuBERT features, runs ABX (fastabx), reads ASR CER, prints table. Run: `uv run python scripts/run_abx_vs_asr.py --recipe_dir models/espnet/egs2/ml_superb/asr1`. See refs/report.md.

### ASR results (eng1, 10 min, test set)

Evaluation uses the same test split (1 utterance in current data; CER/WER are indicative).

| Frontend        | Config                         | CER (test_10min_eng1) | WER (test_10min_eng1) | Note                    |
|-----------------|---------------------------------|------------------------|------------------------|-------------------------|
| HuBERT (frozen) | train_asr_s3prl_10min.yaml     | **33.33%**            | **24.14%**             | 30 ep; baseline         |
| JEPA minimal    | train_asr_jepa_10min.yaml      | 62.22%                | 44.83%                 | 30 ep; no pretrain      |
| WavJEPA (HF)    | train_asr_wavjepa_10min.yaml   | (run pending)         | (run pending)           | 30 ep in progress       |
| WavJEPA 5ep     | train_asr_wavjepa_5ep.yaml     | (run pending)         | (run pending)           | 5 ep quick comparison   |

**Discussion:** HuBERT (pretrained) gives the best CER/WER. JEPA with a small untrained encoder is worse. WavJEPA-Nat (pretrained JEPA from Hugging Face) is expected to sit between the two once 30-ep training finishes; 5-ep and 30-ep runs can be compared to see convergence.

**How to run:** From repo root, with `MLSUPERB` set and data prepared:
- **Quick debug (1 ep, 2 iters):** `./scripts/run_ml_superb_train_eval_all.sh --debug --skip-data --no-sync` — runs JEPA + WavJEPA (skips HuBERT if S3PRL not installed).
- **Full 30 ep:** `./scripts/run_ml_superb_wavjepa.sh --skip-data --no-sync` or `run_ml_superb_jepa.sh`; HuBERT: use `train_asr_s3prl_10min.yaml` with S3PRL installed.
Results: `models/espnet/egs2/ml_superb/asr1/exp/<asr_tag>/RESULTS.md`.

### Encoder pretraining (can it be done so far?)

- **HuBERT:** Yes, in ESPnet (`espnet2/tasks/hubert.py`, `ssl.py`; e.g. egs2/mini_an4/ssl1). This project uses **pretrained** HuBERT via S3PRL; we do not run HuBERT pretraining in the ML-SUPERB recipe.
- **JEPA:** We use WavJEPA (HF pretrained) or the minimal encoder (no pretrain). To **pretrain** a JEPA encoder: clone the [WavJEPA](https://github.com/labhamlet/wavjepa) repo and run from here: `./scripts/setup_wavjepa.sh` then `./scripts/run_wavjepa_pretrain.sh --num-gpus 10 --data audioset` (e.g. 10× H100). See **refs/RESEARCH_PLAN.md** (section “Using the WavJEPA repo”) and `third_party/README.md`.

### Full research run (one command)

**One-liner (recommended on a big GPU instance):** from repo root run  
`./scripts/run_full_pipeline.sh`  
to download ML-SUPERB data, sync deps, run data prep, train all frontends (HuBERT if S3PRL, JEPA, WavJEPA), and eval. Use `--debug` for a quick 1 ep check, `--skip-download` if data is already at `data/ml_superb`, `--skip-data` if data is already prepared. Results in `exp/*/RESULTS.md` and `logs/research_results_*.txt`.

Without download: `./scripts/run_research_full.sh` (set `MLSUPERB` if needed). Plan: **refs/RESEARCH_PLAN.md**.

### What’s remaining

1. **Reproduction (SSL):** Let eng1 1h finish; optionally add fra1/deu1 data and run full 3 lang × 2 dur grid; record CER and compare to ML-SUPERB (2023) tables.
2. **Document:** Report and table with commands, data paths, CER per language/model (see refs/report.md, refs/SSL_TRAINING_LINKS.md).
3. **JEPA integration:** Done. Use `frontend: jepa` and `conf/tuning/train_asr_jepa_10min.yaml`; optional Sony repo (see refs/JEPA_INTEGRATION.md). Run training: `run_one_lang.sh --asr_config conf/tuning/train_asr_jepa_10min.yaml ...`. **WavJEPA (pretrained):** `conf/tuning/train_asr_wavjepa_10min.yaml`, `./scripts/run_ml_superb_wavjepa.sh`.
4. **ABX + ASR with JEPA:** Add JEPA feature extraction to scripts/run_abx_vs_asr.py (or a separate script) and compare JEPA vs HuBERT in the same table.