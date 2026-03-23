# ML-SUPERB — full optional extensions roadmap

This file maps **every extension** from the project brief to **concrete ESPnet entry points** in `models/espnet/egs2/ml_superb/asr1`, and groups work into **phases** so you can run or schedule GPU jobs without losing track.

**Reality check:** “everything” is **not** one weekend: the monolingual grid alone loops **14 languages × 2 durations** per config (`run_mono.sh`). Multilingual + LID variants add more full training passes. Treat this as a **research program**; use phases below.

**Strict benchmark note (paper / reproduction):** the README states you should typically only change **frontend** and **learning rate** in `conf/tuning` for official comparability. **Adapter (LoRA/Houlsby)** is documented as supported. Extra upstreams (wav2vec, custom HF HuBERT) are **allowed as comparisons** but document them as **extensions**, not the vanilla frozen-HuBERT baseline.

---

## 1) Monolingual vs multilingual ASR (comparison)

| Goal | Script | When to use |
|------|--------|-------------|
| One language at a time | `run_one_lang.sh` or full grid `run_mono.sh` | Low-resource **per-language** CER/WER |
| All languages pooled | `run_multi.sh` with `--lid false --only_lid false` | **Multilingual ASR** track |

**Comparison you care about:** same `asr_config` (e.g. `train_asr_s3prl_10min.yaml`) and duration, then contrast **monolingual** metrics vs **multilingual** metrics on overlapping test sets (see scoring outputs under `exp/`).

---

## 2) Cross-language comparison (“finetuning languages”)

| Goal | Mechanism |
|------|-----------|
| Compare which low-resource splits are easier/harder | Run `run_mono.sh` (or selective `run_one_lang.sh`) with the same SSL frontend; aggregate CER/WER per `single_lang` |
| Chinese / Japanese **phone-related** setup | In `local/data.sh`, `cmn` / `jpn` use G2P and **word** tokens with phone-based text pipeline (PER-style behavior for those tracks) |

---

## 3) Other tasks: LID and joint ASR+LID

From `asr1/README.md`:

| Task | Command pattern |
|------|-----------------|
| **LID only** | `./run_multi.sh --asr_config <cfg> --duration {10min,1h} --lid false --only_lid true` |
| **Multilingual ASR + LID** | `./run_multi.sh --asr_config <cfg> --duration {10min,1h} --lid true --only_lid false` |

Data prep toggles `suffix` (`_only_lid`, `_lid`) and scoring uses `local/score.sh` → `lid.py` when LID is on.

---

## 4) Other SSL backbones (wav2vec 2.0, multilingual HuBERT, HF ckpt)

S3PRL upstream is set in YAML:

```yaml
frontend_conf:
  frontend_conf:
    upstream: hubert_large_ll60k   # change to another S3PRL upstream
```

**HF custom example** (from README): `upstream: hf_hubert_custom` + `path_or_url: <huggingface_id>`.

**wav2vec 2.0:** duplicate `train_asr_s3prl_{10min,1h}.yaml`, set `upstream` to a wav2vec upstream supported by your S3PRL install, and **align `preencoder_conf.input_size`** with that model’s hidden size (same comment as for HuBERT in the template YAML).

---

## 5) PEFT: LoRA and Houlsby adapters (ML-SUPERB recipe)

Documented in `asr1/README.md`:

| Method | Config |
|--------|--------|
| LoRA | `conf/tuning/train_asr_s3prl_lora.yaml` |
| Houlsby | `conf/tuning/train_asr_s3prl_houlsby.yaml` |
| Pretrained adapter init (optional) | `https://huggingface.co/espnet/s3prl_adapter_model` |

Run with **`run_mono.sh`** or **`run_multi.sh`** by passing `--asr_config` to the adapter YAML (after data prep through the stage required by your setup; README says through stage 10 for adapter workflow).

---

## 6) Fbank baseline (non-SSL comparison)

Configs: `train_asr_fbank_{10min,1h}.yaml`, `train_asr_fbank_single.yaml`.  
Same scripts: `run_multi.sh` / `run_mono.sh` / `run_one_lang.sh` with `--asr_config` pointing to fbank.

---

## 7) What you already did (extensions beyond the one-line brief)

- **JEPA / WavJEPA frontends** in ASR (not in stock `run_multi.sh` text, but valid **comparison** work).
- **Local WavJEPA checkpoint** in ESPnet frontend.
- **Automated result collection** (`scripts/collect_asr_results.py`).

Keep documenting these explicitly as **methods extensions** when comparing to the official S3PRL-HuBERT baseline.

---

## 8) Suggested execution order (phases)

1. **Finish** current monolingual matrix you care about (e.g. eng/fra/deu, 10 min + 1 h, frozen HuBERT).
2. **Multilingual ASR** — `run_multi.sh` both durations, baseline `train_asr_s3prl_{10min,1h}.yaml`.
3. **LID-only** then **ASR+LID** — same configs, LID flags as above.
4. **PEFT** — LoRA then Houlsby on a **subset** of languages first (e.g. eng1, deu1, jpn), then scale.
5. **Extra SSL** — wav2vec / HF HuBERT on the same subset to validate `input_size` and training stability.
6. **Full monolingual sweep** — only if you need complete paper-style coverage (`run_mono.sh`).

---

## 9) Orchestration

Use the queue script (same repo):

```bash
# after reading the script header for PHASE values
./scripts/run_ml_superb_extensions_queue.sh
```

Logs go under `logs/ml_superb_extensions/`.

---

## 10) ML-SUPERB “2.0”

Your brief mentions **PEFT as in ML-SUPERB 2.0**. In **this** ESPnet tree, the documented PEFT path for the **2023 ML-SUPERB** recipe is **LoRA/Houlsby in `asr1`**. If you need exact **2.0** task definitions, check whether a separate recipe or paper supplement exists for the version you target; treat it as an additional milestone, not a duplicate of `asr1` flags.
