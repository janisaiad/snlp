# WavJEPA HF vs Local Checkpoint Analysis

## Setup
- Split: `dev_10min`
- Utterances analyzed: **311**
- Local checkpoint: `/root/snlp/logs/wavjepa_pretrain/saved_models_jepa_new_masking/Data=AudioSet/Extractor=wavjepa/InSeconds=2.01/BatchSize=32/NrSamples=8/NrGPUs=1/LR=0.0004/TargetProb=0.25/TargetLen=10/ContextProb=0.65/ContextLen=10/MinContextBlock=1/ContextRatio=0.1/last.ckpt`
- Downstream HF exp: `asr_train_asr_wavjepa_10min_eng1_10min`
- Downstream Local exp: `asr_train_asr_wavjepa_local_ckpt_10min_eng1_10min`

## Representation-space drift (utterance mean embeddings)
- Cosine(HF, Local) mean: **-0.0082**
- Cosine std: 0.0317 | min: -0.0974 | max: 0.0766
- Norm ratio Local/HF mean: **1.5940**

Interpretation: cosine near 0 (slightly negative) indicates a strong representational rotation/drift between HF and local ckpt, despite same output dimensionality.

## Word-level centroid drift (first token proxy)
| word | n | cosine(HF,Local) |
|---|---:|---:|
| je | 3 | -0.0894 |
| le | 8 | -0.0647 |
| il | 16 | -0.0622 |
| ce | 3 | -0.0613 |
| vous | 4 | -0.0519 |
| les | 10 | -0.0506 |
| elias | 3 | -0.0477 |
| la | 9 | -0.0463 |
| sie | 3 | -0.0443 |
| einem | 3 | -0.0386 |
| on | 3 | -0.0217 |
| un | 3 | -0.0197 |
| um | 3 | -0.0070 |
| dass | 4 | -0.0044 |
| das | 4 | -0.0008 |
| den | 5 | -0.0006 |
| im | 3 | 0.0046 |
| die | 10 | 0.0101 |
| der | 10 | 0.0150 |
| für | 3 | 0.0198 |

## Downstream CTC head comparison (HF run vs Local run)
- CTC row cosine mean: **0.0321**
- CTC row cosine std: 0.0554

Interpretation: even with similar global CER/WER, the learned class hyperplanes can differ while preserving overall error rate due to dataset size and decoding constraints.

## Raw summary (JSON)
```json
{
  "n_utts": 311,
  "utt_cos_mean": -0.008219027891755104,
  "utt_cos_std": 0.031691718846559525,
  "utt_cos_min": -0.09738926589488983,
  "utt_cos_max": 0.0765913724899292,
  "norm_ratio_mean_local_over_hf": 1.5940232276916504,
  "ctc_row_cos_mean": 0.03207182511687279,
  "ctc_row_cos_std": 0.055441681295633316
}
```