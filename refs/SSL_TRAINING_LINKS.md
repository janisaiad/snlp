# SSL training results (HuBERT frozen + CTC)

## How to see logs

- **Full grid run (all jobs):** from repo root, the script logs to `full_ssl_run.log`.  
  ```bash
  tail -f full_ssl_run.log
  ```
- **Current job’s training log:** one file per experiment, e.g.  
  ```bash
  tail -f models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_10min_eng1_10min/train.log
  ```
  Replace the experiment name with the one that is running (e.g. `train_asr_s3prl_1h_eng1_1h`).

All experiments live under the recipe `exp/` directory. Base path (from repo root):

```
models/espnet/egs2/ml_superb/asr1/exp/
```

## Full grid (3 languages × 2 durations)

| Lang | Duration | Experiment dir | Results (CER/WER) |
|------|----------|----------------|-------------------|
| eng1 | 10min | [train_asr_s3prl_10min_eng1_10min](models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_10min_eng1_10min) | [RESULTS.md](models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_10min_eng1_10min/RESULTS.md) |
| fra1 | 10min | [train_asr_s3prl_10min_fra1_10min](models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_10min_fra1_10min) | [RESULTS.md](models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_10min_fra1_10min/RESULTS.md) |
| deu1 | 10min | [train_asr_s3prl_10min_deu1_10min](models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_10min_deu1_10min) | [RESULTS.md](models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_10min_deu1_10min/RESULTS.md) |
| eng1 | 1h   | [train_asr_s3prl_1h_eng1_1h](models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_1h_eng1_1h) | [RESULTS.md](models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_1h_eng1_1h/RESULTS.md) |
| fra1 | 1h   | [train_asr_s3prl_1h_fra1_1h](models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_1h_fra1_1h) | [RESULTS.md](models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_1h_fra1_1h/RESULTS.md) |
| deu1 | 1h   | [train_asr_s3prl_1h_deu1_1h](models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_1h_deu1_1h) | [RESULTS.md](models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_1h_deu1_1h/RESULTS.md) |

## Absolute paths (copy-paste)

From repo root `/Data/janis.aiad/snlp`:

- **eng1 10min:** `models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_10min_eng1_10min`
- **fra1 10min:** `models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_10min_fra1_10min`
- **deu1 10min:** `models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_10min_deu1_10min`
- **eng1 1h:**   `models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_1h_eng1_1h`
- **fra1 1h:**   `models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_1h_fra1_1h`
- **deu1 1h:**   `models/espnet/egs2/ml_superb/asr1/exp/train_asr_s3prl_1h_deu1_1h`

## Per-experiment contents

Inside each experiment dir:

- `train.log` — training log
- `RESULTS.md` — CER/WER summary (generated at the end of decode + score)
- `decode_asr_asr_model_valid.loss.ave/<set>/score_cer/result.txt` — detailed CER
- `valid.loss.ave_5best.pth` (or similar) — best checkpoint

## Run the full grid

From repo root: `./scripts/run_ml_superb_ssl_full.sh`  
Only languages with data under `MLSUPERB` are run (eng1 needs `mls/eng`, fra1 `voxforge/fra`, deu1 `swc/deu`).
