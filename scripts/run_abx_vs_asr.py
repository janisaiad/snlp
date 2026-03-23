#!/usr/bin/env python3
"""
Run ABX (HuBERT) and compare to ASR CER. Uses ML-SUPERB data under the recipe.
Prerequisites: uv pip install fastabx (Python>=3.12), data prepared under recipe.
For meaningful ABX you need at least a few dozen segments (e.g. full dev set).
Usage (from repo root):
  uv run python scripts/run_abx_vs_asr.py --recipe_dir models/espnet/egs2/ml_superb/asr1
  uv run python scripts/run_abx_vs_asr.py --recipe_dir ... --data_name train_10min_eng1
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_asr_cer_from_results(results_path: str) -> float | None:
    if not os.path.isfile(results_path):
        return None
    with open(results_path) as f:
        content = f.read()
    # find CER table and first data row; Err is the 8th column (index 7)
    in_cer = False
    for line in content.splitlines():
        if "### CER" in line:
            in_cer = True
            continue
        if in_cer and line.startswith("|") and "decode" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 10:
                try:
                    return float(parts[8])
                except ValueError:
                    pass
            break
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe_dir", default=None, help="Path to asr1 recipe")
    parser.add_argument("--data_name", default="dev_10min_eng1", help="Data split name under recipe/data/")
    parser.add_argument("--wav_scp", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--asr_results", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    recipe = Path(args.recipe_dir or (REPO_ROOT / "models/espnet/egs2/ml_superb/asr1")).resolve()
    data_name = args.data_name
    wav_scp = Path(args.wav_scp or recipe / "data" / data_name / "wav.scp")
    text_path = Path(args.text or recipe / "data" / data_name / "text")
    if not wav_scp.is_file() or not text_path.is_file():
        print("Missing wav.scp or text. Run data prep first or pass --wav_scp and --text.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir or recipe / "exp" / "abx_vs_asr")
    out_dir.mkdir(parents=True, exist_ok=True)
    item_file = out_dir / "dev_eng1.item"
    feat_dir = out_dir / "hubert_dev"

    # 1) Load utt list and texts
    with open(wav_scp) as f:
        utt_ids = [line.split(maxsplit=1)[0] for line in f if line.strip()]
    with open(text_path) as f:
        texts = {}
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            texts[parts[0]] = (parts[1].strip().upper() if len(parts) > 1 else "X")

    # 2) Extract HuBERT features (subprocess)
    extract_script = REPO_ROOT / "scripts" / "abx" / "extract_hubert_for_abx.py"
    r = subprocess.run(
        [sys.executable, str(extract_script), "--wav_scp", str(wav_scp), "--text", str(text_path), "--out_dir", str(feat_dir), "--device", "cpu"],
        cwd=str(REPO_ROOT),
    )
    if r.returncode != 0:
        return r.returncode

    # 3) Build item file with onset/offset (fastabx expects these; use feature length at 50Hz)
    frequency = 50
    import numpy as np
    rows = []
    for i, utt_id in enumerate(utt_ids):
        npy_path = feat_dir / f"{utt_id}.npy"
        if not npy_path.is_file():
            continue
        feat = np.load(str(npy_path))
        n_frames = feat.shape[0]
        onset = 0.0
        offset = n_frames / float(frequency)
        txt = texts.get(utt_id, "X")
        phone = txt[0] if txt and txt[0].isalnum() else "X"
        prev_phone = "SIL" if i == 0 else (texts.get(utt_ids[i - 1], "X")[0] if texts.get(utt_ids[i - 1], "X") and texts.get(utt_ids[i - 1], "X")[0].isalnum() else "SIL")
        next_phone = "SIL" if i == len(utt_ids) - 1 else (texts.get(utt_ids[i + 1], "X")[0] if texts.get(utt_ids[i + 1], "X") and texts.get(utt_ids[i + 1], "X")[0].isalnum() else "SIL")
        rows.append((utt_id, onset, offset, utt_id, phone, prev_phone, next_phone))
    with open(item_file, "w") as f:
        f.write("#file onset offset speaker #phone prev-phone next-phone\n")
        for r in rows:
            f.write(f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]} {r[6]}\n")
    print(f"Wrote {item_file} ({len(rows)} segments)")

    if len(rows) < 3:
        print("Warning: ABX needs at least 3 segments for triplets; result may be trivial.", file=sys.stderr)

    # 4) Run ABX
    try:
        import torch
        from fastabx import zerospeech_abx
    except ImportError:
        print("Install fastabx: uv pip install fastabx (Python>=3.12)", file=sys.stderr)
        return 1

    def npy_feature_maker(path: str | Path) -> torch.Tensor:
        arr = np.load(path)
        return torch.from_numpy(arr.astype(np.float32))

    abx_err = zerospeech_abx(
        str(item_file),
        str(feat_dir),
        max_size_group=10,
        max_x_across=5,
        speaker="within",
        context="within",
        distance="cosine",
        frequency=frequency,
        seed=0,
        feature_maker=npy_feature_maker,
        extension=".npy",
    )
    if abx_err is not None:
        print(f"ABX error rate (HuBERT, dev_10min_eng1): {abx_err:.4f}")
    else:
        print("ABX error rate: (not computed — too few segments for triplets)")

    # 5) ASR CER
    asr_path = args.asr_results or recipe / "exp" / "asr_train_asr_s3prl_10min_eng1_10min" / "RESULTS.md"
    cer = get_asr_cer_from_results(str(asr_path))
    if cer is not None:
        print(f"ASR CER (HuBERT frozen + CTC, test_10min_eng1): {cer:.2f}%")
    else:
        print("ASR CER: (no RESULTS.md found)")

    print("\n--- ABX vs ASR (HuBERT) ---")
    print("Model      | ABX err | ASR CER")
    print("-----------|---------|--------")
    abx_str = f"{abx_err:.4f}" if abx_err is not None else "—"
    cer_str = f"{cer:.2f}%" if cer is not None else "—"
    print(f"HuBERT L  | {abx_str}  | {cer_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
