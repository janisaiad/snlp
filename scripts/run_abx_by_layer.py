#!/usr/bin/env python3
"""
Run fastabx per layer for HuBERT or WavJEPA-HF feature trees produced by extract_*_per_layer_for_abx.py.
Writes refs/plots/abx_by_layer_{backend}.csv with columns layer,abx_error.
Usage:
  uv run python scripts/run_abx_by_layer.py --backend hubert --recipe_dir ... --data_name dev_10min --feat_root ...
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _npy_mean1_feature_maker(path: str | Path) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2 and arr.shape[0] > 1:
        arr = arr.mean(axis=0, keepdims=True)
    return torch.from_numpy(arr)


def run_abx_for_dir(
    item_file: Path,
    feat_dir: Path,
    frequency: float,
) -> float | None:
    try:
        from fastabx import zerospeech_abx
    except ImportError:
        print("Install fastabx: uv pip install fastabx", file=sys.stderr)
        return None

    return zerospeech_abx(
        str(item_file),
        str(feat_dir),
        max_size_group=10,
        max_x_across=5,
        speaker="within",
        context="within",
        distance="cosine",
        frequency=frequency,
        seed=0,
        feature_maker=_npy_mean1_feature_maker,
        extension=".npy",
    )


def write_items(
    wav_scp: Path,
    text_path: Path,
    feat_dir: Path,
    item_out: Path,
    hz: float,
    *,
    mean_pooled_item: bool = True,
) -> int:
    with open(wav_scp) as f:
        utt_ids = [line.split(maxsplit=1)[0] for line in f if line.strip()]
    with open(text_path) as f:
        texts: dict[str, str] = {}
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            texts[parts[0]] = (parts[1].strip().upper() if len(parts) > 1 else "X")
    rows: list[tuple] = []
    for i, utt_id in enumerate(utt_ids):
        npy_path = feat_dir / f"{utt_id}.npy"
        if not npy_path.is_file():
            continue
        feat = np.load(str(npy_path))
        n_frames = feat.shape[0]
        onset = 0.0
        offset = (1.0 / float(hz)) if mean_pooled_item else (n_frames / float(hz))
        txt = texts.get(utt_id, "X")
        phone = txt[0] if txt and txt[0].isalnum() else "X"
        prev_phone = "SIL" if i == 0 else (
            texts.get(utt_ids[i - 1], "X")[0]
            if texts.get(utt_ids[i - 1], "X") and texts.get(utt_ids[i - 1], "X")[0].isalnum()
            else "SIL"
        )
        next_phone = "SIL" if i == len(utt_ids) - 1 else (
            texts.get(utt_ids[i + 1], "X")[0]
            if texts.get(utt_ids[i + 1], "X") and texts.get(utt_ids[i + 1], "X")[0].isalnum()
            else "SIL"
        )
        rows.append((utt_id, onset, offset, "MLSUPERB_POOL", phone, prev_phone, next_phone))
    item_out.parent.mkdir(parents=True, exist_ok=True)
    with open(item_out, "w") as f:
        f.write("#file onset offset speaker #phone prev-phone next-phone\n")
        for r in rows:
            f.write(f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]} {r[6]}\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("hubert", "wavjepa"), required=True)
    parser.add_argument(
        "--recipe_dir",
        default=str(REPO_ROOT / "models/espnet/egs2/ml_superb/asr1"),
    )
    parser.add_argument("--data_name", default="dev_10min")
    parser.add_argument(
        "--feat_root",
        required=True,
        help="Directory containing layer_* subfolders with .npy files",
    )
    parser.add_argument(
        "--out_csv",
        default=str(REPO_ROOT / "refs/plots/abx_by_layer.csv"),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to CSV instead of overwriting (for second backend).",
    )
    args = parser.parse_args()

    recipe = Path(args.recipe_dir).resolve()
    wav_scp = recipe / "data" / args.data_name / "wav.scp"
    text_path = recipe / "data" / args.data_name / "text"
    if not wav_scp.is_file() or not text_path.is_file():
        print("Missing wav.scp or text", file=sys.stderr)
        return 1

    hz = 50.0 if args.backend == "hubert" else 100.0
    feat_root = Path(args.feat_root).resolve()
    layer_dirs = sorted(feat_root.glob("layer_*"))
    if not layer_dirs:
        print(f"No layer_* under {feat_root}", file=sys.stderr)
        return 1

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[tuple[str, float | None]] = []
    write_header = not args.append or not out_csv.is_file()

    for layer_dir in layer_dirs:
        layer_name = layer_dir.name
        item_file = feat_root / f"items_{layer_name}.item"
        n = write_items(wav_scp, text_path, layer_dir, item_file, hz)
        if n < 3:
            print(f"Skip {layer_name}: only {n} segments with features", file=sys.stderr)
            rows_out.append((layer_name, None))
            continue
        err = run_abx_for_dir(item_file, layer_dir, hz)
        rows_out.append((layer_name, err))
        print(f"{layer_name}: ABX error = {err}")

    with open(out_csv, "a" if args.append else "w", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["backend", "layer", "abx_error"])
        for layer_name, err in rows_out:
            w.writerow([args.backend, layer_name, "" if err is None else f"{err:.6f}"])

    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
