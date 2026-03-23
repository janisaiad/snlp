#!/usr/bin/env python3
"""
Run ABX (fastabx) for HuBERT, WavJEPA (HF), and JEPA minimal on the same ML-SUPERB split.
Requires: uv pip install fastabx, s3prl, transformers; data under recipe/data/<data_name>/.
Usage: uv run python scripts/run_abx_all_frontends.py [--recipe_dir ...] [--data_name dev_10min_eng1]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_abx_for_dir(
    item_file: Path,
    feat_dir: Path,
    frequency: float,
) -> float | None:
    try:
        import torch
        from fastabx import zerospeech_abx
    except ImportError:
        print("Install fastabx: uv pip install fastabx", file=sys.stderr)
        return None

    def npy_feature_maker(path: str | Path) -> torch.Tensor:
        arr = np.load(path)
        return torch.from_numpy(arr.astype(np.float32))

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
        feature_maker=npy_feature_maker,
        extension=".npy",
    )


def write_items(
    wav_scp: Path,
    text_path: Path,
    feat_dir: Path,
    item_out: Path,
    hz: float,
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
        offset = n_frames / float(hz)
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
        rows.append((utt_id, onset, offset, utt_id, phone, prev_phone, next_phone))
    item_out.parent.mkdir(parents=True, exist_ok=True)
    with open(item_out, "w") as f:
        f.write("#file onset offset speaker #phone prev-phone next-phone\n")
        for r in rows:
            f.write(f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]} {r[6]}\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe_dir",
        default=str(REPO_ROOT / "models/espnet/egs2/ml_superb/asr1"),
    )
    parser.add_argument("--data_name", default="dev_10min_eng1")
    args = parser.parse_args()
    recipe = Path(args.recipe_dir).resolve()
    data_name = args.data_name
    wav_scp = recipe / "data" / data_name / "wav.scp"
    text_path = recipe / "data" / data_name / "text"
    if not wav_scp.is_file() or not text_path.is_file():
        print("Missing wav.scp or text.", file=sys.stderr)
        return 1

    out_dir = recipe / "exp" / "abx_all_frontends"
    hubert_dir = out_dir / "hubert"
    wavjepa_dir = out_dir / "wavjepa_hf"
    jepa_dir = out_dir / "jepa_minimal"
    py = sys.executable

    for script, odir in (
        ("scripts/abx/extract_hubert_for_abx.py", hubert_dir),
        ("scripts/abx/extract_wavjepa_hf_for_abx.py", wavjepa_dir),
        ("scripts/abx/extract_jepa_minimal_for_abx.py", jepa_dir),
    ):
        extra = []
        if "hubert" in script:
            extra = ["--device", "cpu"]
        r = subprocess.run(
            [
                py,
                str(REPO_ROOT / script),
                "--wav_scp",
                str(wav_scp),
                "--text",
                str(text_path),
                "--out_dir",
                str(odir),
            ]
            + extra,
            cwd=str(REPO_ROOT),
        )
        if r.returncode != 0:
            return r.returncode

    item_h = out_dir / "items_hubert.item"
    item_w = out_dir / "items_wavjepa.item"
    item_j = out_dir / "items_jepa.item"
    n1 = write_items(wav_scp, text_path, hubert_dir, item_h, 50.0)
    n2 = write_items(wav_scp, text_path, wavjepa_dir, item_w, 100.0)
    n3 = write_items(wav_scp, text_path, jepa_dir, item_j, 100.0)
    print(f"Segments with features: HuBERT={n1} WavJEPA={n2} JEPA_min={n3}")

    abx_h = run_abx_for_dir(item_h, hubert_dir, 50.0)
    abx_w = run_abx_for_dir(item_w, wavjepa_dir, 100.0)
    abx_j = run_abx_for_dir(item_j, jepa_dir, 100.0)

    def fmt(x: float | None) -> str:
        return f"{x:.4f}" if x is not None else "—"

    print("\n--- ABX error rate (lower is better) ---")
    print(f"HuBERT (s3prl):     {fmt(abx_h)}")
    print(f"WavJEPA (HF):       {fmt(abx_w)}")
    print(f"JEPA minimal:       {fmt(abx_j)}")
    report = out_dir / "ABX_SUMMARY.txt"
    report.write_text(
        "\n".join(
            [
                f"HuBERT_s3prl_abx_err\t{abx_h}",
                f"WavJEPA_HF_abx_err\t{abx_w}",
                f"JEPA_minimal_abx_err\t{abx_j}",
                "",
            ]
        )
    )
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
