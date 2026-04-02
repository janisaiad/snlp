#!/usr/bin/env python3
"""
Extract HuBERT (S3PRL upstream) features per transformer layer for ABX.
Writes out_dir/layer_{idx}/{utt_id}.npy for each selected layer index.
Usage: uv run python scripts/abx/extract_hubert_per_layer_for_abx.py --wav_scp ... --text ... --out_dir ... --layers 0,8,16,23
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_wav_scp(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            out[parts[0]] = parts[1].strip()
    return out


def parse_layers(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--upstream", default="hubert_large_ll60k")
    parser.add_argument(
        "--layers",
        default="",
        help="Comma-separated layer indices (0-based into S3PRL hidden_states list). Empty = auto 6 evenly spaced.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_utts", type=int, default=0, help="0 = all utterances")
    args = parser.parse_args()

    wavs = load_wav_scp(args.wav_scp)
    utt_ids = sorted(wavs.keys())
    if args.max_utts > 0:
        utt_ids = utt_ids[: args.max_utts]

    from s3prl.nn import S3PRLUpstream

    model = S3PRLUpstream(args.upstream)
    model.eval()
    model.to(args.device)

    with torch.no_grad():
        wav_path = wavs[utt_ids[0]]
        wav, sr = sf.read(wav_path, dtype="float32")
        if sr != 16000:
            import resampy

            wav = resampy.resample(wav, sr, 16000)
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(args.device)
        wav_len = torch.tensor([wav_t.shape[1]], device=args.device, dtype=torch.long)
        hs, _hs_len = model(wav_t, wav_len)
        if not isinstance(hs, (list, tuple)):
            print("Upstream did not return a layer list.", file=sys.stderr)
            return 1
        n_layers = len(hs)
        if args.layers.strip():
            layer_ids = parse_layers(args.layers)
        else:
            layer_ids = [int(round(i)) for i in np.linspace(0, n_layers - 1, num=6)]
        for li in layer_ids:
            if li < 0 or li >= n_layers:
                print(f"Layer {li} out of range [0, {n_layers - 1}]", file=sys.stderr)
                return 1

    for li in layer_ids:
        os.makedirs(os.path.join(args.out_dir, f"layer_{li}"), exist_ok=True)

    for utt_id in utt_ids:
        wav_path = wavs[utt_id]
        wav, sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            import resampy

            wav = resampy.resample(wav, sr, 16000)
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(args.device)
        wav_len = torch.tensor([wav_t.shape[1]], device=args.device, dtype=torch.long)
        with torch.no_grad():
            hs, _hs_len = model(wav_t, wav_len)
        for li in layer_ids:
            h = hs[li].squeeze(0).float().cpu().numpy()
            np.save(os.path.join(args.out_dir, f"layer_{li}", f"{utt_id}.npy"), h.astype(np.float32))

    print(f"Extracted {len(utt_ids)} utts, layers {layer_ids}, to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
