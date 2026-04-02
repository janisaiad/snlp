#!/usr/bin/env python3
"""
Extract WavJEPA-Nat (HF) encoder layer outputs for ABX using forward hooks.
Writes out_dir/layer_{idx}/{utt_id}.npy for each selected encoder layer.
Usage: uv run python scripts/abx/extract_wavjepa_hf_per_layer_for_abx.py --wav_scp ... --out_dir ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModel

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    parser.add_argument("--model_name", default="labhamlet/wavjepa-nat-base")
    parser.add_argument(
        "--layers",
        default="",
        help="Comma-separated encoder layer indices. Empty = auto 6 evenly spaced.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_utts", type=int, default=0)
    args = parser.parse_args()

    wavs = load_wav_scp(args.wav_scp)
    utt_ids = sorted(wavs.keys())
    if args.max_utts > 0:
        utt_ids = utt_ids[: args.max_utts]

    extractor = AutoFeatureExtractor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.eval()
    model.to(args.device)

    enc_layers = model.model.encoder.layers
    n_layers = len(enc_layers)
    if args.layers.strip():
        layer_ids = parse_layers(args.layers)
    else:
        layer_ids = [int(round(i)) for i in np.linspace(0, n_layers - 1, num=min(6, n_layers))]
    for li in layer_ids:
        if li < 0 or li >= n_layers:
            print(f"Layer {li} out of range [0, {n_layers - 1}]", file=sys.stderr)
            return 1

    for li in layer_ids:
        os.makedirs(os.path.join(args.out_dir, f"layer_{li}"), exist_ok=True)

    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_module, _inp, out):
            if isinstance(out, tuple):
                captured[idx] = out[0].detach()
            else:
                captured[idx] = out.detach()

        return hook

    handles = [enc_layers[i].register_forward_hook(make_hook(i)) for i in layer_ids]

    try:
        for utt_id in utt_ids:
            wav_path = wavs[utt_id]
            wav, sr = sf.read(wav_path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                import resampy

                wav = resampy.resample(wav, sr, 16000)
            batch = extractor(
                [wav],
                return_tensors="pt",
                sampling_rate=extractor.sampling_rate,
                padding=True,
            )
            iv = batch["input_values"].to(args.device)
            captured.clear()
            with torch.no_grad():
                _ = model(iv)
            for li in layer_ids:
                if li not in captured:
                    print(f"Hook did not capture layer {li} for {utt_id}", file=sys.stderr)
                    return 1
                h = captured[li].squeeze(0).float().cpu().numpy()
                np.save(os.path.join(args.out_dir, f"layer_{li}", f"{utt_id}.npy"), h.astype(np.float32))
    finally:
        for h in handles:
            h.remove()

    print(f"Extracted {len(utt_ids)} utts, encoder layers {layer_ids}, to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
