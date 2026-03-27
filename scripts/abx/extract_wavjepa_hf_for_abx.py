#!/usr/bin/env python3
"""
Extract WavJEPA-Nat (Hugging Face) frame features for ABX, one .npy per utterance.
Usage: uv run python scripts/abx/extract_wavjepa_hf_for_abx.py --wav_scp ... --text ... --out_dir ...
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="labhamlet/wavjepa-nat-base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError:
        print("Install transformers.", file=sys.stderr)
        return 1

    wavs = load_wav_scp(args.wav_scp)
    utt_ids = sorted(wavs.keys())
    os.makedirs(args.out_dir, exist_ok=True)
    extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.eval()
    model.to(args.device)

    for utt_id in utt_ids:
        wav_path = wavs[utt_id]
        wav, sr = sf.read(wav_path, dtype="float32")
        if sr != 16000:
            import resampy

            wav = resampy.resample(wav, sr, 16000)
        waveforms = [wav]
        batch = extractor(
            waveforms,
            return_tensors="pt",
            sampling_rate=extractor.sampling_rate,
            padding=True,
        )
        iv = batch["input_values"].to(args.device)
        with torch.no_grad():
            out = model(iv)
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() == 4:
            out = out.mean(dim=1)
        h = out.squeeze(0).float().cpu().numpy()
        np.save(os.path.join(args.out_dir, f"{utt_id}.npy"), h.astype(np.float32))
    print(f"Extracted {len(utt_ids)} utterances to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
