#!/usr/bin/env python3
"""
Extract ESPnet JEPA minimal (mel + patch encoder) features for ABX.
Usage: uv run python scripts/abx/extract_jepa_minimal_for_abx.py --wav_scp ... --text ... --out_dir ...
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from espnet2.asr.frontend.jepa import JEPAFrontend

    wavs = load_wav_scp(args.wav_scp)
    utt_ids = sorted(wavs.keys())
    os.makedirs(args.out_dir, exist_ok=True)
    fe = JEPAFrontend(frontend_conf={})
    fe.eval()
    fe.to(args.device)

    for utt_id in utt_ids:
        wav_path = wavs[utt_id]
        wav, sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            import resampy

            wav = resampy.resample(wav, sr, 16000)
        w = torch.from_numpy(wav).float().unsqueeze(0).to(args.device)
        lengths = torch.tensor([w.shape[1]], device=args.device, dtype=torch.long)
        with torch.no_grad():
            h, _ = fe(w, lengths)
        arr = h.squeeze(0).float().cpu().numpy()
        np.save(os.path.join(args.out_dir, f"{utt_id}.npy"), arr.astype(np.float32))
    print(f"Extracted {len(utt_ids)} utterances to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
