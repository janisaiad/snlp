#!/usr/bin/env python3
"""
Extract HuBERT (hubert_large_ll60k) features for ABX.
Reads a wav.scp and text; saves one .npy per utterance to out_dir.
Usage: uv run python scripts/abx/extract_hubert_for_abx.py --wav_scp data/dev_10min_eng1/wav.scp --text data/dev_10min_eng1/text --out_dir exp/abx_hubert_dev
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

# we need s3prl and snlp repo on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_wav_scp(wav_scp: str) -> dict[str, str]:
    out = {}
    with open(wav_scp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt_id = parts[0]
            path = parts[1].strip()
            out[utt_id] = path
    return out


def load_text(text_path: str) -> dict[str, str]:
    out = {}
    with open(text_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt_id = parts[0]
            txt = parts[1].strip() if len(parts) > 1 else ""
            out[utt_id] = txt
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract HuBERT features for ABX")
    parser.add_argument("--wav_scp", required=True, help="Path to wav.scp")
    parser.add_argument("--text", required=True, help="Path to text")
    parser.add_argument("--out_dir", required=True, help="Output directory for .npy features")
    parser.add_argument("--upstream", default="hubert_large_ll60k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    wavs = load_wav_scp(args.wav_scp)
    texts = load_text(args.text)
    utt_ids = sorted(wavs.keys())
    os.makedirs(args.out_dir, exist_ok=True)

    from s3prl.nn import S3PRLUpstream
    model = S3PRLUpstream(args.upstream)
    model.eval()
    model.to(args.device)

    for utt_id in utt_ids:
        wav_path = wavs[utt_id]
        wav, sr = sf.read(wav_path, dtype="float32")
        if sr != 16000:
            import resampy
            wav = resampy.resample(wav, sr, 16000)
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(args.device)
        wav_len = torch.tensor([wav_t.shape[1]], device=args.device, dtype=torch.long)
        with torch.no_grad():
            hs, hs_len = model(wav_t, wav_len)
        # hs: list of layers; use last layer or mean
        h = hs[-1] if isinstance(hs, list) else hs
        if isinstance(h, list):
            h = h[-1]
        h = h.squeeze(0).cpu().numpy()
        out_path = os.path.join(args.out_dir, f"{utt_id}.npy")
        np.save(out_path, h.astype(np.float32))
    print(f"Extracted {len(utt_ids)} utterances to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
