#!/usr/bin/env python3
"""
Build a minimal .item file for fastabx from wav.scp + text.
Uses first character of transcript as pseudo-phone so we have multiple "phones" for ABX.
Format: #file speaker #phone prev-phone next-phone (space-separated, header line).
Usage: uv run python scripts/abx/build_item_from_text.py --wav_scp ... --text ... --out item.dev.item
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_wav_scp(wav_scp: str) -> list[str]:
    out = []
    with open(wav_scp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id = line.split(maxsplit=1)[0]
            out.append(utt_id)
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
            txt = parts[1].strip().upper() if len(parts) > 1 else "X"
            out[utt_id] = txt
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True, help="Output .item path")
    args = parser.parse_args()

    utt_ids = load_wav_scp(args.wav_scp)
    texts = load_text(args.text)
    # pseudo-phone: first char of transcript, or "X" if empty
    with open(args.out, "w") as f:
        f.write("#file speaker #phone prev-phone next-phone\n")
        for i, utt_id in enumerate(utt_ids):
            txt = texts.get(utt_id, "X")
            phone = txt[0] if txt and txt[0].isalnum() else "X"
            prev_phone = "SIL" if i == 0 else (texts.get(utt_ids[i - 1], "X")[0] if texts.get(utt_ids[i - 1], "X") else "X")
            next_phone = "SIL" if i == len(utt_ids) - 1 else (texts.get(utt_ids[i + 1], "X")[0] if texts.get(utt_ids[i + 1], "X") else "X")
            if not prev_phone.isalnum():
                prev_phone = "SIL"
            if not next_phone.isalnum():
                next_phone = "SIL"
            f.write(f"{utt_id} {utt_id} {phone} {prev_phone} {next_phone}\n")
    print(f"Wrote {args.out} ({len(utt_ids)} segments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
