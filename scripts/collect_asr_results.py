#!/usr/bin/env python3
"""
Collect CER/WER from ML-SUPERB exp RESULTS.md into a single table.
Run from repo root. Writes refs/ASR_RESULTS_TABLE_eng1.md (monolingual eng1 only)
and prints to stdout. The consolidated multilingual tables live in
refs/ASR_RESULTS_TABLE.md (maintained by hand; do not overwrite).
"""
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "models/espnet/egs2/ml_superb/asr1/exp"

# Map exp dir name -> display name
TAG_TO_NAME = {
    "asr_train_asr_s3prl_10min_eng1_10min": "HuBERT (frozen)",
    "asr_train_asr_jepa_10min_eng1_10min": "JEPA minimal",
    "asr_train_asr_wavjepa_10min_eng1_10min": "WavJEPA (HF)",
    "asr_train_asr_wavjepa_local_ckpt_10min_eng1_10min": "WavJEPA (local pretrain ckpt)",
}


def parse_results_md(path: Path) -> dict[str, str]:
    """Extract CER and WER for test_10min_eng1 from RESULTS.md. Err column is index 8 in split |...|."""
    text = path.read_text()
    out = {}
    for metric in ("CER", "WER"):
        section = re.search(rf"### {metric}\s*\n.*?(?=###|\Z)", text, re.DOTALL)
        if not section:
            continue
        for line in section.group(0).splitlines():
            if "test_10min_eng1" not in line or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 10:
                try:
                    err = float(parts[8].strip())
                    out[metric] = f"{err:.2f}%"
                    break
                except (ValueError, IndexError):
                    pass
    return out


def main():
    rows = []
    for exp_name, display_name in TAG_TO_NAME.items():
        results_path = EXP_DIR / exp_name / "RESULTS.md"
        if not results_path.exists():
            rows.append((display_name, "—", "—", "(no RESULTS.md)"))
            continue
        parsed = parse_results_md(results_path)
        cer = parsed.get("CER", "—")
        wer = parsed.get("WER", "—")
        rows.append((display_name, cer, wer, ""))
    # Write table
    out_path = REPO_ROOT / "refs/ASR_RESULTS_TABLE_eng1.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ASR results — monolingual eng1 only (auto-generated)",
        "",
        "See `refs/ASR_RESULTS_TABLE.md` for multilingual, ASR+LID, and LoRA rows.",
        "",
        "# ASR results (eng1, 10 min, test_10min_eng1)",
        "",
        "| Frontend | CER | WER | Note |",
        "|----------|-----|-----|------|",
    ]
    for display_name, cer, wer, note in rows:
        lines.append(f"| {display_name} | {cer} | {wer} | {note} |")
    lines.extend(["", f"Generated from `exp/*/RESULTS.md` under {EXP_DIR}."])
    out_path.write_text("\n".join(lines))
    print(out_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
