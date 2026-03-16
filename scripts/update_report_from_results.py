#!/usr/bin/env python3
"""
Update refs/rendu1.md ASR results table from exp/*/RESULTS.md.
Parses each RESULTS.md for CER/WER (test_10min_*), then rewrites the table section.
Usage: from repo root, uv run python scripts/update_report_from_results.py
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = REPO_ROOT / "models/espnet/egs2/ml_superb/asr1"
RENDU1 = REPO_ROOT / "refs/rendu1.md"

CONFIG_TO_FRONTEND = {
    "train_asr_s3prl_10min": ("HuBERT (frozen)", "train_asr_s3prl_10min.yaml"),
    "train_asr_s3prl_1h": ("HuBERT 1h", "train_asr_s3prl_1h.yaml"),
    "train_asr_jepa_10min": ("JEPA minimal", "train_asr_jepa_10min.yaml"),
    "train_asr_wavjepa_10min": ("WavJEPA (HF)", "train_asr_wavjepa_10min.yaml"),
    "train_asr_wavjepa_5ep": ("WavJEPA 5ep", "train_asr_wavjepa_5ep.yaml"),
}


def parse_results_md(path: Path) -> list[tuple[str, str, float, float]]:
    """Return list of (test_set, config_base, cer_pct, wer_pct)."""
    text = path.read_text()
    exp_name = path.parent.name
    if not exp_name.startswith("asr_train_"):
        return []
    suffix = exp_name.replace("asr_train_", "", 1)
    config_base = "train_" + suffix.rsplit("_", 2)[0] if suffix.count("_") >= 2 else suffix
    results: dict[tuple[str, str], tuple[float, float]] = {}
    for metric, idx in (("### CER", 0), ("### WER", 1)):
        pos = text.find(metric)
        if pos < 0:
            continue
        block = text[pos : pos + 1500]
        next_h3 = block.find("\n### ", 5)
        if next_h3 > 0:
            block = block[: next_h3]
        for line in block.splitlines():
            if "test_10min_" not in line or not line.strip().startswith("|"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 9:
                continue
            dataset = parts[1]
            try:
                err = float(parts[8])
            except ValueError:
                continue
            m = re.search(r"test_10min_(\w+)", dataset)
            if not m:
                continue
            test_set = f"test_10min_{m.group(1)}"
            key = (test_set, config_base)
            prev = results.get(key, (-1.0, -1.0))
            if idx == 0:
                results[key] = (err, prev[1])
            else:
                results[key] = (prev[0], err)
    return [(t, c, cer, wer) for (t, c), (cer, wer) in results.items()]


def collect_all_results() -> dict[str, dict[str, tuple[float, float]]]:
    """config_base -> test_set -> (cer, wer)."""
    out: dict[str, dict[str, tuple[float, float]]] = {}
    exp_dir = RECIPE_DIR / "exp"
    if not exp_dir.exists():
        return out
    for results_md in exp_dir.glob("asr_train_*/RESULTS.md"):
        for test_set, config_base, cer, wer in parse_results_md(results_md):
            if config_base not in out:
                out[config_base] = {}
            prev = out[config_base].get(test_set, (-1.0, -1.0))
            out[config_base][test_set] = (cer if cer >= 0 else prev[0], wer if wer >= 0 else prev[1])
    return out


def build_table_md(data: dict[str, dict[str, tuple[float, float]]]) -> str:
    """Build markdown table(s) for rendu1."""
    lines = []
    # collect all test sets (e.g. test_10min_eng1, test_10min_fra1, test_10min_deu1)
    all_tests = sorted({t for c in data.values() for t in c})
    if not all_tests:
        return ""
    for test_set in all_tests:
        lang_label = test_set.replace("test_10min_", "")
        lines.append(f"\n#### {test_set}\n")
        lines.append("| Frontend | Config | CER | WER | Note |")
        lines.append("|----------|--------|-----|-----|------|")
        for config_base in sorted(data.keys()):
            frontend, config_file = CONFIG_TO_FRONTEND.get(
                config_base, (config_base, config_base + ".yaml")
            )
            cer_wer = data[config_base].get(test_set, (-1.0, -1.0))
            cer_s = f"{cer_wer[0]:.2f}%" if cer_wer[0] >= 0 else "—"
            wer_s = f"{cer_wer[1]:.2f}%" if cer_wer[1] >= 0 else "—"
            note = "30 ep" if "5ep" not in config_base else "5 ep"
            lines.append(f"| {frontend} | {config_file} | {cer_s} | {wer_s} | {note} |")
    return "\n".join(lines)


def update_rendu1(new_tables: str) -> None:
    """Replace ASR results table section in refs/rendu1.md."""
    content = RENDU1.read_text()
    start_marker = "### ASR results (eng1, 10 min, test set)"
    end_marker = "**Discussion:**"
    if start_marker not in content or end_marker not in content:
        return
    start = content.index(start_marker)
    end = content.index(end_marker, start)
    new_section = (
        start_marker
        + "\n\nEvaluation uses the same test split; CER/WER are indicative.\n\n"
        + new_tables
        + "\n\n"
    )
    content = content[:start] + new_section + content[end:]
    RENDU1.write_text(content)


def main() -> None:
    data = collect_all_results()
    if not data:
        print("[update_report_from_results] No RESULTS.md found; skipping.")
        return
    new_tables = build_table_md(data)
    if not new_tables:
        print("[update_report_from_results] No CER/WER parsed; skipping.")
        return
    update_rendu1(new_tables)
    print(f"[update_report_from_results] Updated {RENDU1}")


if __name__ == "__main__":
    main()
