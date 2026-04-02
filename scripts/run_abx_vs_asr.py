#!/usr/bin/env python3
"""
ABX (fastabx / ZeroSpeech-style) + comparaison ASR CER pour HuBERT vs WavJEPA HF.
Sans DTW : moyenne temporelle des frames par fichier + fenêtre item d’une frame (compatible PyTorch 2.7 / torchdtw cassé).
Usage (depuis la racine du repo) :
  uv run python scripts/run_abx_vs_asr.py --recipe_dir models/espnet/egs2/ml_superb/asr1 --data_name dev_10min
  uv run python scripts/run_abx_vs_asr.py ... --skip-wavjepa   # HuBERT seul
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_asr_cer_from_results(results_path: str) -> float | None:
    if not os.path.isfile(results_path):
        return None
    with open(results_path) as f:
        content = f.read()
    in_cer = False
    pick: list[tuple[list[str], float]] = []
    for line in content.splitlines():
        if "### CER" in line:
            in_cer = True
            continue
        if in_cer and line.startswith("### "):
            break
        if in_cer and line.startswith("|") and "---" not in line and "dataset" not in line.lower():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 10:
                try:
                    err = float(parts[8])
                    pick.append((parts, err))
                except ValueError:
                    pass
    for parts, err in pick:
        row = "|".join(parts)
        if "test_10min" in row:
            return err
    return pick[0][1] if pick else None


def run_abx_on_feats(
    *,
    item_file: Path,
    feat_dir: Path,
    frequency: float,
    pool_mean: bool,
) -> float | None:
    import numpy as np
    import torch
    from fastabx import zerospeech_abx

    def npy_feature_maker(path: str | Path) -> torch.Tensor:
        arr = np.load(str(path)).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if pool_mean:
            arr = arr.mean(axis=0, keepdims=True)
        return torch.from_numpy(arr)

    return zerospeech_abx(
        str(item_file),
        str(feat_dir),
        max_size_group=10,
        max_x_across=5,
        speaker="within",
        context="within",
        distance="cosine",
        frequency=int(frequency),
        seed=0,
        feature_maker=npy_feature_maker,
        extension=".npy",
    )


def write_item_file_one_frame(
    *,
    item_path: Path,
    utt_ids: list[str],
    texts: dict[str, str],
    frequency: float,
) -> None:
    """Fenêtre d’une frame : avec fastabx défaut (sans LibriLight bug), end += 1 après floor."""
    offset = 1.0 / frequency
    with open(item_path, "w") as f:
        f.write("#file onset offset speaker #phone prev-phone next-phone\n")
        pseudo_spk = "S0"
        for i, utt_id in enumerate(utt_ids):
            txt = texts.get(utt_id, "X")
            phone = txt[0] if txt and txt[0].isalnum() else "X"
            prev_txt = texts.get(utt_ids[i - 1], "X") if i > 0 else "X"
            next_txt = texts.get(utt_ids[i + 1], "X") if i + 1 < len(utt_ids) else "X"
            prev_phone = prev_txt[0] if prev_txt and prev_txt[0].isalnum() else "SIL"
            next_phone = next_txt[0] if next_txt and next_txt[0].isalnum() else "SIL"
            f.write(f"{utt_id} 0.0 {offset} {pseudo_spk} {phone} {prev_phone} {next_phone}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe_dir", default=None, help="Chemin recette asr1 ml_superb")
    parser.add_argument(
        "--data_name",
        default="dev_10min",
        help="Split sous recipe/data/ (ex. dev_10min ~300 utt).",
    )
    parser.add_argument("--wav_scp", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Device extraction (HuBERT / WavJEPA).",
    )
    parser.add_argument(
        "--no-pool-mean",
        action="store_true",
        help="Utiliser toutes les frames (exige torchdtw valide ; souvent faux sur torch 2.7).",
    )
    parser.add_argument(
        "--skip-wavjepa",
        action="store_true",
        help="Ne pas extraire / scorer WavJEPA.",
    )
    parser.add_argument(
        "--hubert-results",
        default=None,
        help="RESULTS.md ASR HuBERT (défaut : exp s3prl eng1 10min).",
    )
    parser.add_argument(
        "--wavjepa-results",
        default=None,
        help="RESULTS.md ASR WavJEPA HF (défaut : exp wavjepa eng1 10min).",
    )
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Ne pas ré-extraire ; réutiliser hubert_feats / wavjepa_hf_feats si présents.",
    )
    args = parser.parse_args()

    recipe = Path(args.recipe_dir or (REPO_ROOT / "models/espnet/egs2/ml_superb/asr1")).resolve()
    data_name = args.data_name
    wav_scp = Path(args.wav_scp or recipe / "data" / data_name / "wav.scp")
    text_path = Path(args.text or recipe / "data" / data_name / "text")
    if not wav_scp.is_file() or not text_path.is_file():
        print("wav.scp ou text manquant.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir or recipe / "exp" / "abx_vs_asr")
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_hubert = out_dir / "hubert_feats"
    feat_wavjepa = out_dir / "wavjepa_hf_feats"
    item_file = out_dir / f"items_{data_name}_mean1frame.item"
    pool_mean = not args.no_pool_mean
    frequency = 50.0

    texts: dict[str, str] = {}
    with open(text_path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            texts[parts[0]] = (parts[1].strip().upper() if len(parts) > 1 else "X")

    import torch as torch_mod

    dev = "cuda" if args.device == "auto" and torch_mod.cuda.is_available() else (args.device if args.device != "auto" else "cpu")

    extract_hubert = REPO_ROOT / "scripts" / "abx" / "extract_hubert_for_abx.py"
    if not args.reuse_features or not any(feat_hubert.glob("*.npy")):
        sub = subprocess.run(
            [sys.executable, str(extract_hubert), "--wav_scp", str(wav_scp), "--text", str(text_path), "--out_dir", str(feat_hubert), "--device", dev],
            cwd=str(REPO_ROOT),
        )
        if sub.returncode != 0:
            return sub.returncode

    if not args.skip_wavjepa:
        extract_wj = REPO_ROOT / "scripts" / "abx" / "extract_wavjepa_hf_for_abx.py"
        if not args.reuse_features or not any(feat_wavjepa.glob("*.npy")):
            sub = subprocess.run(
                [sys.executable, str(extract_wj), "--wav_scp", str(wav_scp), "--text", str(text_path), "--out_dir", str(feat_wavjepa), "--device", dev],
                cwd=str(REPO_ROOT),
            )
            if sub.returncode != 0:
                return sub.returncode

    import numpy as np

    with open(wav_scp) as f:
        all_utts = [line.split(maxsplit=1)[0] for line in f if line.strip()]
    have_h = {u for u in all_utts if (feat_hubert / f"{u}.npy").is_file()}
    if args.skip_wavjepa:
        common = sorted(have_h)
    else:
        have_w = {u for u in all_utts if (feat_wavjepa / f"{u}.npy").is_file()}
        common = sorted(have_h & have_w)
    if len(common) < 3:
        print(f"Trop peu d’utterances communes avec features ({len(common)}).", file=sys.stderr)
        return 1

    write_item_file_one_frame(item_path=item_file, utt_ids=common, texts=texts, frequency=frequency)
    print(f"Item file : {item_file} ({len(common)} segments), pool_mean={pool_mean}")

    try:
        abx_h = run_abx_on_feats(item_file=item_file, feat_dir=feat_hubert, frequency=frequency, pool_mean=pool_mean)
    except Exception as exc:
        print(f"ABX HuBERT échoué : {exc}", file=sys.stderr)
        return 1

    abx_w: float | None = None
    if not args.skip_wavjepa:
        try:
            abx_w = run_abx_on_feats(item_file=item_file, feat_dir=feat_wavjepa, frequency=frequency, pool_mean=pool_mean)
        except Exception as exc:
            print(f"ABX WavJEPA échoué : {exc}", file=sys.stderr)
            abx_w = None

    hub_md = args.hubert_results or str(recipe / "exp" / "asr_train_asr_s3prl_10min_eng1_10min" / "RESULTS.md")
    wj_md = args.wavjepa_results or str(recipe / "exp" / "asr_train_asr_wavjepa_10min_eng1_10min" / "RESULTS.md")
    cer_h = get_asr_cer_from_results(hub_md)
    cer_w = get_asr_cer_from_results(wj_md) if not args.skip_wavjepa else None

    def fmt_abx(x: float | None) -> str:
        return f"{x:.4f}" if x is not None else "—"

    def fmt_cer(x: float | None) -> str:
        return f"{x:.2f}%" if x is not None else "—"

    print("\n--- ABX vs ASR (moyenne temporelle + 1 frame item ; pas de DTW) ---")
    print("Modèle      | ABX err | ASR CER (test decode si dispo)")
    print("------------|---------|------------------")
    print(f"HuBERT L    | {fmt_abx(abx_h)} | {fmt_cer(cer_h)}")
    if not args.skip_wavjepa:
        print(f"WavJEPA HF  | {fmt_abx(abx_w)} | {fmt_cer(cer_w)}")
        if abx_h is not None and abx_w is not None:
            delta = abx_w - abx_h
            print(f"Δ (W − H)   | {delta:+.4f} | —")
    return 0


if __name__ == "__main__":
    sys.exit(main())
