#!/usr/bin/env python3
"""
Layer-wise phonetic discriminability proxy (NOT official fastabx).
For each layer folder with .npy (T, D) per utterance, mean-pools to (D,), keeps labels
from first character of reference text, then score = between-class / within-class
mean cosine distance for classes with at least min_count utterances.
Usage:
  uv run python scripts/layer_phonetic_proxy.py --wav_scp ... --text ... --feat_root exp/abx_layers/hubert --out_csv refs/plots/layer_proxy_hubert.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_wav_scp(path: Path) -> list[str]:
    order: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            order.append(line.split(maxsplit=1)[0])
    return order


def load_text(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt = parts[0]
            txt = parts[1].strip().upper() if len(parts) > 1 else "X"
            out[utt] = txt
    return out


def mean_pool(path: Path) -> np.ndarray:
    x = np.load(path).astype(np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D features, got {x.shape} for {path}")
    v = x.mean(axis=0)
    n = np.linalg.norm(v)
    if n > 1e-12:
        v = v / n
    return v


def score_embeddings(emb: np.ndarray, labels: list[str], min_count: int) -> float:
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[lab].append(i)
    kept = [lab for lab, idx in by_label.items() if len(idx) >= min_count]
    if len(kept) < 2:
        return float("nan")
    idxs = [i for lab in kept for i in by_label[lab]]
    sub_emb = emb[idxs]
    sub_lab = [labels[i] for i in idxs]
    centroids: dict[str, np.ndarray] = {}
    for lab in kept:
        ii = [j for j, l in enumerate(sub_lab) if l == lab]
        centroids[lab] = sub_emb[ii].mean(axis=0)
        n = np.linalg.norm(centroids[lab])
        if n > 1e-12:
            centroids[lab] /= n
    within: list[float] = []
    for lab in kept:
        ii = [j for j, l in enumerate(sub_lab) if l == lab]
        c = centroids[lab]
        for j in ii:
            within.append(float(1.0 - np.dot(sub_emb[j], c)))
    between: list[float] = []
    labs = list(centroids.keys())
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            between.append(float(1.0 - np.dot(centroids[labs[i]], centroids[labs[j]])))
    if not within or not between:
        return float("nan")
    return float(np.mean(between) / (np.mean(within) + 1e-8))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--feat_root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--min_count", type=int, default=8)
    args = parser.parse_args()

    wav_scp = Path(args.wav_scp)
    text_path = Path(args.text)
    feat_root = Path(args.feat_root)
    order = load_wav_scp(wav_scp)
    texts = load_text(text_path)
    labels_full = []
    for utt in order:
        t = texts.get(utt, "X")
        ch = t[0] if t and t[0].isalnum() else "X"
        labels_full.append(ch)

    rows: list[tuple[str, float]] = []
    for layer_dir in sorted(feat_root.glob("layer_*")):
        embs: list[np.ndarray] = []
        labels: list[str] = []
        for idx, utt in enumerate(order):
            p = layer_dir / f"{utt}.npy"
            if not p.is_file():
                continue
            embs.append(mean_pool(p))
            labels.append(labels_full[idx])
        if len(embs) < 30:
            rows.append((layer_dir.name, float("nan")))
            continue
        emb = np.stack(embs, axis=0)
        s = score_embeddings(emb, labels, args.min_count)
        rows.append((layer_dir.name, s))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "proxy_score"])
        for name, val in rows:
            w.writerow([name, "" if np.isnan(val) else f"{val:.6f}"])
    print(f"Wrote {out} ({len(rows)} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
