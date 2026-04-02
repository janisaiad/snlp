#!/usr/bin/env python3
"""
Frozen-frontend SER on RAVDESS (4 classes: neutral, happy, sad, angry) using ESPnet JEPA minimal features.
Requires: RAVDESS unpacked with Actor_* folders; soundfile, scikit-learn, resampy.
Usage:
  uv run python scripts/ser_ravdess_frozen_frontend.py --ravdess_root /path/to/RAVDESS --backend jepa_minimal
  uv run python scripts/ser_ravdess_frozen_frontend.py --ravdess_root /path/to/RAVDESS --backend hubert_s3prl  # optional
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EMOTION_MAP = {"01": 0, "03": 1, "04": 2, "05": 3}
LABEL_NAMES = ["neutral", "happy", "sad", "angry"]


def list_ravdess_pairs(root: Path) -> list[tuple[Path, int]]:
    pairs: list[tuple[Path, int]] = []
    for wav in sorted(root.rglob("*.wav")):
        stem = wav.stem
        parts = stem.split("-")
        if len(parts) < 3:
            continue
        emo = parts[2]
        if emo not in EMOTION_MAP:
            continue
        if parts[0] != "03" or parts[1] != "01":
            continue
        pairs.append((wav, EMOTION_MAP[emo]))
    return pairs


def load_wav_mono(path: Path, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import resampy

        wav = resampy.resample(wav, sr, target_sr)
    return wav


def extract_jepa_minimal(wav: np.ndarray, device: str) -> np.ndarray:
    from espnet2.asr.frontend.jepa import JEPAFrontend

    fe = JEPAFrontend(frontend_conf={})
    fe.eval()
    fe.to(device)
    w = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
    lengths = torch.tensor([w.shape[2]], device=device, dtype=torch.long)
    with torch.no_grad():
        h, _ = fe(w, lengths)
    return h.squeeze(0).float().mean(dim=0).cpu().numpy()


def extract_hubert(wav: np.ndarray, device: str) -> np.ndarray:
    from s3prl.nn import S3PRLUpstream

    model = S3PRLUpstream("hubert_large_ll60k")
    model.eval()
    model.to(device)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], device=device, dtype=torch.long)
    with torch.no_grad():
        hs, _ = model(wav_t, wav_len)
        h = hs[-1] if isinstance(hs, list) else hs
        h = h.squeeze(0)
    return h.float().mean(dim=0).cpu().numpy()


def unweighted_average_recall(y_true: np.ndarray, y_pred: np.ndarray, n_class: int) -> float:
    recalls = []
    for c in range(n_class):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravdess_root", type=Path, required=True)
    parser.add_argument("--backend", choices=("jepa_minimal", "hubert_s3prl"), default="jepa_minimal")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = args.ravdess_root.expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    pairs = list_ravdess_pairs(root)
    if len(pairs) < 32:
        print(
            f"Found only {len(pairs)} usable Speech/4-class clips under {root}. "
            "Expected RAVDESS with 03-*-01|03|04|05-* filenames.",
            file=sys.stderr,
        )
        return 2

    random.seed(args.seed)
    random.shuffle(pairs)
    n_train = max(1, int(len(pairs) * args.train_frac))
    train_p, test_p = pairs[:n_train], pairs[n_train:]

    extract = extract_jepa_minimal if args.backend == "jepa_minimal" else extract_hubert

    X_train, y_train = [], []
    for path, lab in train_p:
        wav = load_wav_mono(path, 16000)
        X_train.append(extract(wav, args.device))
        y_train.append(lab)
    X_test, y_test = [], []
    for path, lab in test_p:
        wav = load_wav_mono(path, 16000)
        X_test.append(extract(wav, args.device))
        y_test.append(lab)

    X_train = np.stack(X_train)
    X_test = np.stack(X_test)
    y_train = np.array(y_train, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    clf = LogisticRegression(max_iter=2000, random_state=args.seed)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    uar = unweighted_average_recall(y_test, pred, len(LABEL_NAMES))
    print(f"backend={args.backend} n_train={len(y_train)} n_test={len(y_test)} acc={acc:.4f} uar={uar:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
