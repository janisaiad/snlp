#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F


def parse_wav_scp(path: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        utt, wav = line.split(maxsplit=1)
        out.append((utt, Path(wav.strip())))
    return out


def parse_text(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(maxsplit=1)
        out[p[0]] = p[1] if len(p) > 1 else ""
    return out


def first_word(t: str) -> str:
    tok = t.strip().lower().split()
    return tok[0] if tok else "<empty>"


def load_wave(path: Path) -> torch.Tensor:
    x, sr = sf.read(str(path), dtype="float32")
    if sr != 16000:
        raise RuntimeError(f"Expected 16k, got {sr} for {path}")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return torch.from_numpy(x)


def build_frontends(local_ckpt: Path):
    from espnet2.asr.frontend.wavjepa import WavJEPAFrontend

    hf = WavJEPAFrontend(fs=16000, frontend_conf={"model_name": "labhamlet/wavjepa-nat-base"})
    loc = WavJEPAFrontend(
        fs=16000,
        frontend_conf={
            "model_name": "labhamlet/wavjepa-nat-base",
            "lightning_checkpoint_path": str(local_ckpt),
            "pretrain_in_channels": 1,
            "pretrain_process_seconds": 2.01,
            "pretrain_samples_per_audio": 8,
            "pretrain_average_top_k_layers": 12,
            "pretrain_compile_modules": False,
        },
    )
    hf.eval()
    loc.eval()
    return hf, loc


def embed_utt(frontend, wav: torch.Tensor) -> torch.Tensor:
    x = wav.unsqueeze(0)
    xl = torch.tensor([wav.numel()], dtype=torch.long)
    with torch.no_grad():
        feats, feat_lens = frontend(x, xl)
    t = int(feat_lens[0].item())
    vec = feats[0, :t].mean(dim=0)
    return vec.detach().cpu()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item())


def load_ctc_weight(exp_dir: Path) -> torch.Tensor:
    p = exp_dir / "valid.loss.ave.pth"
    obj = torch.load(p, map_location="cpu", weights_only=False)
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    if "ctc.ctc_lo.weight" in state:
        return state["ctc.ctc_lo.weight"].detach().float()
    for k in state.keys():
        if k.endswith("ctc_lo.weight"):
            return state[k].detach().float()
    raise KeyError(f"ctc weight not found in {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe_dir", default="models/espnet/egs2/ml_superb/asr1")
    ap.add_argument("--split", default="dev_10min")
    ap.add_argument(
        "--local_ckpt",
        default="logs/wavjepa_pretrain/saved_models_jepa_new_masking/Data=AudioSet/Extractor=wavjepa/InSeconds=2.01/BatchSize=32/NrSamples=8/NrGPUs=1/LR=0.0004/TargetProb=0.25/TargetLen=10/ContextProb=0.65/ContextLen=10/MinContextBlock=1/ContextRatio=0.1/last.ckpt",
    )
    ap.add_argument("--hf_exp", default="asr_train_asr_wavjepa_10min_eng1_10min")
    ap.add_argument("--local_exp", default="asr_train_asr_wavjepa_local_ckpt_10min_eng1_10min")
    ap.add_argument("--out", default="refs/WAVJEPA_HF_VS_LOCAL_ANALYSIS.md")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    recipe = (repo / args.recipe_dir).resolve()
    wav_scp = recipe / "data" / args.split / "wav.scp"
    text = recipe / "data" / args.split / "text"
    local_ckpt = (repo / args.local_ckpt).resolve()
    out_path = (repo / args.out).resolve()

    items = parse_wav_scp(wav_scp)
    texts = parse_text(text)
    hf_frontend, loc_frontend = build_frontends(local_ckpt)

    hf_vecs: dict[str, torch.Tensor] = {}
    loc_vecs: dict[str, torch.Tensor] = {}
    cosines: list[float] = []
    norm_ratios: list[float] = []

    for utt, wavp in items:
        w = load_wave(wavp)
        hv = embed_utt(hf_frontend, w)
        lv = embed_utt(loc_frontend, w)
        hf_vecs[utt] = hv
        loc_vecs[utt] = lv
        cosines.append(cosine(hv, lv))
        norm_ratios.append(float(lv.norm().item() / max(hv.norm().item(), 1e-8)))

    # Per-word centroid drift
    groups: dict[str, list[str]] = defaultdict(list)
    for utt, _ in items:
        groups[first_word(texts.get(utt, ""))].append(utt)

    word_rows = []
    for w, utts in groups.items():
        if len(utts) < 3:
            continue
        h = torch.stack([hf_vecs[u] for u in utts], dim=0).mean(dim=0)
        l = torch.stack([loc_vecs[u] for u in utts], dim=0).mean(dim=0)
        c = cosine(h, l)
        word_rows.append((w, len(utts), c))
    word_rows.sort(key=lambda x: x[2])  # most drift first

    # CTC head comparison
    hf_ctc = load_ctc_weight(recipe / "exp" / args.hf_exp)
    loc_ctc = load_ctc_weight(recipe / "exp" / args.local_exp)
    min_classes = min(hf_ctc.size(0), loc_ctc.size(0))
    hf_ctc_m = hf_ctc[:min_classes]
    loc_ctc_m = loc_ctc[:min_classes]
    row_cos = F.cosine_similarity(hf_ctc_m, loc_ctc_m, dim=1)

    summary = {
        "n_utts": len(items),
        "utt_cos_mean": float(torch.tensor(cosines).mean().item()),
        "utt_cos_std": float(torch.tensor(cosines).std(unbiased=False).item()),
        "utt_cos_min": float(torch.tensor(cosines).min().item()),
        "utt_cos_max": float(torch.tensor(cosines).max().item()),
        "norm_ratio_mean_local_over_hf": float(torch.tensor(norm_ratios).mean().item()),
        "ctc_row_cos_mean": float(row_cos.mean().item()),
        "ctc_row_cos_std": float(row_cos.std(unbiased=False).item()),
    }

    lines = []
    lines.append("# WavJEPA HF vs Local Checkpoint Analysis")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Split: `{args.split}`")
    lines.append(f"- Utterances analyzed: **{len(items)}**")
    lines.append(f"- Local checkpoint: `{local_ckpt}`")
    lines.append(f"- Downstream HF exp: `{args.hf_exp}`")
    lines.append(f"- Downstream Local exp: `{args.local_exp}`")
    lines.append("")
    lines.append("## Representation-space drift (utterance mean embeddings)")
    lines.append(f"- Cosine(HF, Local) mean: **{summary['utt_cos_mean']:.4f}**")
    lines.append(f"- Cosine std: {summary['utt_cos_std']:.4f} | min: {summary['utt_cos_min']:.4f} | max: {summary['utt_cos_max']:.4f}")
    lines.append(f"- Norm ratio Local/HF mean: **{summary['norm_ratio_mean_local_over_hf']:.4f}**")
    lines.append("")
    lines.append("Interpretation: cosine near 0 (slightly negative) indicates a strong representational rotation/drift between HF and local ckpt, despite same output dimensionality.")
    lines.append("")
    lines.append("## Word-level centroid drift (first token proxy)")
    lines.append("| word | n | cosine(HF,Local) |")
    lines.append("|---|---:|---:|")
    for w, n, c in word_rows[:20]:
        lines.append(f"| {w} | {n} | {c:.4f} |")
    lines.append("")
    lines.append("## Downstream CTC head comparison (HF run vs Local run)")
    lines.append(f"- CTC row cosine mean: **{summary['ctc_row_cos_mean']:.4f}**")
    lines.append(f"- CTC row cosine std: {summary['ctc_row_cos_std']:.4f}")
    lines.append("")
    lines.append("Interpretation: even with similar global CER/WER, the learned class hyperplanes can differ while preserving overall error rate due to dataset size and decoding constraints.")
    lines.append("")
    lines.append("## Raw summary (JSON)")
    lines.append("```json")
    lines.append(json.dumps(summary, indent=2))
    lines.append("```")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

