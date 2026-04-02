from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


OUTPUT_DIR = Path("/root/snlp/refs/plots")
PRIMARY = "#355C7D"
SECONDARY = "#C06C84"
ACCENT = "#F67280"
SUCCESS = "#2A9D8F"
WARNING = "#E9C46A"
DANGER = "#E76F51"
DARK = "#264653"
LIGHT = "#F7F3E9"


@dataclass(frozen=True)
class PlotArtifact:
    filename: str
    title: str
    description: str
    kind: str


def set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def save(fig: plt.Figure, stem: str, artifacts: list[PlotArtifact], title: str, description: str, kind: str) -> None:
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    artifacts.append(
        PlotArtifact(
            filename=f"{stem}.png",
            title=title,
            description=description,
            kind=kind,
        )
    )
    plt.close(fig)


def wrap_labels(labels: Sequence[str], max_words: int = 2) -> list[str]:
    wrapped: list[str] = []
    for label in labels:
        words = label.split()
        if len(words) <= max_words:
            wrapped.append(label)
            continue
        chunks = [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]
        wrapped.append("\n".join(chunks))
    return wrapped


def plot_monolingual_frontend_bars(artifacts: list[PlotArtifact]) -> None:
    labels = ["HuBERT", "JEPA minimal", "WavJEPA HF", "WavJEPA local"]
    cer = np.array([33.33, 62.22, 33.33, 33.33])
    wer = np.array([24.14, 44.83, 24.14, 24.14])
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(x - width / 2, cer, width, label="CER", color=PRIMARY)
    ax.bar(x + width / 2, wer, width, label="WER", color=SECONDARY)
    for idx, value in enumerate(cer):
        ax.text(idx - width / 2, value + 1.2, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(wer):
        ax.text(idx + width / 2, value + 1.2, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels(labels))
    ax.set_ylabel("Error (%)")
    ax.set_title("Monolingual eng1 10 min: frontend comparison")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 70)
    ax.text(
        0.02,
        0.95,
        "HuBERT and WavJEPA are tied here; minimal JEPA is clearly behind.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=DARK,
    )
    save(
        fig,
        "01_monolingual_frontend_comparison",
        artifacts,
        "Monolingual eng1 10 min frontend comparison",
        "Grouped CER/WER bars for HuBERT, minimal JEPA, WavJEPA HF, and local WavJEPA checkpoint.",
        "data",
    )


def plot_delta_vs_hubert(artifacts: list[PlotArtifact]) -> None:
    labels = ["JEPA minimal", "WavJEPA HF", "WavJEPA local"]
    delta_cer = np.array([28.89, 0.0, 0.0])
    delta_wer = np.array([20.69, 0.0, 0.0])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axvline(0.0, color="#444444", linewidth=1.2)
    ax.barh(y + 0.18, delta_cer, height=0.32, color=PRIMARY, label="Delta CER vs HuBERT")
    ax.barh(y - 0.18, delta_wer, height=0.32, color=SECONDARY, label="Delta WER vs HuBERT")
    for idx, value in enumerate(delta_cer):
        ax.text(value + 0.7, idx + 0.18, f"{value:+.2f}", va="center", fontsize=9)
    for idx, value in enumerate(delta_wer):
        ax.text(value + 0.7, idx - 0.18, f"{value:+.2f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Error delta (%)")
    ax.set_title("Delta vs HuBERT baseline")
    ax.legend(loc="lower right")
    save(
        fig,
        "02_delta_vs_hubert",
        artifacts,
        "Delta vs HuBERT baseline",
        "Horizontal delta bars showing how far each JEPA/WavJEPA option sits from HuBERT on CER and WER.",
        "data",
    )


def plot_abx_vs_asr(artifacts: list[PlotArtifact]) -> None:
    names = ["HuBERT", "WavJEPA HF"]
    abx = np.array([0.5216, 0.5895])
    wer = np.array([24.14, 24.14])
    cer = np.array([33.33, 33.33])
    colors = [PRIMARY, SECONDARY]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharex=True)
    for ax, target, ylabel, target_name in [
        (axes[0], wer, "WER (%)", "WER"),
        (axes[1], cer, "CER (%)", "CER"),
    ]:
        for idx, name in enumerate(names):
            ax.scatter(abx[idx], target[idx], s=180, color=colors[idx], edgecolor="black", linewidth=0.7)
            ax.annotate(name, (abx[idx], target[idx]), textcoords="offset points", xytext=(8, 8), fontsize=10)
        ax.set_xlabel("ABX error")
        ax.set_ylabel(ylabel)
        ax.set_title(f"ABX vs {target_name}")
        ax.grid(True, alpha=0.35)
    fig.suptitle("ABX vs ASR: currently available comparable points", fontsize=15, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "ABX is currently available in the deck only for HuBERT and WavJEPA HF. Their WER/CER tie despite different ABX.",
        ha="center",
        fontsize=10,
        color=DARK,
    )
    save(
        fig,
        "03_abx_vs_asr_scatter",
        artifacts,
        "ABX vs ASR scatter",
        "Two-panel scatter showing that HuBERT and WavJEPA HF differ on ABX while tying on CER/WER in the current eng1 10 min slice.",
        "data",
    )


def plot_multilingual_scaling(artifacts: list[PlotArtifact]) -> None:
    budgets = ["10 min", "1 h"]
    cer = np.array([24.96, 20.76])
    wer = np.array([23.48, 18.30])
    x = np.arange(len(budgets))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(x, cer, marker="o", linewidth=2.5, markersize=9, color=PRIMARY, label="CER")
    ax.plot(x, wer, marker="s", linewidth=2.5, markersize=8, color=SECONDARY, label="WER")
    for idx, value in enumerate(cer):
        ax.text(idx, value + 0.9, f"{value:.2f}", ha="center", fontsize=9)
    for idx, value in enumerate(wer):
        ax.text(idx, value - 1.8, f"{value:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(budgets)
    ax.set_ylabel("Error (%)")
    ax.set_title("Multilingual ASR-only scaling: 10 min to 1 h")
    ax.legend(loc="upper right")
    ax.set_ylim(15, 28)
    save(
        fig,
        "04_multilingual_scaling",
        artifacts,
        "Multilingual ASR scaling 10 min to 1 h",
        "Slope chart for multilingual ASR-only showing improved CER and WER when moving from 10 min to 1 h.",
        "data",
    )


def plot_multilingual_tradeoffs(artifacts: list[PlotArtifact]) -> None:
    labels = ["ASR-only\n10 min", "ASR-only\n1 h", "ASR+LID\n10 min", "LoRA ASR\n10 min"]
    cer = np.array([24.96, 20.76, 26.33, 24.95])
    wer = np.array([23.48, 18.30, 25.49, 23.66])
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.bar(x - width / 2, cer, width, color=SUCCESS, label="CER")
    ax.bar(x + width / 2, wer, width, color=WARNING, label="WER")
    for idx, value in enumerate(cer):
        ax.text(idx - width / 2, value + 0.8, f"{value:.2f}", ha="center", fontsize=9)
    for idx, value in enumerate(wer):
        ax.text(idx + width / 2, value + 0.8, f"{value:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (%)")
    ax.set_title("Multilingual trade-offs: ASR-only vs ASR+LID vs LoRA")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 30)
    save(
        fig,
        "05_multilingual_tradeoffs",
        artifacts,
        "Multilingual trade-offs",
        "Grouped bars comparing multilingual ASR-only, ASR+LID, and LoRA settings.",
        "data",
    )


def plot_runtime_comparison(artifacts: list[PlotArtifact]) -> None:
    labels = [
        "WavJEPA local\neng1 10 min",
        "WavJEPA HF\neng1 10 min",
        "LID-only\n10 min",
        "deu1 HuBERT\n10 min",
        "ASR-only\n10 min",
        "deu1 HuBERT\n1 h",
        "ASR-only\n1 h",
        "LID-only\n1 h",
        "fra1 HuBERT\n1 h",
    ]
    hours = np.array([1.62, 1.99, 4.93, 5.21, 5.93, 9.44, 11.22, 11.17, 14.74])
    colors = [SECONDARY, ACCENT, WARNING, PRIMARY, SUCCESS, PRIMARY, SUCCESS, WARNING, PRIMARY]
    order = np.argsort(hours)
    labels = [labels[idx] for idx in order]
    hours = hours[order]
    colors = [colors[idx] for idx in order]
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.barh(np.arange(len(labels)), hours, color=colors)
    for idx, value in enumerate(hours):
        ax.text(value + 0.2, idx, f"{value:.2f} h", va="center", fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Training time (hours)")
    ax.set_title("Measured training times across representative runs")
    save(
        fig,
        "06_runtime_comparison",
        artifacts,
        "Measured runtime comparison",
        "Horizontal bar chart of representative L4 training times from the runtime table.",
        "data",
    )


def plot_performance_vs_cost(artifacts: list[PlotArtifact]) -> None:
    points = [
        ("WavJEPA HF\neng1 10 min", 1.99, 24.14, SECONDARY),
        ("WavJEPA local\neng1 10 min", 1.62, 24.14, ACCENT),
        ("ASR-only\n10 min", 5.93, 23.48, SUCCESS),
        ("ASR+LID\n10 min", 5.93, 25.49, WARNING),
        ("LoRA ASR\n10 min", 5.93, 23.66, PRIMARY),
        ("ASR-only\n1 h", 11.22, 18.30, DARK),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for label, runtime, wer, color in points:
        ax.scatter(runtime, wer, s=180, color=color, edgecolor="black", linewidth=0.7)
        ax.annotate(label, (runtime, wer), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Training time (hours)")
    ax.set_ylabel("WER (%)")
    ax.set_title("Performance vs cost")
    ax.grid(True, alpha=0.35)
    ax.text(
        0.98,
        0.05,
        "Lower-left is better.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=DARK,
    )
    save(
        fig,
        "07_performance_vs_cost",
        artifacts,
        "Performance vs cost",
        "Scatter of representative runs with runtime on x-axis and WER on y-axis.",
        "data",
    )


def plot_coverage_heatmap(artifacts: list[PlotArtifact]) -> None:
    rows = ["HuBERT", "JEPA minimal", "WavJEPA HF", "WavJEPA local", "ASR+LID", "LoRA", "Small local from scratch"]
    cols = ["eng1 10m", "eng1 1h", "fra/deu", "multilingual", "ABX", "clean compare", "paper-ready"]
    status = np.array(
        [
            [2, 0, 2, 2, 2, 2, 1],
            [2, 0, 0, 0, 1, 1, 0],
            [2, 0, 0, 0, 2, 2, 1],
            [2, 0, 0, 0, 1, 1, 1],
            [2, 1, 0, 2, 0, 1, 1],
            [2, 1, 0, 2, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    cmap = matplotlib.colors.ListedColormap(["#EFEFEF", "#F4D35E", "#2A9D8F"])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(status, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    labels = {0: "planned", 1: "partial", 2: "done"}
    for i in range(status.shape[0]):
        for j in range(status.shape[1]):
            ax.text(j, i, labels[int(status[i, j])], ha="center", va="center", fontsize=8, color="#1F1F1F")
    ax.set_title("Experiment coverage map")
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor="#2A9D8F", markeredgecolor="none", label="done"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor="#F4D35E", markeredgecolor="none", label="partial"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor="#EFEFEF", markeredgecolor="#BBBBBB", label="planned"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    save(
        fig,
        "08_experiment_coverage_heatmap",
        artifacts,
        "Experiment coverage heatmap",
        "Heatmap summarizing what is done, partial, or planned across model families and evaluation axes.",
        "mixed",
    )


def plot_multilingual_queue_timeline(artifacts: list[PlotArtifact]) -> None:
    starts = [
        datetime.fromisoformat("2026-03-21 23:35:52"),
        datetime.fromisoformat("2026-03-22 05:43:19"),
        datetime.fromisoformat("2026-03-22 20:09:22"),
        datetime.fromisoformat("2026-03-23 01:13:12"),
    ]
    ends = [
        datetime.fromisoformat("2026-03-22 05:43:19"),
        datetime.fromisoformat("2026-03-22 17:19:59"),
        datetime.fromisoformat("2026-03-23 01:13:12"),
        datetime.fromisoformat("2026-03-23 12:42:41"),
    ]
    labels = ["ASR-only 10 min", "ASR-only 1 h", "LID-only 10 min", "LID-only 1 h"]
    colors = [SUCCESS, DARK, WARNING, ACCENT]
    base = starts[0]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    y = np.arange(len(labels))
    for idx, (start, end, label, color) in enumerate(zip(starts, ends, labels, colors)):
        left = (start - base).total_seconds() / 3600.0
        width = (end - start).total_seconds() / 3600.0
        ax.barh(idx, width, left=left, color=color, height=0.58)
        ax.text(left + width + 0.15, idx, f"{width:.2f} h", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Hours since queue start")
    ax.set_title("Multilingual queue timeline")
    ax.invert_yaxis()
    save(
        fig,
        "09_multilingual_queue_timeline",
        artifacts,
        "Multilingual queue timeline",
        "Gantt-like timeline of completed multilingual queue stages from the runtime log.",
        "data",
    )


def plot_scope_roadmap(artifacts: list[PlotArtifact]) -> None:
    categories = ["Delivered now", "Strong next step", "Optional extension"]
    items = {
        "Delivered now": ["HuBERT baseline", "WavJEPA HF", "Local ckpt A/B", "ABX probe", "Multilingual branches"],
        "Strong next step": ["Small from-scratch JEPA", "Second SSL family", "Per-layer ABX/ASR", "Clean A/B checkpoints"],
        "Optional extension": ["SpidR / SpidR-Adapt", "Full paper matrix", "SER cross-task bridge"],
    }
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    xs = [0, 1, 2]
    palette = [SUCCESS, WARNING, SECONDARY]
    for idx, category in enumerate(categories):
        ax.scatter([xs[idx]] * len(items[category]), np.arange(len(items[category]))[::-1], s=240, color=palette[idx], alpha=0.9)
        for item_idx, item in enumerate(items[category][::-1]):
            ax.text(xs[idx] + 0.08, item_idx, item, va="center", fontsize=10)
        ax.text(xs[idx], len(items[category]) + 0.2, category, ha="center", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.35, 2.95)
    ax.set_ylim(-0.7, 5.1)
    ax.axis("off")
    ax.set_title("Scope vs publishable next-step roadmap")
    save(
        fig,
        "10_scope_publishable_roadmap",
        artifacts,
        "Scope vs publishable roadmap",
        "Three-column roadmap separating delivered content, strong next steps, and optional extensions.",
        "schematic",
    )


def plot_hf_local_drift(artifacts: list[PlotArtifact]) -> None:
    labels = ["Embed cosine\nmean", "Embed cosine\nstd", "Norm ratio\nLocal/HF", "CTC row cosine\nmean"]
    values = np.array([-0.0082, 0.0317, 1.5940, 0.0321])
    colors = [SECONDARY, PRIMARY, ACCENT, SUCCESS]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    bars = ax.bar(np.arange(len(labels)), values, color=colors)
    for bar, value in zip(bars, values):
        offset = 0.04 if value >= 0 else -0.08
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.4f}", ha="center", fontsize=9)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("HF vs local checkpoint drift summary")
    ax.text(
        0.02,
        0.94,
        "Large norm drift with near-zero average cosine alignment: geometry changes a lot even when CER/WER stay close.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=DARK,
    )
    save(
        fig,
        "11_hf_local_drift_summary",
        artifacts,
        "HF vs local drift summary",
        "Compact drift summary from the 311-utterance analysis slide.",
        "data",
    )


def plot_protocol_fidelity(artifacts: list[PlotArtifact]) -> None:
    points = [
        ("ESPnet + S3PRL\ncurrent stack", 9.4, 3.0, PRIMARY),
        ("speech_encoder\nsame pipeline", 5.0, 3.6, WARNING),
        ("cached discrete units\nredesigned task", 3.5, 8.0, SECONDARY),
        ("off-protocol speed hack", 1.8, 9.2, DANGER),
    ]
    fig, ax = plt.subplots(figsize=(8.9, 5.4))
    for label, x_value, y_value, color in points:
        ax.scatter(x_value, y_value, s=220, color=color, edgecolor="black", linewidth=0.7)
        ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Expected speedup potential")
    ax.set_ylabel("Protocol fidelity to MLSUPERB")
    ax.set_title("Protocol fidelity vs speedup potential")
    ax.grid(True, alpha=0.35)
    ax.text(0.98, 0.04, "Schematic positioning for discussion.", transform=ax.transAxes, ha="right", fontsize=9, color=DARK)
    save(
        fig,
        "12_protocol_fidelity_vs_speedup",
        artifacts,
        "Protocol fidelity vs speedup",
        "Schematic scatter for the speech_encoder discussion: strict protocol fidelity is not where the largest speedups usually live.",
        "schematic",
    )


def _read_layer_proxy_csv(path: Path) -> tuple[list[int], list[float]]:
    layers: list[int] = []
    scores: list[float] = []
    if not path.is_file():
        return layers, scores
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("layer", "")
            if not name.startswith("layer_"):
                continue
            try:
                idx = int(name.replace("layer_", ""))
            except ValueError:
                continue
            raw = (row.get("proxy_score") or "").strip()
            if not raw:
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if np.isnan(val):
                continue
            layers.append(idx)
            scores.append(val)
    order = np.argsort(layers)
    return [layers[i] for i in order], [scores[i] for i in order]


def plot_asr_vs_ser_universality(artifacts: list[PlotArtifact]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    points = [
        ("HuBERT", 24.14, 0.7305, PRIMARY),
        ("WavJEPA", 24.14, 0.7448, SECONDARY),
        ("wav2vec2", None, 0.5768, WARNING),
        ("JEPA minimal", 44.83, None, DANGER),
    ]
    for name, wer, uar, color in points:
        if wer is not None and uar is not None:
            ax.scatter(wer, uar, s=200, color=color, edgecolor="black", linewidth=0.8, zorder=3)
            ax.annotate(name, (wer, uar), textcoords="offset points", xytext=(8, 6), fontsize=10)
        elif wer is not None:
            ax.scatter(wer, 0.52, s=200, facecolors="none", edgecolors=color, linewidth=2.2, linestyle="--", zorder=3)
            ax.annotate(f"{name}\n(SER TBD)", (wer, 0.52), textcoords="offset points", xytext=(8, -18), fontsize=9, color=DARK)
        else:
            ax.scatter(38.0, uar, s=200, facecolors="none", edgecolors=color, linewidth=2.0, linestyle=":", zorder=3)
            ax.annotate(f"{name}\n(no eng1 ASR row)", (38.0, uar), textcoords="offset points", xytext=(8, 4), fontsize=8, color=DARK)
    ax.set_xlabel("ML-SUPERB eng1 10 min WER (%) — lower is better")
    ax.set_ylabel("SER UAR on RAVDESS (full train) — higher is better")
    ax.set_title("ASR vs paralinguistic universality (Bruny SER + Janis ASR)")
    ax.grid(True, alpha=0.35)
    ax.text(
        0.03,
        0.97,
        "Ideal corner: low WER + high UAR. JEPA minimal SER is pending (see ser_ravdess_frozen_frontend.py).",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=DARK,
    )
    save(
        fig,
        "13_asr_vs_ser_universality",
        artifacts,
        "ASR vs SER universality map",
        "Scatter combining Janis monolingual WER with Bruny RAVDESS UAR (full regime).",
        "data",
    )


def plot_ranking_inversion(artifacts: list[PlotArtifact]) -> None:
    models = ["HuBERT", "WavJEPA", "wav2vec2", "JEPA minimal"]
    tasks = ["ASR\n(1/WER)", "ABX\n(1/err)", "SER\nRAVDESS", "SER\nIEMOCAP"]
    raw = np.array(
        [
            [1.0 / 24.14, 1.0 / 0.5216, 0.7305, 0.5521],
            [1.0 / 24.14, 1.0 / 0.5895, 0.7448, 0.6013],
            [np.nan, np.nan, 0.5768, 0.4218],
            [1.0 / 44.83, np.nan, np.nan, np.nan],
        ]
    )
    ranks = np.full_like(raw, np.nan, dtype=float)
    for j in range(raw.shape[1]):
        idxs = [i for i in range(raw.shape[0]) if not np.isnan(raw[i, j])]
        if len(idxs) < 2:
            continue
        vals = np.array([raw[i, j] for i in idxs], dtype=float)
        sorted_idx = np.argsort(-vals)
        r_local = np.empty(len(vals), dtype=int)
        rk = 1
        for k in range(len(sorted_idx)):
            vi = sorted_idx[k]
            if k > 0 and vals[vi] < vals[sorted_idx[k - 1]]:
                rk = k + 1
            r_local[vi] = rk
        for t, glob_row in enumerate(idxs):
            ranks[glob_row, j] = r_local[t]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    mx = int(np.nanmax(ranks)) if np.isfinite(np.nanmax(ranks)) else 1
    cmap = matplotlib.colors.ListedColormap(["#F4F1DE", "#E9C46A", "#2A9D8F", "#264653"][: max(mx, 1)])
    bounds = np.concatenate(([0.5], np.arange(1.5, mx + 1.5, 1.0)))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(ranks, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            val = ranks[i, j]
            txt = "—" if np.isnan(val) else str(int(val))
            ax.text(j, i, txt, ha="center", va="center", fontsize=11, color="#1a1a1a")
    ax.set_title("Per-task ranking (1 = best among models with a score)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(1, mx + 1))
    cbar.set_ticklabels([str(i) for i in range(1, mx + 1)])
    save(
        fig,
        "14_ranking_inversion_across_tasks",
        artifacts,
        "Ranking inversion across tasks",
        "Rank heatmap: ASR uses inverse WER; ABX uses inverse error from the deck; SER from report.tex appendix.",
        "data",
    )


def plot_layer_phonetic_proxy(artifacts: list[PlotArtifact]) -> None:
    lx, ly = _read_layer_proxy_csv(Path("/root/snlp/refs/plots/layer_proxy_hubert.csv"))
    wx, wy = _read_layer_proxy_csv(Path("/root/snlp/refs/plots/layer_proxy_wavjepa.csv"))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    if lx:
        ax.plot(lx, ly, marker="o", linewidth=2.0, color=PRIMARY, label="HuBERT (S3PRL layers)")
    if wx:
        ax.plot(wx, wy, marker="s", linewidth=2.0, color=SECONDARY, label="WavJEPA HF encoder layers")
    ax.set_xlabel("Layer index (upstream / encoder depth)")
    ax.set_ylabel("Phonetic separability proxy (between / within class, cosine)")
    ax.set_title("Layer-wise phonetic proxy on dev 311 utts (not official fastabx)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.35)
    ax.text(
        0.02,
        0.02,
        "Proxy from mean-pooled embeddings + first-char pseudo labels; WavJEPA activations NaN-sanitized.",
        transform=ax.transAxes,
        fontsize=8,
        color=DARK,
        va="bottom",
    )
    save(
        fig,
        "15_layer_phonetic_proxy",
        artifacts,
        "Layer-wise phonetic proxy",
        "Curve from layer_phonetic_proxy.py on exp/abx_layers/*_full (complements Vadim CTC weights + Bruny SER depth).",
        "proxy",
    )


def write_index(artifacts: Sequence[PlotArtifact]) -> None:
    readme = OUTPUT_DIR / "README.md"
    lines = [
        "# Presentation plots",
        "",
        "Generated with `python scripts/generate_presentation_plots.py`.",
        "",
        "These figures are split between:",
        '- `data`: built from canonical values already present in `refs/` and the Beamer deck.',
        '- `mixed`: a compact synthesis figure using current project status.',
        '- `schematic`: explanatory figures intended for presentation framing rather than raw benchmark reporting.',
        '- `proxy`: layer curves that are not official fastabx but help bridge to Vadim/Bruny depth analyses.',
        "",
        "## Files",
        "",
    ]
    for artifact in artifacts:
        stem = artifact.filename.replace(".png", "")
        lines.extend(
            [
                f"### `{artifact.filename}`",
                f"- Title: {artifact.title}",
                f"- Type: {artifact.kind}",
                f"- Description: {artifact.description}",
                f"- Also exported as: `{stem}.pdf`",
                "",
            ]
        )
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    artifacts: list[PlotArtifact] = []
    plot_monolingual_frontend_bars(artifacts)
    plot_delta_vs_hubert(artifacts)
    plot_abx_vs_asr(artifacts)
    plot_multilingual_scaling(artifacts)
    plot_multilingual_tradeoffs(artifacts)
    plot_runtime_comparison(artifacts)
    plot_performance_vs_cost(artifacts)
    plot_coverage_heatmap(artifacts)
    plot_multilingual_queue_timeline(artifacts)
    plot_scope_roadmap(artifacts)
    plot_hf_local_drift(artifacts)
    plot_protocol_fidelity(artifacts)
    plot_asr_vs_ser_universality(artifacts)
    plot_ranking_inversion(artifacts)
    plot_layer_phonetic_proxy(artifacts)
    write_index(artifacts)
    print(f"Generated {len(artifacts)} plot sets in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
