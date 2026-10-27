"""Plot the CPU-Qwen blend's prompt-grouped OOF confusion matrix and its change vs the CPU baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ntl.constants import LABELS  # noqa: E402

# Reference palette: sequential blue 100->700, diverging blue<->red around a gray midpoint.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#e34948", "#f0efec", "#256abf"]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
FAMILY_STARTS = [5, 9, 14, 15, 17]  # GPT | Gemini | Claude | Qwen | GLM | Kimi


def row_shares(truth, predictions):
    import numpy as np

    index = {label: position for position, label in enumerate(LABELS)}
    counts = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for true, predicted in zip(truth, predictions, strict=True):
        counts[index[true], index[predicted]] += 1
    return counts, counts / counts.sum(axis=1, keepdims=True)


def frame_axes(ax, title: str, note: str) -> None:
    import numpy as np

    ax.set_xticks(range(len(LABELS)), labels=LABELS, rotation=60, ha="right", fontsize=9, color=INK)
    ax.set_yticks(range(len(LABELS)), labels=LABELS, fontsize=9, color=INK)
    ax.set_xlabel("Predicted model", fontsize=12, labelpad=12, color=INK)
    ax.set_ylabel("True model", fontsize=12, labelpad=12, color=INK)
    ax.set_title(title, fontsize=15, loc="left", pad=34, color=INK)
    # A muted reading note between the title and the grid, clear of the rotated tick labels.
    ax.text(0, 1.012, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color=MUTED)
    ax.set_xticks(np.arange(len(LABELS) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(LABELS) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)  # 2px surface gap between cells
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for start in FAMILY_STARTS:  # family blocks: where most confusion lives
        ax.axhline(start - 0.5, color=MUTED, linewidth=1.2)
        ax.axvline(start - 0.5, color=MUTED, linewidth=1.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=ROOT / "artifacts/qwen-blend/oof.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/plots")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.ticker import PercentFormatter

    oof = pd.read_parquet(args.oof)
    blend_counts, blend = row_shares(oof["true_model"], oof["prediction::blend"])
    _, cpu = row_shares(oof["true_model"], oof["prediction::recorded_reference"])
    balanced = {name: float(np.diag(values).mean()) for name, values in (("blend", blend), ("cpu", cpu))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = f"Five prompt-grouped folds · {blend_counts.sum():,} out-of-fold passages"

    # 1. Blend confusion matrix, row-normalized. Zero cells stay surface-colored.
    fig, ax = plt.subplots(figsize=(16, 14), layout="constrained", facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    sequential = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    heatmap = ax.imshow(np.ma.masked_equal(blend, 0), cmap=sequential, vmin=0, vmax=1)
    frame_axes(ax, f"CPU + Qwen blend (w = 0.25) — confusion matrix\n{common} · "
                   f"balanced accuracy {balanced['blend']:.1%}",
               "Each row sums to 100%. Labels: diagonal and confusions of 5% or more. "
               "Lines separate model families.")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            value = blend[row, column]
            if row == column or value >= 0.05:  # selective labels: diagonal and confusions >= 5%
                ax.text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=8,
                        color=SURFACE if value > 0.45 else INK, fontweight="bold" if row == column else "normal")
    colorbar = fig.colorbar(heatmap, ax=ax, shrink=0.75, pad=0.025)
    colorbar.set_label("Share of the true model's passages", color=INK)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    colorbar.outline.set_visible(False)
    fig.savefig(args.output_dir / "blend_confusion_matrix.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 2. Change vs the submitted CPU baseline, in percentage points of each true model's passages.
    change = blend - cpu
    limit = max(0.15, float(np.abs(change).max()))
    fig, ax = plt.subplots(figsize=(16, 14), layout="constrained", facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    diverging = LinearSegmentedColormap.from_list("red-gray-blue", DIVERGING)
    heatmap = ax.imshow(change, cmap=diverging, norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit))
    frame_axes(ax, "Change from the CPU baseline to the blend\n"
                   f"{common} · balanced accuracy {balanced['cpu']:.1%} → {balanced['blend']:.1%}",
               "Blue: more passages in that cell; red: fewer. Diagonal blue and off-diagonal red are improvements. "
               "Labels: diagonal and changes of 3 points or more.")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            value = change[row, column]
            if row == column or abs(value) >= 0.03:
                ax.text(column, row, f"{value * 100:+.0f}", ha="center", va="center", fontsize=8,
                        color=SURFACE if abs(value) > limit * 0.6 else INK,
                        fontweight="bold" if row == column else "normal")
    colorbar = fig.colorbar(heatmap, ax=ax, shrink=0.75, pad=0.025)
    colorbar.set_label("Change in share of the true model's passages (percentage points)", color=INK)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    colorbar.outline.set_visible(False)
    fig.savefig(args.output_dir / "blend_confusion_change.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    for name in ("blend_confusion_matrix.png", "blend_confusion_change.png"):
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
