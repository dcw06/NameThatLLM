"""Plot the saved prompt-grouped validation confusion matrix without retraining."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("artifacts/kaggle-version-1/cpu-cv/confusion_matrix.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/plots"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with args.input.open(newline="") as handle:
        reader = csv.reader(handle)
        labels = next(reader)[1:]
        records = list(reader)
    if [row[0] for row in records] != labels:
        raise ValueError("Row and column labels must match in the same order")
    counts = np.array([[int(value) for value in row[1:]] for row in records])
    if counts.shape != (len(labels), len(labels)) or (counts < 0).any():
        raise ValueError("Expected a square nonnegative confusion matrix")
    support = counts.sum(axis=1)
    if (support == 0).any():
        raise ValueError("Every true model must have validation examples")
    normalized = counts / support[:, None]
    balanced_accuracy = np.diag(normalized).mean()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for percentages in (False, True):
        fig, ax = plt.subplots(figsize=(16, 14), layout="constrained")
        values = normalized if percentages else counts
        maximum = 1.0 if percentages else int(support.max())
        heatmap = ax.imshow(values, cmap="Blues", vmin=0, vmax=maximum)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=60, ha="right", fontsize=9)
        ax.set_yticks(range(len(labels)), labels=labels, fontsize=9)
        ax.set_xlabel("Predicted model", fontsize=12, labelpad=12)
        ax.set_ylabel("True model", fontsize=12, labelpad=12)
        ax.set_title(
            "CPU baseline — Kaggle validation confusion matrix\n"
            f"Five prompt-grouped folds · {counts.sum():,} passages · "
            f"balanced accuracy {balanced_accuracy:.2%}",
            fontsize=15, loc="left", pad=20,
        )
        ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = values[row, column]
                annotation = f"{value:.0%}" if percentages else str(int(value))
                ax.text(
                    column, row, annotation,
                    ha="center", va="center", fontsize=8,
                    color="white" if value > maximum * 0.55 else "#203047",
                    fontweight="bold" if row == column else "normal",
                )
        colorbar = fig.colorbar(heatmap, ax=ax, shrink=0.75, pad=0.025)
        colorbar.set_label("Share of true model's passages" if percentages else "Passage count")
        if percentages:
            from matplotlib.ticker import PercentFormatter

            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
        suffix = "row_percent" if percentages else "counts"
        destination = args.output_dir / f"confusion_matrix_{suffix}"
        fig.savefig(destination.with_suffix(".png"), dpi=150, facecolor="white")
        fig.savefig(destination.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)
        print(destination.with_suffix(".png"))


if __name__ == "__main__":
    main()
