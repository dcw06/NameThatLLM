"""Error inspection on the incumbent's saved out-of-fold probabilities; nothing is fitted.

For each target confusion it records, per passage, the true label's rank, the probability gap
from the predicted label, the top probability and the passage length (the marker study's
tokenizer). It compares error margins with the margins of correct predictions in the same
classes, counts prompts whose two passages are swapped with each other, and simulates scaling
one class's probability by a constant factor. That simulation uses every row's label, so it is
an in-sample description of what a score adjustment could reach, not a validated result.

The written report (with passage excerpts) stays in the gitignored output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from . import claude_glm, ensemble, gemini_markers
from .constants import LABELS
from .data import read_train

DIRECTIONS = {
    "Fable ↔ GLM Flash": [("claude-fable-5", "glm-5.3-flash"), ("glm-5.3-flash", "claude-fable-5")],
    "GLM ↔ GLM Flash": [("glm-5.3", "glm-5.3-flash"), ("glm-5.3-flash", "glm-5.3")],
    "GLM → Claude Opus 5": [("glm-5.3", "claude-opus-5")],
}
FACTORS = (1.1, 1.25, 1.5, 2.0)
NARROW = 0.05


def passage_table(train: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    """One row per passage: true and predicted labels, the true label's rank and probability gap."""
    truth = ensemble._targets(train)
    rows = np.arange(len(train))
    order = np.argsort(-probabilities, axis=1, kind="stable")
    true_probability = probabilities[rows, truth]
    predicted = order[:, 0]
    return pd.DataFrame({
        "id": train["id"].to_numpy(), "prompt_id": train["prompt_id"].to_numpy(),
        "true": np.asarray(LABELS)[truth], "predicted": np.asarray(LABELS)[predicted],
        "second": np.asarray(LABELS)[order[:, 1]], "third": np.asarray(LABELS)[order[:, 2]],
        "p_true": true_probability, "p_predicted": probabilities[rows, predicted],
        "p_second": probabilities[rows, order[:, 1]], "p_third": probabilities[rows, order[:, 2]],
        # 1 when correct; otherwise one more than the number of labels scored above the true one.
        "true_rank": 1 + (probabilities > true_probability[:, None]).sum(axis=1),
        "gap": probabilities[rows, predicted] - true_probability,
        "margin": probabilities[rows, order[:, 0]] - probabilities[rows, order[:, 1]],
        "words": [len(gemini_markers.TOKEN_RE.findall(text)) for text in train["text"]],
    })


def _quartiles(values: pd.Series) -> list[float]:
    return [float(values.quantile(q)) for q in (0.25, 0.5, 0.75)] if len(values) else []


def direction_summary(table: pd.DataFrame, true: str, predicted: str) -> dict:
    errors = table[(table["true"] == true) & (table["predicted"] == predicted)]
    correct = table[(table["true"] == true) & (table["predicted"] == true)]
    ranks = errors["true_rank"]
    return {
        "errors": int(len(errors)),
        "true_rank": {"2": int((ranks == 2).sum()), "3": int((ranks == 3).sum()), "4+": int((ranks >= 4).sum())},
        "gap_quartiles": _quartiles(errors["gap"]),
        f"gap_below_{NARROW}": int((errors["gap"] < NARROW).sum()),
        "predicted_probability_quartiles": _quartiles(errors["p_predicted"]),
        "true_probability_quartiles": _quartiles(errors["p_true"]),
        "words_quartiles": _quartiles(errors["words"]),
        "correct_words_quartiles": _quartiles(correct["words"]),
        "correct_margin_quartiles": _quartiles(correct["margin"]),
        f"correct_margin_below_{NARROW}": int((correct["margin"] < NARROW).sum()), "correct": int(len(correct)),
    }


def mutual_swaps(table: pd.DataFrame, first: str, second: str) -> dict:
    """Prompts in which the `first` passage is called `second` and the `second` passage `first`."""
    one = set(table.loc[(table["true"] == first) & (table["predicted"] == second), "prompt_id"])
    other = set(table.loc[(table["true"] == second) & (table["predicted"] == first), "prompt_id"])
    return {"prompts_both_ways": int(len(one & other)), "prompts": sorted(one & other)}


def boost(truth: np.ndarray, probabilities: np.ndarray, label: str, factor: float) -> dict:
    """Scale one class's probability by `factor` (in sample): balanced-accuracy change and moves by class."""
    adjusted = np.array(probabilities, copy=True)
    adjusted[:, LABELS.index(label)] *= factor
    first, second = probabilities.argmax(axis=1), adjusted.argmax(axis=1)
    gained, lost = (first != truth) & (second == truth), (first == truth) & (second != truth)
    return {"factor": factor,
            "balanced_accuracy_change": float(balanced_accuracy_score(truth, second)
                                              - balanced_accuracy_score(truth, first)),
            "gained": int(gained.sum()), "lost": int(lost.sum()),
            "by_class": {name: {"gained": int(gained[truth == index].sum()), "lost": int(lost[truth == index].sum())}
                         for index, name in enumerate(LABELS) if gained[truth == index].any() or lost[truth == index].any()}}


def examples(table: pd.DataFrame, true: str, predicted: str) -> pd.DataFrame:
    """The narrowest, the median and the most confident error of one direction."""
    errors = table[(table["true"] == true) & (table["predicted"] == predicted)].sort_values("gap")
    if errors.empty:
        return errors
    picks = sorted({0, (len(errors) - 1) // 2, len(errors) - 1})
    return errors.iloc[picks].assign(kind=[{0: "narrowest", len(errors) - 1: "most confident"}.get(i, "median")
                                           for i in picks])


def inspect(train: pd.DataFrame, probabilities: np.ndarray) -> tuple[pd.DataFrame, dict]:
    table = passage_table(train, probabilities)
    truth = ensemble._targets(train)
    errors = table[table["true"] != table["predicted"]]
    summary = {
        "balanced_accuracy": float(balanced_accuracy_score(truth, probabilities.argmax(axis=1))),
        "all_errors": {"errors": int(len(errors)), "true_rank_2": int((errors["true_rank"] == 2).sum()),
                       "gap_quartiles": _quartiles(errors["gap"])},
        "groups": {group: {f"{true} -> {predicted}": direction_summary(table, true, predicted)
                           for true, predicted in pairs} for group, pairs in DIRECTIONS.items()},
        "mutual_swaps": {f"{first} <-> {second}": mutual_swaps(table, first, second)
                         for first, second in (("claude-fable-5", "glm-5.3-flash"), ("glm-5.3", "glm-5.3-flash"))},
        "boost_in_sample": {label: [boost(truth, probabilities, label, factor) for factor in FACTORS]
                            for label in ("claude-fable-5", "glm-5.3", "glm-5.3-flash")},
    }
    return table, summary


def _report(table: pd.DataFrame, summary: dict, texts: pd.Series) -> str:
    lines = ["# Fable / GLM error inspection: examples", "",
             "Generated by `ntl.error_inspection`. Competition text: keep this file out of Git.", ""]
    for group, pairs in DIRECTIONS.items():
        lines += [f"## {group}", ""]
        for true, predicted in pairs:
            lines += [f"### True {true}, predicted {predicted}", ""]
            for _, row in examples(table, true, predicted).iterrows():
                lines += [f"- **{row['kind']}** `{row['id']}` ({row['prompt_id']}, {row['words']} words): "
                          f"true rank {row['true_rank']}, p({true}) = {row['p_true']:.3f}, "
                          f"p({predicted}) = {row['p_predicted']:.3f}, gap {row['gap']:.3f}; "
                          f"second {row['second']} ({row['p_second']:.3f}), third {row['third']} ({row['p_third']:.3f})",
                          f"  > {texts[row['id']][:300].strip()}…", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="the locked incumbent (ntl.style_adapt lock)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    train = read_train(args.train)
    reference = json.loads(args.reference.read_text())
    locked = np.load(args.reference.with_suffix(".npz"), allow_pickle=False)
    if (not np.array_equal(locked["ids"], train["id"].to_numpy().astype(str))
            or reference["groups_sha256"] != claude_glm.groups_sha256(train)):
        raise ValueError("The locked incumbent is for other rows")
    table, summary = inspect(train, locked["probabilities"])
    if summary["balanced_accuracy"] != reference["balanced_accuracy"]:
        raise ValueError("The locked probabilities do not reproduce the incumbent's balanced accuracy")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = {true for pairs in DIRECTIONS.values() for true, _ in pairs}
    table[table["true"].isin(targets)].to_csv(output / "passages.csv", index=False)
    claude_glm._json(output / "summary.json", summary)
    (output / "report.md").write_text(_report(table, summary, train.set_index("id")["text"]))
    print(json.dumps({group: {key: {field: value[field] for field in ("errors", "true_rank", "gap_quartiles",
                                                                     f"gap_below_{NARROW}")}
                              for key, value in directions.items()}
                      for group, directions in summary["groups"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    from ntl.error_inspection import main as imported_main

    raise SystemExit(imported_main())
