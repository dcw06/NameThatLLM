"""GPT specialist (GPT-5.5, GPT-5.6 Luna / Sol / Terra, GPT-6 Astra), integrated into the complete incumbent.

The incumbent is the validated system: the CPU + frozen-Qwen blend whose Claude-GLM mass the
Qwen + StyleDistance specialist redistributes at the s its inner folds choose. A specialist
trained on the true GPT rows of each training partition, with the Claude-GLM specialist's recipe,
redistributes the incumbent's GPT mass: p'_k = (1 - w) p_k + w M q_k on the five GPT classes,
with M their total; the other 13 probabilities are unchanged and w = 0 is the incumbent exactly.

Inner prompt folds of each outer fold rebuild the complete incumbent and choose one of nine
options by mean inner 18-class balanced accuracy: the incumbent, or Qwen or Qwen + StyleDistance
features at w in {1/4, 1/2, 3/4, 1}. Ties keep the incumbent, then the smaller w, then Qwen.
The chosen option is fit on the outer training partition and applied to the locked incumbent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from . import claude_glm, ensemble, gemini_pair, neural
from .constants import LABELS
from .data import fingerprint_frame, read_train
from .lora import AdapterConfig, validate_plan
from .selection import first_best, snap_tie

GPT = ("gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra")
GPT_INDEX = np.array([LABELS.index(label) for label in GPT])
FEATURE_SETS = ("qwen", "qwen+styledistance")
WEIGHTS = (0.25, 0.5, 0.75, 1.0)
REFERENCE = "incumbent"
GAIN = 0.005


def options(feature_sets: tuple[str, ...] = FEATURE_SETS) -> list[tuple[str, float] | None]:
    """In tie order: the incumbent, then the smaller w, then the feature sets in order."""
    return [None] + [(feature_set, weight) for weight in WEIGHTS for feature_set in feature_sets]


def option_name(option: tuple[str, float] | None) -> str:
    return REFERENCE if option is None else f"{option[0]}@w={option[1]:g}"


def specialist(blocks: dict[str, np.ndarray], feature_set: str, truth: np.ndarray, fit: np.ndarray, rows: np.ndarray,
               frozen: neural.FrozenConfig) -> np.ndarray:
    """Five GPT probabilities for `rows`, fit on the true GPT rows among `fit`."""
    return claude_glm.specialist_probabilities(blocks, feature_set, truth, fit, rows, frozen, GPT_INDEX)


def integrated(incumbent: np.ndarray, gpt: dict[str, np.ndarray], option: tuple[str, float] | None) -> np.ndarray:
    if option is None:
        return incumbent
    feature_set, weight = option
    return claude_glm.redistribute(incumbent, gpt[feature_set], weight, GPT_INDEX)


def _ba(truth: np.ndarray, probabilities: np.ndarray) -> float:
    return float(balanced_accuracy_score(truth, probabilities.argmax(axis=1)))


def _metrics(truth: np.ndarray, predictions: np.ndarray, folds: np.ndarray) -> dict:
    result = ensemble._system_metrics(truth, predictions, folds)
    precision, _, _, _ = precision_recall_fscore_support(truth, predictions, labels=np.arange(len(LABELS)),
                                                         zero_division=0)
    result["precision"] = {label: float(precision[index]) for index, label in enumerate(LABELS)}
    return result


def gpt_confusion(truth: np.ndarray, predictions: np.ndarray) -> dict:
    """True GPT rows by predicted GPT class, with every other prediction as "other"."""
    result = {}
    for index, label in zip(GPT_INDEX, GPT):
        predicted = predictions[truth == index]
        counts = {other: int((predicted == column).sum()) for column, other in zip(GPT_INDEX, GPT)}
        counts["other"] = int(len(predicted) - sum(counts.values()))
        result[label] = counts
    return result


def routing(truth: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict:
    """Passages moving into or out of the GPT classes, by true class."""
    inside = lambda values: np.isin(values, GPT_INDEX)  # noqa: E731
    return {label: {"gpt_to_other": int(((truth == index) & inside(first) & ~inside(second)).sum()),
                    "other_to_gpt": int(((truth == index) & ~inside(first) & inside(second)).sum())}
            for index, label in enumerate(LABELS)}


def compare(truth: np.ndarray, folds: np.ndarray, groups: np.ndarray, first: np.ndarray, second: np.ndarray,
            samples: int, seed: int) -> dict:
    """Second against first on all rows: the paired bootstrap, fold changes and gains and losses by class."""
    systems = {name: _metrics(truth, values, folds) for name, values in (("first", first), ("second", second))}
    paired = ensemble.prompt_bootstrap(truth, first, second, groups, samples, seed)
    gained, lost = (first != truth) & (second == truth), (first == truth) & (second != truth)
    return {"delta": snap_tie(systems["second"]["balanced_accuracy"] - systems["first"]["balanced_accuracy"]),
            "paired": paired,
            "fold_changes": [snap_tie(after - before) for after, before in
                             zip(systems["second"]["fold_balanced_accuracy"], systems["first"]["fold_balanced_accuracy"])],
            "gained": int(gained.sum()), "lost": int(lost.sum()),
            "by_class": {label: {"gained": int(gained[truth == index].sum()), "lost": int(lost[truth == index].sum())}
                         for index, label in enumerate(LABELS)}}


def evaluate(train: pd.DataFrame, blocks: dict[str, np.ndarray], plan: pd.DataFrame, current_oof: pd.DataFrame,
             current_cache: pd.DataFrame, locked: np.ndarray, output_dir: str | Path, *, gate: str = "prompt",
             frozen: neural.FrozenConfig | None = None, samples: int = 10000, seed: int = 2026,
             provenance: dict | None = None) -> dict:
    frozen = frozen or neural.FrozenConfig()
    plan = validate_plan(train, plan, AdapterConfig())
    output = Path(output_dir)
    if (output / "comparison.json").exists():
        raise ValueError("Use a new output directory; an existing comparison must not be overwritten")
    if set(current_cache["groups_sha256"]) != {claude_glm.groups_sha256(train)}:
        raise ValueError("Cached current-system inner predictions were built under a different row grouping")
    truth, groups = ensemble._targets(train), train["prompt_id"].to_numpy()
    folds = plan["outer_fold"].to_numpy()
    current = claude_glm._aligned_current(current_oof, train, folds)
    grid = options()
    nested = {feature_set: [None] + [(feature_set, weight) for weight in WEIGHTS] for feature_set in FEATURE_SETS}
    candidate = np.empty_like(locked)
    alone = {feature_set: np.empty_like(locked) for feature_set in FEATURE_SETS}
    by_option = {option_name(option): np.empty_like(locked) for option in grid}
    gpt_scores = {feature_set: np.empty((len(train), len(GPT))) for feature_set in FEATURE_SETS}
    selection = []
    for fold in range(5):
        rebuilt, s, s_scores = gemini_pair.inner_incumbents(train, blocks, plan, current_cache, fold, frozen)
        inner = []
        for _, fit, valid, base, claude in rebuilt:
            if np.isin(valid, np.flatnonzero(folds == fold)).any() or np.isin(fit, valid).any():
                raise ValueError("An inner partition overlaps its validation rows or the outer holdout")
            inner.append((valid, claude_glm.redistribute(base, claude, s),
                          {feature_set: specialist(blocks, feature_set, truth, fit, valid, frozen)
                           for feature_set in FEATURE_SETS}))
        scores = {option_name(option): [_ba(truth[valid], integrated(incumbent, gpt, option))
                                        for valid, incumbent, gpt in inner] for option in grid}
        means = {name: float(np.mean(values)) for name, values in scores.items()}
        chosen = first_best(grid, lambda option: means[option_name(option)])
        chosen_alone = {feature_set: first_best(choices, lambda option: means[option_name(option)])
                        for feature_set, choices in nested.items()}
        rows, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        claude = claude_glm.specialist_probabilities(blocks, gemini_pair.INCUMBENT, truth, fit, rows, frozen)
        difference = float(np.abs(claude_glm.redistribute(current[rows], claude, s) - locked[rows]).max())
        if difference > 1e-12 or (claude_glm.redistribute(current[rows], claude, s).argmax(axis=1)
                                  != locked[rows].argmax(axis=1)).any():
            raise ValueError(f"Fold {fold}: the rebuilt incumbent does not match its lock")
        gpt = {feature_set: specialist(blocks, feature_set, truth, fit, rows, frozen) for feature_set in FEATURE_SETS}
        for feature_set, values in gpt.items():
            gpt_scores[feature_set][rows] = values
        for option in grid:
            by_option[option_name(option)][rows] = integrated(locked[rows], gpt, option)
        candidate[rows] = by_option[option_name(chosen)][rows]
        for feature_set, option in chosen_alone.items():
            alone[feature_set][rows] = by_option[option_name(option)][rows]
        selection.append({"outer_fold": fold, "s": s, "selected": option_name(chosen),
                          "selected_by_feature_set": {key: option_name(value) for key, value in chosen_alone.items()},
                          "inner_mean_balanced_accuracy": means, "inner_balanced_accuracy": scores,
                          "s_inner_mean_balanced_accuracy": {str(key): value for key, value in s_scores.items()},
                          "incumbent_lock_max_abs_difference": difference})
    predictions = {"incumbent": locked.argmax(axis=1), "candidate": candidate.argmax(axis=1),
                   **{f"{feature_set}_alone": values.argmax(axis=1) for feature_set, values in alone.items()}}
    systems = {name: _metrics(truth, values, folds) for name, values in predictions.items()}
    report = compare(truth, folds, groups, predictions["incumbent"], predictions["candidate"], samples, seed)
    delta, lower = report["delta"], report["paired"]["delta_95_interval"][0]
    if gate == "prompt":
        passes = delta >= GAIN and lower > 0
    elif gate == "topic":
        passes = delta > 0 and sum(change >= 0 for change in report["fold_changes"]) >= 3
    else:
        raise ValueError(f"Unknown gate {gate!r}")
    true_gpt = np.isin(truth, GPT_INDEX)
    position = np.full(len(LABELS), -1)
    position[GPT_INDEX] = np.arange(len(GPT))

    def five_way(values: np.ndarray) -> float:
        return float(balanced_accuracy_score(position[truth[true_gpt]], values[true_gpt].argmax(axis=1)))

    output.mkdir(parents=True, exist_ok=True)
    frame = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"}).assign(outer_fold=folds)
    for name, values in predictions.items():
        frame[f"prediction::{name}"] = np.asarray(LABELS)[values]
    frame.to_parquet(output / "predictions.parquet", index=False)
    np.savez(output / "probabilities.npz", ids=train["id"].to_numpy().astype(str), incumbent=locked,
             candidate=candidate, **{f"gpt_{feature_set.replace('+', '_')}": values
                                     for feature_set, values in gpt_scores.items()})
    result = {
        "incumbent": gemini_pair.INCUMBENT, "gpt": list(GPT), "feature_sets": list(FEATURE_SETS),
        "weights": list(WEIGHTS), "options": [option_name(option) for option in grid], "selection": selection,
        "systems": systems,
        "promotion": {"gate": gate, "delta": delta, "interval": report["paired"]["delta_95_interval"],
                      "fold_changes": report["fold_changes"],
                      "nonnegative_folds": int(sum(change >= 0 for change in report["fold_changes"])),
                      "passes": bool(passes)},
        "candidate_vs_incumbent": report,
        "feature_sets_alone": {feature_set: compare(truth, folds, groups, predictions["incumbent"],
                                                    predictions[f"{feature_set}_alone"], samples, seed)
                               for feature_set in FEATURE_SETS},
        "qwen+styledistance_vs_qwen": compare(truth, folds, groups, predictions["qwen_alone"],
                                              predictions["qwen+styledistance_alone"], samples, seed),
        "gpt_confusion": {name: gpt_confusion(truth, values) for name, values in predictions.items()},
        "routing": routing(truth, predictions["incumbent"], predictions["candidate"]),
        # Diagnostics, never a selection.
        "five_way_balanced_accuracy": {"incumbent_forced": five_way(locked[:, GPT_INDEX]),
                                       "candidate_forced": five_way(candidate[:, GPT_INDEX]),
                                       **{f"specialist_{feature_set}": five_way(values)
                                          for feature_set, values in gpt_scores.items()}},
        "option_balanced_accuracy": {name: _ba(truth, values) for name, values in by_option.items()},
        "groups": int(train["prompt_id"].nunique()), "groups_sha256": claude_glm.groups_sha256(train),
        "train_fingerprint": fingerprint_frame(train), "provenance": provenance or {},
    }
    claude_glm._json(output / "comparison.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--qwen-features", type=Path, required=True)
    parser.add_argument("--styledistance-features", type=Path, required=True)
    parser.add_argument("--current-oof", type=Path, required=True)
    parser.add_argument("--current-inner", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path, required=True, help="the locked incumbent (ntl.style_adapt lock)")
    parser.add_argument("--groups", type=Path, help="CSV of id, prompt_id regrouping rows (e.g. by topic)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    train = read_train(args.train)
    blocks = claude_glm._load_blocks(train, args.qwen_features, args.styledistance_features)
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    train, plan = claude_glm._grouped(train, plan, args.groups)
    reference = json.loads(args.reference.read_text())
    locked = np.load(args.reference.with_suffix(".npz"), allow_pickle=False)
    if (not np.array_equal(locked["ids"], train["id"].to_numpy().astype(str))
            or reference["groups_sha256"] != claude_glm.groups_sha256(train)):
        raise ValueError("The locked incumbent is for other rows or another grouping")
    cache = pd.concat([pd.read_parquet(path) for path in args.current_inner], ignore_index=True)
    inputs = [args.plan, args.qwen_features, args.styledistance_features, args.current_oof, args.reference,
              args.reference.with_suffix(".npz"), *args.current_inner, *([args.groups] if args.groups else [])]
    result = evaluate(train, blocks, plan, pd.read_parquet(args.current_oof), cache, locked["probabilities"],
                      args.output_dir, gate="topic" if args.groups else "prompt",
                      provenance={"input_sha256": {str(path): claude_glm._sha256(path) for path in inputs}})
    print(json.dumps({
        "balanced_accuracy": {name: round(values["balanced_accuracy"], 4) for name, values in result["systems"].items()},
        "promotion": result["promotion"],
        "selected": [{"fold": item["outer_fold"], "s": item["s"], "selected": item["selected"],
                      "by_feature_set": item["selected_by_feature_set"]} for item in result["selection"]],
        "recall": {label: {name: round(values["recall"][label], 3) for name, values in result["systems"].items()}
                   for label in GPT},
        "five_way_balanced_accuracy": result["five_way_balanced_accuracy"],
        "changes": {key: result["candidate_vs_incumbent"][key] for key in ("gained", "lost")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.gpt_specialist import main as imported_main

    raise SystemExit(imported_main())
