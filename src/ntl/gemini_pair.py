"""Gemini 3.7 / 3.8 Flash pair specialist, integrated into the complete incumbent.

The incumbent is the validated system: the current CPU + frozen-Qwen blend whose
Claude-GLM probability mass the Qwen + StyleDistance specialist redistributes at the
weight s its inner folds choose (`claude_glm.evaluate`). A binary specialist, trained
only on the true Gemini 3.7 and 3.8 Flash rows of each training partition, redistributes
the incumbent's mass on that pair: p'_k = (1 - a) p_k + a M q_k with M = p_3.7 + p_3.8.
The other 16 probabilities are unchanged, and a = 0 is the incumbent exactly.

Two configurations: the CPU baseline's character, word and style features with its
class-weighted logistic regression (C in {0.1, 1, 4}), and the cached frozen Qwen
features, standardized, with the frozen recipe's logistic regression (C in {0.005, 0.05,
0.5}). Every vocabulary and scaler is fit on training rows only. Inner prompt folds of
each outer training partition choose s first, as the incumbent does, then one option:
the unchanged incumbent, or a configuration and C at a in {0.25, 0.5, 0.75, 1}, scored on
all 18 classes. Ties keep the incumbent, then the smaller a, then CPU, then the smaller C.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from . import claude_glm, ensemble, neural
from .constants import LABELS
from .data import fingerprint_frame, read_train
from .lora import AdapterConfig, validate_plan
from .modeling import BaselineConfig, build_pipeline
from .selection import first_best

PAIR = ("gemini-3.7-flash", "gemini-3.8-flash")
PAIR_INDEX = np.array([LABELS.index(label) for label in PAIR])
GEMINI = ("gemini-3.1-pro-preview", "gemini-3.6-flash", *PAIR)
INCUMBENT = "qwen+styledistance"
CONFIGURATIONS = {"cpu": (0.1, 1.0, 4.0), "qwen": (0.005, 0.05, 0.5)}
SPECIALISTS = [(configuration, C) for configuration, grid in CONFIGURATIONS.items() for C in grid]
WEIGHTS = (0.25, 0.5, 0.75, 1.0)
REFERENCE = "incumbent"


def options() -> list[tuple[str, float, float] | None]:
    """In tie order: the incumbent, then the smaller a, then CPU before Qwen, then the smaller C."""
    return [None] + [(configuration, C, weight) for weight in WEIGHTS for configuration, C in SPECIALISTS]


def option_name(option: tuple[str, float, float] | None) -> str:
    return REFERENCE if option is None else "{}@C={:g}@a={:g}".format(*option)


def specialist_name(configuration: str, C: float) -> str:
    return f"{configuration}@C={C:g}"


def pair_probabilities(texts: np.ndarray, qwen: np.ndarray, truth: np.ndarray, fit: np.ndarray, rows: np.ndarray,
                       configuration: str, C: float, seed: int) -> np.ndarray:
    """Fit on the true-pair rows among `fit`; (3.7, 3.8) probabilities for `rows`."""
    fit = fit[np.isin(truth[fit], PAIR_INDEX)]
    targets = (truth[fit] == PAIR_INDEX[1]).astype(int)
    if set(targets) != {0, 1}:
        raise ValueError("Both Gemini versions need training rows")
    if configuration == "cpu":  # the CPU baseline's features and classifier; only C varies
        model, inputs = build_pipeline(BaselineConfig(seed=seed, C=C)), texts
    elif configuration == "qwen":  # the frozen recipe's fresh StandardScaler and logistic head
        model, inputs = neural._head(neural.FrozenConfig(C=C, seed=seed)), qwen
    else:
        raise ValueError(f"Unknown configuration {configuration!r}")
    with threadpool_limits(limits=2):
        model.fit(inputs[fit], targets)
        probabilities = model.predict_proba(inputs[rows])
    if not np.array_equal(model.classes_, [0, 1]) or not np.isfinite(probabilities).all():
        raise ValueError("Pair specialist must give finite probabilities for both versions")
    return probabilities


def integrated(incumbent: np.ndarray, pair: dict[tuple[str, float], np.ndarray],
               option: tuple[str, float, float] | None) -> np.ndarray:
    if option is None:
        return incumbent
    configuration, C, weight = option
    return claude_glm.redistribute(incumbent, pair[(configuration, C)], weight, PAIR_INDEX)


def _ba(truth: np.ndarray, probabilities: np.ndarray) -> float:
    return float(balanced_accuracy_score(truth, probabilities.argmax(axis=1)))


def _swaps(truth: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict:
    """Direct 3.7 <-> 3.8 confusions: counts under each system, corrected and introduced."""
    result = {}
    for true, other in (PAIR_INDEX, PAIR_INDEX[::-1]):
        rows = truth == true
        result[f"{LABELS[true]}->{LABELS[other]}"] = {
            "incumbent": int((first[rows] == other).sum()), "candidate": int((second[rows] == other).sum()),
            "corrected": int(((first == other) & (second == true) & rows).sum()),
            "introduced": int(((first == true) & (second == other) & rows).sum())}
    return result


def _binary(truth: np.ndarray, pair: np.ndarray, incumbent: np.ndarray, candidate: np.ndarray,
            fixed: dict[str, np.ndarray]) -> dict:
    """Balanced accuracy between the two versions on the true-pair rows."""
    rows = np.isin(truth, PAIR_INDEX)
    target = (truth[rows] == PAIR_INDEX[1]).astype(int)

    def score(values: np.ndarray) -> float:
        return float(balanced_accuracy_score(target, values[rows].argmax(axis=1)))

    return {"rows": int(rows.sum()), "specialist": score(pair),
            "incumbent_forced": score(incumbent[:, PAIR_INDEX]), "candidate_forced": score(candidate[:, PAIR_INDEX]),
            "fixed_specialists": {name: score(values) for name, values in fixed.items()}}


def evaluate(train: pd.DataFrame, blocks: dict[str, np.ndarray], plan: pd.DataFrame, current_oof: pd.DataFrame,
             current_cache: pd.DataFrame, output_dir: str | Path, *, frozen: neural.FrozenConfig | None = None,
             samples: int = 10000, seed: int = 2026, provenance: dict | None = None) -> dict:
    frozen = frozen or neural.FrozenConfig()
    plan = validate_plan(train, plan, AdapterConfig())
    output = Path(output_dir)
    if (output / "comparison.json").exists():
        raise ValueError("Use a new output directory; an existing comparison must not be overwritten")
    if set(current_cache["groups_sha256"]) != {claude_glm.groups_sha256(train)}:
        raise ValueError("Cached current-system inner predictions were built under a different row grouping")
    output.mkdir(parents=True, exist_ok=True)
    truth, prompts = ensemble._targets(train), train["prompt_id"].to_numpy()
    texts, qwen = train["text"].to_numpy(dtype=object), blocks["qwen"]
    folds = plan["outer_fold"].to_numpy()
    current = claude_glm._aligned_current(current_oof, train, folds)
    grid = options()
    incumbent, candidate = np.empty_like(current), np.empty_like(current)
    by_option = {option_name(option): np.empty_like(current) for option in grid}
    fixed = {specialist_name(*key): np.empty((len(train), 2)) for key in SPECIALISTS}
    reported = np.empty((len(train), 2))
    selection = []
    for fold in range(5):
        column = plan[f"inner_{fold}"].to_numpy()
        inner = []
        for inner_fold, (valid, inner_current) in claude_glm._inner_current(current_cache, train, plan,
                                                                            fold).items():
            fit = np.flatnonzero((folds != fold) & (column != inner_fold))
            specialist = claude_glm.specialist_probabilities(blocks, INCUMBENT, truth, fit, valid, frozen)
            pair = {key: pair_probabilities(texts, qwen, truth, fit, valid, *key, frozen.seed) for key in SPECIALISTS}
            inner.append((valid, inner_current, specialist, pair))
        # The incumbent's own choice of s, as claude_glm.evaluate makes it.
        s_scores = {s: float(np.mean([_ba(truth[valid], claude_glm.redistribute(base, specialist, s))
                                      for valid, base, specialist, _ in inner]))
                    for s in claude_glm.WEIGHTS}
        s = first_best(claude_glm.WEIGHTS, s_scores.__getitem__)
        scores = {option_name(option): [_ba(truth[valid], integrated(claude_glm.redistribute(base, specialist, s),
                                                                     pair, option))
                                        for valid, base, specialist, pair in inner]
                  for option in grid}
        means = {name: float(np.mean(values)) for name, values in scores.items()}
        chosen = first_best(grid, lambda option: means[option_name(option)])
        # The specialist reported where the incumbent is kept: its inner-best configuration and C.
        shown = chosen or first_best(grid[1:], lambda option: means[option_name(option)])
        rows, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        specialist = claude_glm.specialist_probabilities(blocks, INCUMBENT, truth, fit, rows, frozen)
        incumbent[rows] = claude_glm.redistribute(current[rows], specialist, s)
        pair = {key: pair_probabilities(texts, qwen, truth, fit, rows, *key, frozen.seed) for key in SPECIALISTS}
        for key, values in pair.items():
            fixed[specialist_name(*key)][rows] = values
        reported[rows] = pair[shown[:2]]
        for option in grid:
            by_option[option_name(option)][rows] = integrated(incumbent[rows], pair, option)
        candidate[rows] = by_option[option_name(chosen)][rows]
        selection.append({"outer_fold": fold, "s": s, "selected": option_name(chosen),
                          "reported_specialist": specialist_name(*shown[:2]),
                          "inner_mean_balanced_accuracy": means, "inner_balanced_accuracy": scores,
                          "s_inner_mean_balanced_accuracy": {str(key): value for key, value in s_scores.items()}})
    predictions = {"incumbent": incumbent.argmax(axis=1), "candidate": candidate.argmax(axis=1)}
    systems = {name: ensemble._system_metrics(truth, values, folds) for name, values in predictions.items()}
    paired = ensemble.prompt_bootstrap(truth, predictions["incumbent"], predictions["candidate"], prompts,
                                       samples, seed)
    gained = (predictions["incumbent"] != truth) & (predictions["candidate"] == truth)
    lost = (predictions["incumbent"] == truth) & (predictions["candidate"] != truth)
    outside = ~np.isin(truth, PAIR_INDEX)
    frame = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"}).assign(outer_fold=folds)
    for name, values in predictions.items():
        frame[f"prediction::{name}"] = np.asarray(LABELS)[values]
    frame.to_parquet(output / "predictions.parquet", index=False)
    pair_frame = train[["id"]].copy()
    for name, values in {**fixed, "reported": reported}.items():
        pair_frame[f"{name}::{PAIR[1]}"] = values[:, 1]
    pair_frame.to_parquet(output / "pair-probabilities.parquet", index=False)
    result = {
        "incumbent": INCUMBENT, "pair": list(PAIR), "configurations": {key: list(grid)
                                                                       for key, grid in CONFIGURATIONS.items()},
        "weights": list(WEIGHTS), "systems": systems, "paired": paired, "selection": selection,
        "binary": _binary(truth, reported, incumbent, candidate, fixed),
        "gemini_recall": {label: {name: systems[name]["recall"][label] for name in systems} for label in GEMINI},
        "swaps": _swaps(truth, predictions["incumbent"], predictions["candidate"]),
        "changes": {"gained": int(gained.sum()), "lost": int(lost.sum()),
                    "gained_outside_pair": int((gained & outside).sum()),
                    "lost_outside_pair": int((lost & outside).sum()),
                    "by_class": {label: {"gained": int(gained[truth == index].sum()),
                                         "lost": int(lost[truth == index].sum())}
                                 for index, label in enumerate(LABELS)}},
        # Each option applied in every fold: a diagnostic, never a selection.
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
    parser.add_argument("--groups", type=Path, help="CSV of id, prompt_id regrouping rows (e.g. by topic)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    train = read_train(args.train)
    blocks = claude_glm._load_blocks(train, args.qwen_features, args.styledistance_features)
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    train, plan = claude_glm._grouped(train, plan, args.groups)
    cache = pd.concat([pd.read_parquet(path) for path in args.current_inner], ignore_index=True)
    inputs = [args.plan, args.qwen_features, args.styledistance_features, args.current_oof, *args.current_inner,
              *([args.groups] if args.groups else [])]
    result = evaluate(train, blocks, plan, pd.read_parquet(args.current_oof), cache, args.output_dir,
                      provenance={"input_sha256": {str(path): claude_glm._sha256(path) for path in inputs}})
    paired = result["paired"]
    print(json.dumps({
        "balanced_accuracy": {name: round(values["balanced_accuracy"], 4) for name, values in result["systems"].items()},
        "delta": round(paired["balanced_accuracy_delta"], 4),
        "delta_95_interval": [round(bound, 4) for bound in paired["delta_95_interval"]],
        "s": [item["s"] for item in result["selection"]],
        "selected": [item["selected"] for item in result["selection"]],
        "binary": {key: value for key, value in result["binary"].items() if key != "fixed_specialists"},
        "fixed_specialists": {key: round(value, 4) for key, value in result["binary"]["fixed_specialists"].items()},
        "gemini_recall": result["gemini_recall"], "swaps": result["swaps"],
        "changes": {key: value for key, value in result["changes"].items() if key != "by_class"},
    }, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.gemini_pair import main as imported_main

    raise SystemExit(imported_main())
