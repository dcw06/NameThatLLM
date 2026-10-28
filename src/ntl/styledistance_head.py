"""Does the StyleDistance signal help beyond Claude-GLM? An 18-class Qwen + StyleDistance head.

The incumbent is the validated system: the current CPU + frozen-Qwen blend whose
Claude-GLM probability mass the Qwen + StyleDistance specialist redistributes at the
weight s its inner folds choose (`claude_glm.evaluate`). The candidate head is the
specialist's classifier (block-variance balanced, the same C) trained on all 18 classes
and blended after the incumbent: p = (1 - b) p_incumbent + b r, so b = 0 is the
incumbent exactly.

Inner prompt folds of each outer training partition choose s first, as the incumbent
does, then b from {0, 0.25, 0.5, 0.75, 1}, scored on all 18 classes; ties keep the
smaller b. A Qwen-only 18-class head, selected and blended the same way, is a diagnostic
control: it is the blend's own Qwen head, so it separates StyleDistance from more weight
on Qwen. Only the candidate is compared against the promotion rule.
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
from .selection import first_best

INCUMBENT = "qwen+styledistance"
CANDIDATE = "qwen+styledistance"
CONTROL = "qwen"
HEADS = {"candidate": CANDIDATE, "control": CONTROL}
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
GPT = ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra")


def head_probabilities(blocks: dict[str, np.ndarray], feature_set: str, truth: np.ndarray, fit: np.ndarray,
                       rows: np.ndarray, frozen: neural.FrozenConfig) -> np.ndarray:
    """The specialist's classifier fit on every class of `fit`; 18-class probabilities for `rows`."""
    names = claude_glm.feature_names(feature_set)
    features = np.hstack([blocks[name] for name in names])
    head = claude_glm.specialist_head(tuple(blocks[name].shape[1] for name in names), frozen)
    with threadpool_limits(limits=2):
        head.fit(features[fit], truth[fit])
        return neural._probabilities(head, features[rows])


def blended(incumbent: np.ndarray, head: np.ndarray, weight: float) -> np.ndarray:
    return incumbent if weight == 0 else (1 - weight) * incumbent + weight * head


def _ba(truth: np.ndarray, probabilities: np.ndarray) -> float:
    return float(balanced_accuracy_score(truth, probabilities.argmax(axis=1)))


def _gpt_confusion(truth: np.ndarray, predictions: np.ndarray) -> dict:
    """Luna/Sol/Terra rows by predicted label (outside the three as "other"), and the share
    of those rows predicted as another of the three."""
    matrix, within = {}, 0
    for label in GPT:
        predicted = predictions[truth == LABELS.index(label)]
        counts = {other: int((predicted == LABELS.index(other)).sum()) for other in GPT}
        counts["other"] = int(len(predicted) - sum(counts.values()))
        matrix[label] = counts
        within += sum(count for other, count in counts.items() if other not in (label, "other"))
    rows = int(np.isin(truth, [LABELS.index(label) for label in GPT]).sum())
    return {"matrix": matrix, "within_family_errors": within, "within_family_error_share": within / rows}


def _contributions(first: dict, second: dict) -> dict:
    """Each class's part of the balanced-accuracy change: its recall change over 18."""
    by_class = {label: (second["recall"][label] - first["recall"][label]) / len(LABELS) for label in LABELS}
    delta = second["balanced_accuracy"] - first["balanced_accuracy"]
    gpt = sum(by_class[label] for label in GPT)
    return {"by_class": by_class, "gpt_5_6": gpt, "gpt_5_6_share": gpt / delta if delta else None}


def _changes(truth: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict:
    gained, lost = (first != truth) & (second == truth), (first == truth) & (second != truth)
    return {"gained": int(gained.sum()), "lost": int(lost.sum()),
            "by_class": {label: {"gained": int(gained[truth == index].sum()), "lost": int(lost[truth == index].sum())}
                         for index, label in enumerate(LABELS)}}


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
    folds = plan["outer_fold"].to_numpy()
    current = claude_glm._aligned_current(current_oof, train, folds)
    incumbent = np.empty_like(current)
    heads = {system: np.empty_like(current) for system in HEADS}
    probabilities = {system: np.empty_like(current) for system in HEADS}
    by_weight = {system: {weight: np.empty_like(current) for weight in WEIGHTS} for system in HEADS}
    selection = []
    for fold in range(5):
        column = plan[f"inner_{fold}"].to_numpy()
        inner = []
        for inner_fold, (valid, inner_current) in claude_glm._inner_current(current_cache, train, plan,
                                                                            fold).items():
            fit = np.flatnonzero((folds != fold) & (column != inner_fold))
            specialist = claude_glm.specialist_probabilities(blocks, INCUMBENT, truth, fit, valid, frozen)
            inner.append((valid, inner_current, specialist,
                          {system: head_probabilities(blocks, name, truth, fit, valid, frozen)
                           for system, name in HEADS.items()}))
        # The incumbent's own choice of s, as claude_glm.evaluate makes it.
        s_scores = {s: float(np.mean([_ba(truth[valid], claude_glm.redistribute(base, specialist, s))
                                      for valid, base, specialist, _ in inner]))
                    for s in claude_glm.WEIGHTS}
        s = first_best(claude_glm.WEIGHTS, s_scores.__getitem__)
        scores = {system: {weight: [_ba(truth[valid], blended(claude_glm.redistribute(base, specialist, s),
                                                               head[system], weight))
                                    for valid, base, specialist, head in inner]
                           for weight in WEIGHTS}
                  for system in HEADS}
        means = {system: {weight: float(np.mean(values)) for weight, values in by.items()}
                 for system, by in scores.items()}
        # Ties, including those float rounding splits, keep the smaller b: stay with the incumbent.
        chosen = {system: first_best(WEIGHTS, means[system].__getitem__) for system in HEADS}
        rows, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        specialist = claude_glm.specialist_probabilities(blocks, INCUMBENT, truth, fit, rows, frozen)
        incumbent[rows] = claude_glm.redistribute(current[rows], specialist, s)
        for system, name in HEADS.items():
            heads[system][rows] = head_probabilities(blocks, name, truth, fit, rows, frozen)
            for weight in WEIGHTS:
                by_weight[system][weight][rows] = blended(incumbent[rows], heads[system][rows], weight)
            probabilities[system][rows] = by_weight[system][chosen[system]][rows]
        selection.append({"outer_fold": fold, "s": s, "b": chosen,
                          "inner_mean_balanced_accuracy": {system: {str(key): value for key, value in by.items()}
                                                           for system, by in means.items()},
                          "inner_balanced_accuracy": {system: {str(key): value for key, value in by.items()}
                                                      for system, by in scores.items()},
                          "s_inner_mean_balanced_accuracy": {str(key): value for key, value in s_scores.items()}})
    predictions = {"incumbent": incumbent.argmax(axis=1)} | {system: values.argmax(axis=1)
                                                             for system, values in probabilities.items()}
    systems = {name: ensemble._system_metrics(truth, values, folds) for name, values in predictions.items()}
    paired = {f"{name}_vs_incumbent": ensemble.prompt_bootstrap(truth, predictions["incumbent"], predictions[name],
                                                                prompts, samples, seed)
              for name in HEADS}
    paired["candidate_vs_control"] = ensemble.prompt_bootstrap(truth, predictions["control"],
                                                               predictions["candidate"], prompts, samples, seed)
    frame = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"}).assign(outer_fold=folds)
    for name, values in predictions.items():
        frame[f"prediction::{name}"] = np.asarray(LABELS)[values]
    frame.to_parquet(output / "predictions.parquet", index=False)
    for system, values in heads.items():
        np.save(output / f"head-{HEADS[system]}.npy", values)
    result = {
        "incumbent": INCUMBENT, "heads": HEADS, "weights": list(WEIGHTS), "gpt_classes": list(GPT),
        "systems": systems, "paired": paired, "selection": selection,
        "changes": {system: _changes(truth, predictions["incumbent"], predictions[system]) for system in HEADS},
        "contributions": {system: _contributions(systems["incumbent"], systems[system]) for system in HEADS},
        "gpt_confusion": {name: _gpt_confusion(truth, values) for name, values in predictions.items()},
        # Each head alone, and each b applied in every fold: diagnostics, never a selection.
        "head_alone": {system: ensemble._system_metrics(truth, values.argmax(axis=1), folds)
                       for system, values in heads.items()},
        "weight_balanced_accuracy": {system: {str(weight): _ba(truth, values) for weight, values in by.items()}
                                     for system, by in by_weight.items()},
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
    print(json.dumps({
        "balanced_accuracy": {name: round(values["balanced_accuracy"], 4) for name, values in result["systems"].items()},
        "paired": {name: [round(values["balanced_accuracy_delta"], 4)]
                   + [round(bound, 4) for bound in values["delta_95_interval"]]
                   for name, values in result["paired"].items()},
        "s": [item["s"] for item in result["selection"]],
        "b": {system: [item["b"][system] for item in result["selection"]] for system in HEADS},
        "gpt_recall": {name: {label: round(values["recall"][label], 3) for label in GPT}
                       for name, values in result["systems"].items()},
        "gpt_5_6_points": {system: round(100 * values["gpt_5_6"], 3) for system, values in result["contributions"].items()},
        "changes": {system: [values["gained"], values["lost"]] for system, values in result["changes"].items()},
        "head_alone": {system: round(values["balanced_accuracy"], 4) for system, values in result["head_alone"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.styledistance_head import main as imported_main

    raise SystemExit(imported_main())
