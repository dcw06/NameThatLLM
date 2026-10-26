"""Four-version CPU specialist, alone and blended with frozen StyleDistance scores.

The control refits the 18-class reference's exact character, word and style features
and class-weighted logistic regression on the true-Gemini outer-fitting rows only.
The blend averages its probabilities with a StyleDistance readout. The readout's C and
the blend weight are chosen jointly by the frozen experiment's inner prompt folds;
each outer validation partition is predicted once. End-to-end scores substitute the
specialist's version into the saved 18-class router predictions on all 2,699 rows.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from threadpoolctl import threadpool_limits

from .constants import LABELS
from .data import read_train
from .frozen_gemini import (
    FrozenGeminiConfig,
    _aligned_probabilities,
    _pipeline,
    _validate_frame,
    embeddings_sha256,
    inner_prompt_splits,
    load_feature_archive,
)
from .gemini import (
    GEMINI_LABELS,
    OFFICIAL_BASELINE_COUNTS,
    _cohort_metrics,
    _save_class_reports,
    prepare_gemini_evaluation,
)
from .modeling import BaselineConfig, build_pipeline

CONTROL = "cpu-specialist"
BLEND = "cpu-styledistance-blend"


@dataclass(frozen=True)
class SpecialistConfig:
    cpu: BaselineConfig = field(default_factory=BaselineConfig)
    readout: FrozenGeminiConfig = field(default_factory=FrozenGeminiConfig)
    weight_grid: tuple[float, ...] = tuple(step / 10 for step in range(11))
    bootstrap_samples: int = 10000

    def validate(self) -> None:
        self.readout.validate()
        if not self.weight_grid or len(set(self.weight_grid)) != len(self.weight_grid):
            raise ValueError("Blend weight grid must be nonempty and contain no duplicates")
        if any(not 0 <= weight <= 1 for weight in self.weight_grid):
            raise ValueError("Blend weights must lie in [0, 1]")
        if self.bootstrap_samples < 1:
            raise ValueError("Bootstrap sample count must be positive")


def _json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _cpu_probabilities(fit_texts: np.ndarray, fit_targets: np.ndarray, texts: np.ndarray,
                       config: BaselineConfig) -> np.ndarray:
    # The reference's features, C and class weighting; only the label set differs.
    pipeline = build_pipeline(config)
    pipeline.fit(fit_texts, fit_targets)
    return _aligned_probabilities(pipeline, texts)


def _blend(cpu: np.ndarray, readout: np.ndarray, weight: float) -> np.ndarray:
    return (1 - weight) * cpu + weight * readout


def _select_blend(
    texts: np.ndarray, features: np.ndarray, targets: np.ndarray, prompts: np.ndarray,
    config: SpecialistConfig, outer_fold: int,
) -> tuple[float, float, dict]:
    """Choose readout C and weight from inner prompt folds; no outer rows are passed in."""
    splits, split_metadata = inner_prompt_splits(targets, prompts, config.readout, outer_fold)
    cpu, readout = [], {C: [] for C in config.readout.C_grid}
    for fit, valid in splits:
        cpu.append(_cpu_probabilities(texts[fit], targets[fit], texts[valid], config.cpu))
        for C in config.readout.C_grid:
            estimator = _pipeline(C, config.readout)
            estimator.fit(features[fit], targets[fit])
            readout[C].append(_aligned_probabilities(estimator, features[valid]))
    candidates = []
    for C in sorted(config.readout.C_grid):
        for weight in sorted(config.weight_grid):
            scores = [float(balanced_accuracy_score(targets[valid], _blend(p_cpu, p_readout, weight).argmax(axis=1)))
                      for (_, valid), p_cpu, p_readout in zip(splits, cpu, readout[C], strict=True)]
            candidates.append({"C": C, "weight": weight, "inner_balanced_accuracy": scores,
                               "mean_inner_balanced_accuracy": float(np.mean(scores))})
    # Ties prefer the smaller StyleDistance weight (closer to the control), then smaller C.
    best = min(candidates, key=lambda item: (-item["mean_inner_balanced_accuracy"], item["weight"], item["C"]))
    control_scores = [float(balanced_accuracy_score(targets[valid], p_cpu.argmax(axis=1)))
                      for (_, valid), p_cpu in zip(splits, cpu, strict=True)]
    return best["C"], best["weight"], {
        "outer_fold": outer_fold, "selected_C": best["C"], "selected_weight": best["weight"],
        "selected_mean_inner_balanced_accuracy": best["mean_inner_balanced_accuracy"],
        "control_inner_balanced_accuracy": control_scores, "candidates": candidates,
        "inner_splits": split_metadata,
        "selection_metric": "Unweighted mean inner-fold balanced accuracy; ties choose smaller weight, then smaller C.",
    }


def _paired(frame: pd.DataFrame, first: np.ndarray, second: np.ndarray, samples: int, seed: int) -> dict:
    """Paired accuracy change, with a bootstrap that resamples whole prompts."""
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    _, prompt_index = np.unique(frame["prompt_id"].to_numpy(), return_inverse=True)
    prompt_delta = np.bincount(prompt_index, weights=second.astype(float) - first.astype(float))
    prompt_rows = np.bincount(prompt_index)
    draws = np.random.default_rng(seed).integers(0, len(prompt_rows), size=(samples, len(prompt_rows)))
    deltas = prompt_delta[draws].sum(axis=1) / prompt_rows[draws].sum(axis=1)
    return {
        "rows": len(frame), "prompts": len(prompt_rows),
        "first_correct": int(first.sum()), "second_correct": int(second.sum()),
        "accuracy_delta": float(second.mean() - first.mean()),
        "both_correct": int((first & second).sum()), "second_only_correct": int((~first & second).sum()),
        "first_only_correct": int((first & ~second).sum()), "both_wrong": int((~first & ~second).sum()),
        "prompt_bootstrap": {"samples": samples, "seed": seed,
                             "accuracy_delta_95_interval": [float(np.quantile(deltas, 0.025)),
                                                            float(np.quantile(deltas, 0.975))],
                             "fraction_delta_at_most_zero": float((deltas <= 0).mean())},
    }


def end_to_end(baseline_oof: pd.DataFrame, versions: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    """Score every official row after substituting each specialist's Gemini version.

    Only the 570 true-Gemini rows routed to Gemini change. A non-Gemini passage routed
    to Gemini stays wrong under every Gemini version, and a true-Gemini passage routed
    elsewhere stays wrong, so these 18-class recalls are exact for the two-stage system.
    """
    result = baseline_oof[["id", "prompt_id", "true_model", "fold"]].copy()
    result["reference"] = baseline_oof["predicted_model"].to_numpy()
    true_gemini = result["true_model"].isin(GEMINI_LABELS)
    routed = result["reference"].isin(GEMINI_LABELS)
    for name, frame in versions.items():
        replacement = frame.loc[frame["selected"]].set_index("id")["prediction"]
        result[name] = result["reference"]
        mask = result["id"].isin(replacement.index)
        if not (mask == (true_gemini & routed)).all():
            raise ValueError("Specialist rows must be exactly the true-Gemini rows routed to Gemini")
        result.loc[mask, name] = result.loc[mask, "id"].map(replacement)
    systems = {}
    for name in ["reference", *versions]:
        recall = recall_score(result["true_model"], result[name], labels=list(LABELS), average=None, zero_division=0)
        systems[name] = {
            "balanced_accuracy": float(balanced_accuracy_score(result["true_model"], result[name])),
            "accuracy": float(accuracy_score(result["true_model"], result[name])),
            "correct": int((result["true_model"] == result[name]).sum()),
            "gemini_recall": {label: float(recall[LABELS.index(label)]) for label in GEMINI_LABELS},
        }
    return {
        "rows": len(result),
        "rows_routed_to_gemini": int(routed.sum()),
        "true_gemini_routed_to_gemini": int((true_gemini & routed).sum()),
        "non_gemini_routed_to_gemini": int((~true_gemini & routed).sum()),
        "true_gemini_routed_elsewhere": int((true_gemini & ~routed).sum()),
        "systems": systems,
        "scoring_note": "Non-Gemini rows routed to Gemini are scored incorrect, as they are under any Gemini version.",
    }, result


def _save_variant(frame: pd.DataFrame, probabilities: np.ndarray, directory: Path,
                  components: dict[str, np.ndarray] | None = None) -> tuple[pd.DataFrame, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    result = frame[["id", "prompt_id", "model", "fold", "baseline_model", "selected", "baseline_correct"]].copy()
    result["prediction"] = np.asarray(GEMINI_LABELS)[probabilities.argmax(axis=1)]
    result["confidence"] = probabilities.max(axis=1)
    result["specialist_correct"] = result["prediction"] == result["model"]
    for index, label in enumerate(GEMINI_LABELS):
        result[f"probability::{label}"] = probabilities[:, index]
    for name, values in (components or {}).items():
        for index, label in enumerate(GEMINI_LABELS):
            result[f"{name}::{label}"] = values[:, index]
    selected = result.loc[result["selected"]].copy()
    metrics = {"all_true_gemini": _cohort_metrics(result), "correctly_routed_gemini": _cohort_metrics(selected)}
    result.to_parquet(directory / "oof.parquet", index=False)
    selected.to_csv(directory / "selected_predictions.csv", index=False)
    _save_class_reports(result, directory, "")
    _save_class_reports(selected, directory, "_selected")
    _json(directory / "metrics.json", metrics)
    return result, metrics


def evaluate_specialists(
    gemini_frame: pd.DataFrame, features: np.ndarray, baseline_oof: pd.DataFrame, output_dir: str | Path,
    config: SpecialistConfig | None = None, feature_metadata: dict | None = None,
    protocol_metadata: dict | None = None,
) -> dict:
    config = config or SpecialistConfig()
    config.validate()
    frame = _validate_frame(gemini_frame)
    if "text" not in frame or not frame["text"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("The CPU specialist needs the exact nonempty passage text of every Gemini row")
    features = np.asarray(features)
    if len(features) != len(frame):
        raise ValueError("Features must already be aligned to every Gemini evaluation row")
    feature_hash = embeddings_sha256(features)
    if feature_metadata is not None and feature_metadata.get("aligned_embeddings_sha256") != feature_hash:
        raise ValueError("Aligned feature hash differs from the provided provenance")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory; an existing experiment must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    texts = frame["text"].to_numpy(dtype=object)
    targets = frame["model"].map({label: index for index, label in enumerate(GEMINI_LABELS)}).to_numpy()
    prompts = frame["prompt_id"].to_numpy()
    control, readout, blended = (np.full((len(frame), 4), np.nan) for _ in range(3))
    counts = np.zeros(len(frame), dtype=np.int64)
    nested_results = []
    started = time.perf_counter()
    for outer_fold in range(5):
        fit, valid = (frame["fold"] != outer_fold).to_numpy(), (frame["fold"] == outer_fold).to_numpy()
        if set(prompts[fit]) & set(prompts[valid]):
            raise ValueError("Prompt leakage detected before the outer specialist fit")
        if set(targets[fit]) != set(range(4)) or set(targets[valid]) != set(range(4)):
            raise ValueError("Every outer fitting/validation partition must contain all four Gemini classes")
        with threadpool_limits(limits=2):
            C, weight, nested = _select_blend(texts[fit], features[fit], targets[fit], prompts[fit], config, outer_fold)
            # The control is the blend's own CPU component, fitted once on the outer partition.
            control[valid] = _cpu_probabilities(texts[fit], targets[fit], texts[valid], config.cpu)
            estimator = _pipeline(C, config.readout)
            estimator.fit(features[fit], targets[fit])
            readout[valid] = _aligned_probabilities(estimator, features[valid])
        blended[valid] = _blend(control[valid], readout[valid], weight)
        counts[valid] += 1
        nested.update({
            "outer_train_rows": int(fit.sum()), "outer_validation_rows": int(valid.sum()),
            "outer_train_selected_rows": int(frame.loc[fit, "selected"].sum()),
            "outer_validation_control_balanced_accuracy": float(balanced_accuracy_score(targets[valid], control[valid].argmax(axis=1))),
            "outer_validation_blend_balanced_accuracy": float(balanced_accuracy_score(targets[valid], blended[valid].argmax(axis=1))),
        })
        nested_results.append(nested)
        _json(output / "inner_cv.json", nested_results)
    if not np.all(counts == 1) or not all(np.isfinite(values).all() for values in (control, readout, blended)):
        raise ValueError("Every true-Gemini row must have exactly one complete outer OOF prediction")
    control_result, control_metrics = _save_variant(frame, control, output / CONTROL)
    blend_result, blend_metrics = _save_variant(frame, blended, output / BLEND,
                                                {"cpu": control, "styledistance": readout})
    samples, seed = config.bootstrap_samples, config.readout.seed
    paired = {}
    for cohort, mask in (("correctly_routed_gemini", frame["selected"].to_numpy()),
                         ("all_true_gemini", np.ones(len(frame), dtype=bool))):
        cohort_frame = frame.loc[mask]
        reference = frame.loc[mask, "baseline_correct"].to_numpy()
        control_correct = control_result.loc[mask, "specialist_correct"].to_numpy()
        blend_correct = blend_result.loc[mask, "specialist_correct"].to_numpy()
        paired[cohort] = {
            "control_vs_reference": _paired(cohort_frame, reference, control_correct, samples, seed),
            "blend_vs_control": _paired(cohort_frame, control_correct, blend_correct, samples, seed),
            "blend_vs_reference": _paired(cohort_frame, reference, blend_correct, samples, seed),
        }
    full, full_predictions = end_to_end(baseline_oof, {CONTROL: control_result, BLEND: blend_result})
    full_predictions.to_parquet(output / "end_to_end_predictions.parquet", index=False)
    comparison = {
        CONTROL: control_metrics, BLEND: blend_metrics, "paired": paired, "end_to_end": full,
        "selected_C_per_outer_fold": [item["selected_C"] for item in nested_results],
        "selected_weight_per_outer_fold": [item["selected_weight"] for item in nested_results],
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": ("Paired comparisons are conditional on true Gemini; end_to_end scores the two-stage "
                           "system on all 2,699 rows with the saved CPU router."),
    }
    versions = {package: importlib.metadata.version(package) for package in
                ("numpy", "pandas", "scikit-learn", "scipy")}
    protocol = None if protocol_metadata is None else dict(protocol_metadata)
    if protocol is not None:
        protocol["outer_validation_policy"] = ("Blend C and weight are selected only by inner prompt-grouped "
                                               "balanced accuracy; each outer partition is predicted once.")
    _json(output / "metadata.json", {
        "schema_version": 1, "kind": "gemini-four-class-cpu-specialist-and-styledistance-blend",
        "labels": list(GEMINI_LABELS), "configuration": asdict(config), "versions": versions,
        "feature_provenance": feature_metadata, "aligned_embeddings_sha256": feature_hash,
        "protocol": protocol,
        "blend": "(1 - weight) * CPU probabilities + weight * StyleDistance probabilities; weight 0 is the control.",
        "training_population": "Every true-Gemini row in outer fitting folds, including CPU routing false negatives.",
        "learned_transforms": "Fresh TF-IDF vocabularies, scalers and classifiers fit on each inner/outer fitting partition.",
    })
    _json(output / "comparison.json", comparison)
    return comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--baseline-oof", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args(argv)
    baseline_oof = pd.read_parquet(args.baseline_oof)
    frame, protocol = prepare_gemini_evaluation(read_train(args.train), baseline_oof,
                                               args.fold_manifest, OFFICIAL_BASELINE_COUNTS)
    features, metadata = load_feature_archive(args.features, frame)
    config = SpecialistConfig(bootstrap_samples=args.bootstrap_samples)
    comparison = evaluate_specialists(frame, features, baseline_oof, args.output_dir, config, metadata, protocol)
    summary = {name: {cohort: comparison[name][cohort]["specialist_correct"] for cohort in comparison[name]}
               for name in (CONTROL, BLEND)}
    summary["end_to_end_balanced_accuracy"] = {name: values["balanced_accuracy"]
                                               for name, values in comparison["end_to_end"]["systems"].items()}
    summary["selected_weight_per_outer_fold"] = comparison["selected_weight_per_outer_fold"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
