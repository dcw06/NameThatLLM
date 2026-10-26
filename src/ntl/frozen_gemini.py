"""Nested prompt-grouped linear readouts of immutable frozen Gemini embeddings.

Encoder extraction is independent of labels. All learned scaling and regularized
classification are fit within the inner/outer training partitions. The original
570-row CPU-routed cohort changes reporting only; all 600 true-Gemini rows enter
the original five outer prompt folds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import read_train
from .gemini import (
    GEMINI_LABELS,
    OFFICIAL_BASELINE_COUNTS,
    _cohort_metrics,
    _save_class_reports,
    prepare_gemini_evaluation,
)


@dataclass(frozen=True)
class FrozenGeminiConfig:
    C_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    inner_folds: int = 3
    max_iter: int = 2000
    seed: int = 2026

    def validate(self) -> None:
        if not self.C_grid or any(not np.isfinite(c) or c <= 0 for c in self.C_grid):
            raise ValueError("Regularization grid must contain finite positive C values")
        if len(set(self.C_grid)) != len(self.C_grid):
            raise ValueError("Regularization grid contains duplicate C values")
        if self.inner_folds < 2 or self.max_iter < 1:
            raise ValueError("Inner fold count must be at least two and max_iter must be positive")


def _json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _string_ids(ids: Sequence[str]) -> list[str]:
    values = list(ids)
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Embedding ids must be nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError("Embedding ids must be unique")
    return values


def ids_sha256(ids: Sequence[str]) -> str:
    """SHA-256 of UTF-8 compact JSON list, retaining the archive's row order."""
    return hashlib.sha256(json.dumps(_string_ids(ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def embeddings_sha256(features: np.ndarray) -> str:
    """Hash sorted-key JSON shape/dtype.str followed by C-order raw array bytes.

    Archive dtype and byte order are preserved; this does not round, cast or
    normalize values. Import this helper instead of duplicating the contract.
    """
    array = np.asarray(features)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
        raise ValueError("Embeddings must be a two-dimensional floating-point array")
    if min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError("Embeddings must be nonempty and finite")
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(array.shape), "dtype": array.dtype.str}, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def input_text_sha256(ids: Sequence[str], texts: Sequence[str]) -> str:
    """Hash compact UTF-8 JSON [id, exact_text] lines in archive row order."""
    ids = _string_ids(ids)
    texts = list(texts)
    if len(ids) != len(texts) or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Each embedding id needs its exact nonempty input text")
    digest = hashlib.sha256()
    for pair in zip(ids, texts, strict=True):
        digest.update(json.dumps(pair, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_provenance(metadata: dict) -> None:
    if metadata.get("schema_version") != 1 or metadata.get("encoder_frozen") is not True:
        raise ValueError("Features require schema_version 1 and encoder_frozen=true provenance")
    if not isinstance(metadata.get("model_id"), str) or not metadata["model_id"].strip():
        raise ValueError("Feature provenance is missing model_id")
    if not isinstance(metadata.get("model_revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", metadata["model_revision"]):
        raise ValueError("Feature provenance requires an immutable model revision")
    options = metadata.get("feature_options")
    if not isinstance(options, dict) or not {"pooling", "max_length", "normalization"}.issubset(options):
        raise ValueError("Feature provenance must describe pooling, max_length and normalization")
    if not isinstance(options["pooling"], str) or not options["pooling"].strip():
        raise ValueError("Feature pooling must be explicit")
    if not isinstance(options["max_length"], int) or options["max_length"] < 1:
        raise ValueError("Feature max_length must be a positive integer")
    if not isinstance(options["normalization"], (str, bool)):
        raise ValueError("Fixed feature normalization must be explicit")


def save_feature_archive(
    path: str | Path, ids: Sequence[str], embeddings: np.ndarray, metadata: dict,
    *, texts: Sequence[str],
) -> dict:
    """Write a pickle-free NPZ with exact row/text/array identity and source pins."""
    ids = _string_ids(ids)
    array = np.asarray(embeddings)
    array_hash = embeddings_sha256(array)
    if len(ids) != len(array):
        raise ValueError("Embedding row count does not match ids")
    value = dict(metadata)
    value["schema_version"] = 1
    value["ids_sha256"] = ids_sha256(ids)
    value["embeddings_sha256"] = array_hash
    value["input_text_sha256"] = input_text_sha256(ids, texts)
    _validate_provenance(value)
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("Embedding archive filename must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, ids=np.asarray(ids, dtype=str), embeddings=array,
                        metadata=np.asarray(json.dumps(value, sort_keys=True)))
    return value


def load_feature_archive(path: str | Path, gemini_frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Verify stored row order before aligning features to the evaluation frame."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"ids", "embeddings", "metadata"}:
            raise ValueError("Feature archive must contain exactly ids, embeddings and metadata")
        ids_array = archive["ids"]
        if ids_array.ndim != 1 or ids_array.dtype.kind != "U":
            raise ValueError("Feature ids must be a one-dimensional Unicode array")
        ids = _string_ids(ids_array.tolist())
        features = archive["embeddings"]
        encoded_metadata = archive["metadata"]
        if encoded_metadata.ndim != 0 or encoded_metadata.dtype.kind != "U":
            raise ValueError("Feature metadata must be a scalar JSON string")
        metadata = json.loads(str(encoded_metadata.item()))
    if not isinstance(metadata, dict):
        raise ValueError("Feature metadata must be a JSON object")
    _validate_provenance(metadata)
    if metadata.get("ids_sha256") != ids_sha256(ids):
        raise ValueError("Feature id hash mismatch")
    if metadata.get("embeddings_sha256") != embeddings_sha256(features):
        raise ValueError("Feature array hash mismatch")
    if len(features) != len(ids):
        raise ValueError("Embedding rows and ids differ in length")
    expected_ids = _string_ids(gemini_frame["id"].tolist())
    if len(ids) != len(expected_ids) or set(ids) != set(expected_ids):
        raise ValueError("Feature ids must exactly cover all true-Gemini evaluation rows")
    texts = gemini_frame.set_index("id")["text"]
    if metadata.get("input_text_sha256") != input_text_sha256(ids, texts.loc[ids].tolist()):
        raise ValueError("Feature input-text hash mismatch")
    row_lookup = {value: index for index, value in enumerate(ids)}
    aligned = features[[row_lookup[value] for value in expected_ids]]
    provenance = dict(metadata)
    provenance["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance["aligned_embeddings_sha256"] = embeddings_sha256(aligned)
    return aligned, provenance


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "prompt_id", "model", "fold", "baseline_model", "selected", "baseline_correct"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Gemini evaluation frame is missing required data")
    if frame[list(required)].isna().any().any() or frame["id"].duplicated().any():
        raise ValueError("Gemini evaluation rows contain nulls or repeated ids")
    _string_ids(frame["id"].tolist())
    if set(frame["model"]) != set(GEMINI_LABELS):
        raise ValueError("Gemini evaluation frame must contain all four canonical Gemini labels")
    if not pd.api.types.is_integer_dtype(frame["fold"]) or set(frame["fold"]) != set(range(5)):
        raise ValueError("Gemini evaluation requires the original five outer folds")
    if frame.groupby("prompt_id")["fold"].nunique().max() != 1:
        raise ValueError("Prompt leakage between outer folds")
    if not pd.api.types.is_bool_dtype(frame["selected"]) or not pd.api.types.is_bool_dtype(frame["baseline_correct"]):
        raise ValueError("Cohort and baseline correctness masks must be boolean")
    if not frame["selected"].equals(frame["baseline_model"].isin(GEMINI_LABELS)):
        raise ValueError("Selected cohort differs from the saved Gemini-routing condition")
    if not frame["baseline_correct"].equals(frame["baseline_model"] == frame["model"]):
        raise ValueError("Baseline correctness mask differs from the true labels")
    if not frame["selected"].any():
        raise ValueError("The selected Gemini reporting cohort is empty")
    return frame.reset_index(drop=True).copy()


def _pipeline(C: float, config: FrozenGeminiConfig) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                           max_iter=config.max_iter, random_state=config.seed)),
    ])


def _aligned_probabilities(pipeline: Pipeline, features: np.ndarray) -> np.ndarray:
    classes = list(pipeline.named_steps["classifier"].classes_)
    if len(classes) != 4 or set(classes) != set(range(4)):
        raise ValueError("Classifier did not fit the four canonical Gemini label indices")
    values = pipeline.predict_proba(features)[:, [classes.index(index) for index in range(4)]]
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("Classifier probabilities are non-finite or invalid")
    return values


def _select_regularization(
    fitting_features: np.ndarray, fitting_targets: np.ndarray, fitting_prompts: np.ndarray,
    config: FrozenGeminiConfig, outer_fold: int,
) -> tuple[float, dict]:
    """No outer-validation features, labels or scores enter this selection API."""
    splitter = StratifiedGroupKFold(n_splits=config.inner_folds, shuffle=True,
                                    random_state=config.seed + outer_fold)
    splits = list(splitter.split(fitting_features, fitting_targets, groups=fitting_prompts))
    split_metadata = []
    for inner_fold, (fit, valid) in enumerate(splits):
        fit_groups, valid_groups = set(fitting_prompts[fit]), set(fitting_prompts[valid])
        if fit_groups & valid_groups:
            raise ValueError("Prompt leakage in inner regularization folds")
        if set(fitting_targets[fit]) != set(range(4)) or set(fitting_targets[valid]) != set(range(4)):
            raise ValueError("Every inner fitting/validation partition must contain all four Gemini classes")
        split_metadata.append({"inner_fold": inner_fold, "training_rows": len(fit),
                               "validation_rows": len(valid), "training_prompts": sorted(fit_groups),
                               "validation_prompts": sorted(valid_groups)})
    candidates = []
    for C in sorted(config.C_grid):
        scores = []
        for fit, valid in splits:
            estimator = _pipeline(C, config)
            estimator.fit(fitting_features[fit], fitting_targets[fit])
            probabilities = _aligned_probabilities(estimator, fitting_features[valid])
            scores.append(float(balanced_accuracy_score(fitting_targets[valid], probabilities.argmax(axis=1))))
        candidates.append({"C": C, "inner_balanced_accuracy": scores,
                           "mean_inner_balanced_accuracy": float(np.mean(scores))})
    best = min(candidates, key=lambda item: (-item["mean_inner_balanced_accuracy"], item["C"]))
    return best["C"], {"outer_fold": outer_fold, "selected_C": best["C"], "candidates": candidates,
                       "inner_splits": split_metadata,
                       "selection_metric": "Unweighted mean inner-fold balanced accuracy; ties choose smaller C."}


def evaluate_embeddings(
    gemini_frame: pd.DataFrame, features: np.ndarray, output_dir: str | Path,
    config: FrozenGeminiConfig | None = None, feature_metadata: dict | None = None,
    protocol_metadata: dict | None = None,
) -> dict:
    config = config or FrozenGeminiConfig()
    config.validate()
    frame = _validate_frame(gemini_frame)
    features = np.asarray(features)
    feature_hash = embeddings_sha256(features)
    if len(features) != len(frame):
        raise ValueError("Features must already be aligned to every Gemini evaluation row")
    if feature_metadata is not None:
        _validate_provenance(feature_metadata)
        if feature_metadata.get("aligned_embeddings_sha256", feature_metadata.get("embeddings_sha256")) != feature_hash:
            raise ValueError("Aligned feature hash differs from the provided provenance")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists() or (output / "oof.parquet").exists():
        raise ValueError("Use a new output directory for each frozen-encoder evaluation")
    targets = frame["model"].map({label: index for index, label in enumerate(GEMINI_LABELS)}).to_numpy()
    probabilities = np.full((len(frame), 4), np.nan)
    counts = np.zeros(len(frame), dtype=np.int64)
    nested_results = []
    started = time.perf_counter()
    for outer_fold in range(5):
        fit, valid = (frame["fold"] != outer_fold).to_numpy(), (frame["fold"] == outer_fold).to_numpy()
        if set(frame.loc[fit, "prompt_id"]) & set(frame.loc[valid, "prompt_id"]):
            raise ValueError("Prompt leakage detected before the outer classifier fit")
        if set(targets[fit]) != set(range(4)) or set(targets[valid]) != set(range(4)):
            raise ValueError("Every outer fitting/validation partition must contain all four Gemini classes")
        with threadpool_limits(limits=2):
            C, nested = _select_regularization(features[fit], targets[fit], frame.loc[fit, "prompt_id"].to_numpy(),
                                               config, outer_fold)
            estimator = _pipeline(C, config)
            estimator.fit(features[fit], targets[fit])
            probabilities[valid] = _aligned_probabilities(estimator, features[valid])
        counts[valid] += 1
        nested["outer_train_rows"] = int(fit.sum())
        nested["outer_validation_rows"] = int(valid.sum())
        nested["outer_train_selected_rows"] = int(frame.loc[fit, "selected"].sum())
        nested["outer_validation_balanced_accuracy"] = float(balanced_accuracy_score(targets[valid], probabilities[valid].argmax(axis=1)))
        nested_results.append(nested)
        _json(output / "inner_cv.json", nested_results)
    if not np.all(counts == 1) or not np.isfinite(probabilities).all():
        raise ValueError("Every true-Gemini row must have exactly one complete outer OOF prediction")
    result = frame[["id", "prompt_id", "model", "fold", "baseline_model", "selected", "baseline_correct"]].copy()
    result["prediction"] = np.asarray(GEMINI_LABELS)[probabilities.argmax(axis=1)]
    result["confidence"] = probabilities.max(axis=1)
    result["specialist_correct"] = result["prediction"] == result["model"]
    for index, label in enumerate(GEMINI_LABELS):
        result[f"probability::{label}"] = probabilities[:, index]
    selected = result.loc[result["selected"]].copy()
    metrics = {"all_true_gemini": _cohort_metrics(result), "correctly_routed_gemini": _cohort_metrics(selected),
               "selected_C_per_outer_fold": [item["selected_C"] for item in nested_results],
               "elapsed_seconds": time.perf_counter() - started,
               "interpretation": "Nested prompt-grouped frozen-feature comparison; 570-row selection is for reporting only, and is not a hidden-test or end-to-end routing score."}
    versions = {package: importlib.metadata.version(package) for package in
                ("numpy", "pandas", "scikit-learn", "scipy")}
    protocol = None if protocol_metadata is None else dict(protocol_metadata)
    if protocol is not None:
        protocol["outer_validation_policy"] = "C is selected only by inner prompt-grouped balanced accuracy; each outer validation partition is predicted once."
    metadata = {"schema_version": 1, "kind": "frozen-encoder-nested-gemini-linear-readout",
                "labels": list(GEMINI_LABELS), "configuration": asdict(config), "versions": versions,
                "feature_provenance": feature_metadata,
                "aligned_embeddings_sha256": feature_hash,
                "aligned_ids_sha256": ids_sha256(frame["id"].tolist()),
                "protocol": protocol,
                "selection_policy": "Inner prompt-grouped folds only; outer validation is predicted once after choosing C.",
                "training_population": "Every true-Gemini row in outer fitting folds, including CPU routing false negatives.",
                "learned_transforms": "Fresh StandardScaler and L2 logistic regression fit on each inner/outer fitting partition."}
    result.to_parquet(output / "oof.parquet", index=False)
    selected.to_parquet(output / "selected_oof.parquet", index=False)
    selected.to_csv(output / "selected_predictions.csv", index=False)
    _save_class_reports(result, output, "")
    _save_class_reports(selected, output, "_selected")
    _json(output / "metadata.json", metadata)
    _json(output / "metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--baseline-oof", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--C-grid", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    frame, protocol = prepare_gemini_evaluation(read_train(args.train), pd.read_parquet(args.baseline_oof),
                                               args.fold_manifest, OFFICIAL_BASELINE_COUNTS)
    features, metadata = load_feature_archive(args.features, frame)
    config = FrozenGeminiConfig(C_grid=tuple(args.C_grid), inner_folds=args.inner_folds,
                                max_iter=args.max_iter, seed=args.seed)
    metrics = evaluate_embeddings(frame, features, args.output_dir, config, metadata, protocol)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
