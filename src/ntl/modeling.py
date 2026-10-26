"""Fold-local CPU training, aligned OOF scores and portable inference artifacts."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from . import __version__
from .constants import DEFAULT_SEED, LABELS
from .data import (
    data_quality_report,
    fingerprint_frame,
    validate_test,
    validate_train,
    write_submission,
)
from .features import build_features
from .folds import build_fold_manifest, load_fold_manifest, save_fold_manifest


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = DEFAULT_SEED
    char_max_features: int = 50000
    word_max_features: int = 30000
    C: float = 4.0
    max_iter: int = 1000


def build_pipeline(config: BaselineConfig | None = None) -> Pipeline:
    config = config or BaselineConfig()
    if config.C <= 0 or config.max_iter < 1 or min(config.char_max_features, config.word_max_features) < 1:
        raise ValueError("Feature limits, C and max_iter must be positive")
    return Pipeline([
        ("features", build_features(config.char_max_features, config.word_max_features)),
        ("classifier", LogisticRegression(
            C=config.C, max_iter=config.max_iter, class_weight="balanced", solver="lbfgs",
            random_state=config.seed,
        )),
    ])


def aligned_probabilities(pipeline: Pipeline, texts) -> np.ndarray:
    """Match estimator classes to LABELS explicitly; sklearn sorts labels itself."""
    classes = list(pipeline.named_steps["classifier"].classes_)
    if len(classes) != len(LABELS) or set(classes) != set(LABELS):
        raise ValueError("The fitted classifier does not contain all 18 competition labels")
    probabilities = pipeline.predict_proba(texts)
    if probabilities.shape != (len(texts), len(classes)):
        raise ValueError("Unexpected classifier probability shape")
    aligned = probabilities[:, [classes.index(label) for label in LABELS]]
    if not np.isfinite(aligned).all() or not np.allclose(aligned.sum(axis=1), 1.0):
        raise ValueError("Invalid prediction probabilities")
    return aligned


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _metadata(train: pd.DataFrame, config: BaselineConfig) -> dict:
    return {
        "schema_version": 1,
        "package_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "char_word_stylometry_logistic_regression",
        "labels": list(LABELS),
        "train_fingerprint": fingerprint_frame(train),
        "n_rows": len(train),
        "n_prompts": int(train["prompt_id"].nunique()),
        "configuration": asdict(config),
        "data_quality": data_quality_report(train),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "text_processing": "Case and punctuation preserved; TF-IDF vocabularies and numeric scaling fit on training rows only.",
        "probability_note": "Class-weighted classifier probabilities are scores, not calibrated deployment posteriors.",
    }


def run_cross_validation(
    train: pd.DataFrame, output_dir: str | Path, n_splits: int = 5,
    config: BaselineConfig | None = None, fold_manifest: str | Path | None = None,
) -> dict:
    train = validate_train(train).reset_index(drop=True)
    config = config or BaselineConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if fold_manifest is None:
        manifest, manifest_metadata = build_fold_manifest(train, n_splits=n_splits, seed=config.seed)
    else:
        manifest, manifest_metadata = load_fold_manifest(train, fold_manifest)
        if manifest_metadata["n_splits"] != n_splits:
            raise ValueError("Requested fold count differs from the supplied manifest")
    save_fold_manifest(manifest, manifest_metadata, output)
    folds = train["id"].map(manifest.set_index("id")["fold"]).to_numpy()
    probabilities = np.full((len(train), len(LABELS)), np.nan)
    counts = np.zeros(len(train), dtype=np.int64)
    fold_metrics = []
    for fold in range(n_splits):
        fit_mask, valid_mask = folds != fold, folds == fold
        if set(train.loc[fit_mask, "prompt_id"]) & set(train.loc[valid_mask, "prompt_id"]):
            raise ValueError("Prompt leakage detected before model fitting")
        # A new pipeline per fold ensures no vocabulary/scaler is fit on held-out rows.
        pipeline = build_pipeline(config)
        pipeline.fit(train.loc[fit_mask, "text"], train.loc[fit_mask, "model"])
        fold_probabilities = aligned_probabilities(pipeline, train.loc[valid_mask, "text"])
        probabilities[valid_mask] = fold_probabilities
        counts[valid_mask] += 1
        predictions = np.asarray(LABELS)[fold_probabilities.argmax(axis=1)]
        fold_metrics.append({
            "fold": fold,
            "n_train": int(fit_mask.sum()), "n_validation": int(valid_mask.sum()),
            "n_train_prompts": int(train.loc[fit_mask, "prompt_id"].nunique()),
            "n_validation_prompts": int(train.loc[valid_mask, "prompt_id"].nunique()),
            "balanced_accuracy": float(balanced_accuracy_score(train.loc[valid_mask, "model"], predictions)),
        })
    if not np.all(counts == 1) or not np.isfinite(probabilities).all():
        raise ValueError("Every training row must receive exactly one finite OOF prediction")
    predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    oof = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"}).copy()
    oof["fold"] = folds
    oof["predicted_model"] = predictions
    for index, label in enumerate(LABELS):
        oof[f"probability::{label}"] = probabilities[:, index]
    oof.to_parquet(output / "oof.parquet", index=False)
    precision, recall, f1, support = precision_recall_fscore_support(
        train["model"], predictions, labels=list(LABELS), zero_division=0,
    )
    pd.DataFrame({"model": LABELS, "precision": precision, "recall": recall,
                  "f1": f1, "support": support}).to_csv(output / "class_metrics.csv", index=False)
    pd.DataFrame(confusion_matrix(train["model"], predictions, labels=list(LABELS)),
                 index=pd.Index(LABELS, name="true_model"), columns=LABELS).to_csv(output / "confusion_matrix.csv")
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(train["model"], predictions)),
        "accuracy": float(accuracy_score(train["model"], predictions)),
        "fold_balanced_accuracy_mean": float(np.mean([m["balanced_accuracy"] for m in fold_metrics])),
        "fold_balanced_accuracy_std": float(np.std([m["balanced_accuracy"] for m in fold_metrics])),
        "n_rows": len(train), "n_splits": n_splits, "folds": fold_metrics,
        "synthetic_data": bool(train["id"].str.startswith("synthetic-").all()),
    }
    _write_json(output / "metrics.json", metrics)
    metadata = _metadata(train, config)
    metadata["validation"] = manifest_metadata
    _write_json(output / "metadata.json", metadata)
    return metrics


def train_baseline(train: pd.DataFrame, output_dir: str | Path, config: BaselineConfig | None = None) -> Path:
    train = validate_train(train)
    config = config or BaselineConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline(config)
    pipeline.fit(train["text"], train["model"])
    # Fail early if an accidental training subset omitted or reordered labels.
    aligned_probabilities(pipeline, train["text"].iloc[:1])
    metadata = _metadata(train, config)
    metadata["estimator_classes"] = list(pipeline.named_steps["classifier"].classes_)
    model_path = output / "model.joblib"
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, model_path)
    _write_json(output / "metadata.json", metadata)
    return model_path


def predict_baseline(model_path: str | Path, test: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    test = validate_test(test)
    # joblib loads executable Python objects: only load artifacts you created/trust.
    artifact = joblib.load(model_path)
    metadata = artifact.get("metadata", {})
    if metadata.get("schema_version") != 1 or metadata.get("labels") != list(LABELS):
        raise ValueError("Unsupported artifact schema or label order")
    probabilities = aligned_probabilities(artifact["pipeline"], test["text"])
    predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)].tolist()
    return write_submission(test["id"], predictions, output_path)
