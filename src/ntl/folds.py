"""Persisted, fingerprinted prompt-grouped cross-validation assignments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .constants import DEFAULT_SEED, LABELS
from .data import fingerprint_frame, validate_train


def build_fold_manifest(
    train: pd.DataFrame, n_splits: int = 5, seed: int = DEFAULT_SEED
) -> tuple[pd.DataFrame, dict]:
    train = validate_train(train)
    if n_splits < 2:
        raise ValueError("Cross-validation requires at least two folds")
    if train["prompt_id"].nunique() < n_splits:
        raise ValueError("There are fewer unique prompts than requested folds")
    prompt_counts = train.groupby("model")["prompt_id"].nunique()
    if (prompt_counts < n_splits).any():
        raise ValueError("Each model must appear in at least n_splits different prompts")
    # Sorting makes this stable even if a CSV is loaded in a different row order.
    ordered = train.sort_values(["prompt_id", "id"], kind="stable").reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignments = np.full(len(ordered), -1, dtype=np.int64)
    for fold, (_, valid_index) in enumerate(
        splitter.split(ordered["text"], ordered["model"], groups=ordered["prompt_id"])
    ):
        assignments[valid_index] = fold
    manifest = ordered[["id", "prompt_id"]].assign(fold=assignments).sort_values("id").reset_index(drop=True)
    metadata = {
        "schema_version": 1,
        "algorithm": "StratifiedGroupKFold",
        "group_column": "prompt_id",
        "n_splits": n_splits,
        "seed": seed,
        "train_fingerprint": fingerprint_frame(train),
        "n_rows": len(train),
        "n_prompts": int(train["prompt_id"].nunique()),
    }
    validate_fold_manifest(train, manifest, metadata)
    return manifest, metadata


def validate_fold_manifest(train: pd.DataFrame, manifest: pd.DataFrame, metadata: dict) -> None:
    train = validate_train(train)
    if metadata.get("schema_version") != 1 or metadata.get("group_column") != "prompt_id":
        raise ValueError("Unsupported fold manifest schema or grouping column")
    if metadata.get("train_fingerprint") != fingerprint_frame(train):
        raise ValueError("Fold manifest fingerprint differs from the training data")
    if set(manifest.columns) != {"id", "prompt_id", "fold"}:
        raise ValueError("Fold manifest must have id, prompt_id and fold columns")
    if manifest.isna().any().any() or manifest["id"].duplicated().any():
        raise ValueError("Fold manifest contains nulls or duplicate ids")
    if len(manifest) != len(train) or set(manifest["id"]) != set(train["id"]):
        raise ValueError("Fold manifest must cover every training id exactly once")
    n_splits = metadata.get("n_splits")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("Invalid number of folds in manifest metadata")
    fold_values = manifest["fold"]
    if not pd.api.types.is_integer_dtype(fold_values):
        raise ValueError("Fold assignments must be integers")
    if set(fold_values) != set(range(n_splits)):
        raise ValueError("Fold assignments must cover exactly 0 through n_splits - 1")
    aligned = train.merge(manifest, on="id", suffixes=("", "_manifest"), validate="one_to_one")
    if not (aligned["prompt_id"] == aligned["prompt_id_manifest"]).all():
        raise ValueError("Fold manifest prompt ids do not match the training data")
    if aligned.groupby("prompt_id")["fold"].nunique().max() != 1:
        raise ValueError("Prompt leakage: a prompt occurs in more than one fold")
    for fold in range(n_splits):
        fit = aligned.loc[aligned["fold"] != fold]
        valid = aligned.loc[aligned["fold"] == fold]
        if set(fit["prompt_id"]) & set(valid["prompt_id"]):
            raise ValueError("Prompt leakage between fitting and validation rows")
        if set(fit["model"]) != set(LABELS) or set(valid["model"]) != set(LABELS):
            raise ValueError(f"Fold {fold} does not include all labels in both training and validation")


def save_fold_manifest(manifest: pd.DataFrame, metadata: dict, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(directory / "folds.csv", index=False)
    (directory / "folds.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return directory / "folds.csv"


def load_fold_manifest(train: pd.DataFrame, path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if path.is_dir():
        path = path / "folds.csv"
    manifest = pd.read_csv(path, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    metadata = json.loads(path.with_suffix(".json").read_text())
    validate_fold_manifest(train, manifest, metadata)
    return manifest, metadata
