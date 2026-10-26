"""Strict schemas and content fingerprints. Text is never normalized here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .constants import LABELS

TRAIN_COLUMNS = ("id", "prompt_id", "text", "model")
TEST_COLUMNS = ("id", "text")


def _strings(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed")
    for column in columns:
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"{column} contains null values")
        if not values.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain strings")
        if values.map(lambda value: not value.strip()).any():
            raise ValueError(f"{column} contains empty or whitespace-only strings")
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate passage ids are not allowed")


def validate_train(frame: pd.DataFrame, require_all_labels: bool = True) -> pd.DataFrame:
    """Validate labeled training rows without changing their text or order."""
    _strings(frame, TRAIN_COLUMNS)
    if frame.empty:
        raise ValueError("Training data is empty")
    observed = set(frame["model"])
    unknown = observed - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown model labels: {sorted(unknown)}")
    if require_all_labels and observed != set(LABELS):
        raise ValueError(f"Training data is missing model labels: {sorted(set(LABELS) - observed)}")
    return frame.copy()


def validate_test(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the hidden-test schema and reject accidental target leakage."""
    _strings(frame, TEST_COLUMNS)
    if frame.empty:
        raise ValueError("Test data is empty; expected at least one passage")
    forbidden = {"model", "prompt_id"} & set(frame.columns)
    if forbidden:
        raise ValueError(f"Test data must not contain training-only columns: {sorted(forbidden)}")
    return frame.copy()


def data_quality_report(train: pd.DataFrame) -> dict:
    """Describe identical outputs without rejecting legitimate shared responses.

    Text shared between prompts can make prompt-grouped validation optimistic;
    cross-label identical outputs can also impose an attribution ambiguity.
    Neither proves a schema error, so report both rather than removing rows.
    """
    train = validate_train(train)
    grouped = train.groupby("text", sort=False).agg(
        rows=("id", "size"), labels=("model", "nunique"), prompts=("prompt_id", "nunique")
    )
    repeated = grouped["rows"] > 1
    ambiguous = grouped["labels"] > 1
    cross_prompt = grouped["prompts"] > 1
    warnings = []
    if cross_prompt.any():
        warnings.append(
            "Identical text appears under multiple prompt_ids; prompt-grouped validation may be optimistic."
        )
    if ambiguous.any():
        warnings.append("Identical text has multiple model labels; these rows may be intrinsically ambiguous.")
    return {
        "unique_texts": len(grouped),
        "exact_duplicate_text_groups": int(repeated.sum()),
        "exact_duplicate_text_rows": int(grouped.loc[repeated, "rows"].sum()),
        "exact_duplicate_text_excess_rows": int((grouped.loc[repeated, "rows"] - 1).sum()),
        "cross_label_identical_text_groups": int(ambiguous.sum()),
        "cross_label_identical_text_rows": int(grouped.loc[ambiguous, "rows"].sum()),
        "cross_prompt_identical_text_groups": int(cross_prompt.sum()),
        "cross_prompt_identical_text_rows": int(grouped.loc[cross_prompt, "rows"].sum()),
        "warnings": warnings,
    }


def _read_csv(path: str | Path) -> pd.DataFrame:
    # keep_default_na=False preserves legitimate text such as "NA" or "null".
    # Empty cells are rejected by the schema, rather than silently imputed.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_train(path: str | Path, require_all_labels: bool = True) -> pd.DataFrame:
    return validate_train(_read_csv(path), require_all_labels=require_all_labels)


def read_test(path: str | Path) -> pd.DataFrame:
    return validate_test(_read_csv(path))


def fingerprint_frame(frame: pd.DataFrame) -> str:
    """SHA-256 of exact required training content, invariant to CSV row order."""
    validated = validate_train(frame, require_all_labels=False)
    digest = hashlib.sha256()
    digest.update(json.dumps(TRAIN_COLUMNS).encode("utf-8"))
    ordered = validated.sort_values("id", kind="stable")
    for row in ordered.loc[:, list(TRAIN_COLUMNS)].itertuples(index=False, name=None):
        digest.update(b"\n")
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def validate_submission(
    submission: pd.DataFrame,
    expected_ids: Sequence[str] | pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Check columns, labels and exact id coverage, independent of row order."""
    if set(submission.columns) != {"id", "model"}:
        raise ValueError("Submission must contain exactly the id and model columns")
    _strings(submission, ("id", "model"))
    if submission.empty:
        raise ValueError("Submission is empty")
    if isinstance(expected_ids, pd.DataFrame):
        expected_ids = expected_ids["id"]
    expected = list(expected_ids)
    if any(not isinstance(value, str) or not value.strip() for value in expected):
        raise ValueError("Expected ids must be nonempty strings")
    if len(set(expected)) != len(expected):
        raise ValueError("Expected ids contain duplicates")
    if len(submission) != len(expected) or set(submission["id"]) != set(expected):
        raise ValueError("Submission ids do not exactly match the test ids")
    unknown = set(submission["model"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Submission contains unknown labels: {sorted(unknown)}")
    return submission.copy()


def write_submission(
    ids: Sequence[str] | pd.Series,
    predictions: Sequence[str],
    path: str | Path,
) -> pd.DataFrame:
    ids = list(ids)
    predictions = list(predictions)
    if len(ids) != len(predictions):
        raise ValueError("Prediction count does not match test row count")
    submission = pd.DataFrame({"id": ids, "model": predictions})
    validate_submission(submission, ids)
    destination = Path(path)
    if destination.suffix != ".parquet":
        raise ValueError("Competition submissions must be Parquet files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(destination, index=False)
    validate_submission(pd.read_parquet(destination), ids)
    return submission
