"""Clearly synthetic fixtures for exercising plumbing, never competition scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .constants import LABELS
from .modeling import BaselineConfig, predict_baseline, run_cross_validation, train_baseline


def synthetic_frames(n_prompts: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Artificial class markers make these data unsuitable for modeling claims."""
    rows = []
    for prompt in range(n_prompts):
        for model_index, label in enumerate(LABELS):
            rows.append({
                "id": f"synthetic-{prompt:03d}-{model_index:02d}",
                "prompt_id": f"synthetic-prompt-{prompt:03d}",
                "text": (f"SYNTHETIC FIXTURE ONLY. ClassMarker{model_index:02d} "
                         f"ClassMarker{model_index:02d}. Prompt theme {prompt}; "
                         "Case, punctuation—‘quotes’—and line breaks\nare preserved. "
                         "This manufactured sample is not an authentic model response."),
                "model": label,
            })
    train = pd.DataFrame(rows)
    test = pd.DataFrame({
        "id": ["synthetic-test-000", "synthetic-test-001", "synthetic-test-002"],
        "text": [rows[index]["text"] for index in (0, 7, 17)],
    })
    return train, test


def run_smoke(output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train, test = synthetic_frames()
    train.to_csv(output / "synthetic_train.csv", index=False)
    test.to_csv(output / "synthetic_test.csv", index=False)
    config = BaselineConfig(char_max_features=3000, word_max_features=1000, max_iter=300)
    cv = run_cross_validation(train, output / "cv", n_splits=5, config=config)
    model_path = train_baseline(train, output / "model", config=config)
    submission = predict_baseline(model_path, test, output / "submission.parquet")
    summary = {"synthetic_data": True, "purpose": "Pipeline plumbing only; scores have no competition meaning.",
               "training_rows": len(train), "test_rows": len(test), "submission_rows": len(submission),
               "cv_completed": True, "model_saved": str(model_path), "cv": cv}
    (output / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
