"""Adapted-model comparison: inner-only blend weights, full-row scoring, exact v = 0 fallback."""

import numpy as np
import pandas as pd
import pytest
from test_lora import save_tokenizer

from ntl import ensemble, lora, lora_compare, neural
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.modeling import BaselineConfig
from ntl.smoke import synthetic_frames


@pytest.fixture
def setup(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(neural, "validate_snapshot", lambda *_, **__: {"revision": "synthetic"})
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    manifest_path = save_fold_manifest(manifest, metadata, tmp_path / "folds")
    features = np.random.default_rng(3).normal(size=(len(train), 8)).astype(np.float32)
    features[:, 0] += ensemble._targets(train)  # some signal for every head
    config = ensemble.BlendConfig(cpu=BaselineConfig(char_max_features=300, word_max_features=100, max_iter=200),
                                  bootstrap_samples=20)
    ensemble.evaluate_blend(train, features, manifest_path, tmp_path / "current", config)
    adapter_config = lora.AdapterConfig(durations=(0.0, 1.0), learning_rates=(1e-3,), micro_batch=8,
                                        accumulation=1, warmup_steps=1)
    plan = lora.split_plan(train, manifest_path, adapter_config)
    experiment = lora.Experiment(train, features, adapters=False, model_path=save_tokenizer(tmp_path / "tokenizer"),
                                 config=adapter_config, frozen=neural.FrozenConfig(), device="cpu")
    for fold in range(5):
        lora.run_fold(train, plan, experiment, fold, tmp_path / "control" / f"fold{fold}")
    return train, features, plan, config, tmp_path


def test_compare_scores_every_row_and_chooses_weights_from_inner_rows(setup):
    train, features, plan, config, root = setup
    current = pd.read_parquet(root / "current" / "oof.parquet")
    result = lora_compare.compare(train, features, plan, current, {"control": root / "control"}, root / "compare",
                                  config, samples=30)
    assert set(result["systems"]) == {"current", "control", "control+current"}
    assert result["systems"]["current"]["balanced_accuracy"] == pytest.approx(
        ensemble.balanced_accuracy_score(current["true_model"], current["prediction::blend"]))
    predictions = pd.read_parquet(root / "compare" / "predictions.parquet")
    assert len(predictions) == len(train) and predictions["id"].is_unique
    for item in result["selection"]["control"]:
        assert item["weight"] in lora_compare.WEIGHTS
        if item["weight"] == 0:  # v = 0 must reproduce the current system exactly
            rows = predictions["outer_fold"] == item["outer_fold"]
            assert (predictions.loc[rows, "prediction::control+current"]
                    == predictions.loc[rows, "prediction::current"]).all()
    assert {"control_vs_current", "control+current_vs_current"} <= set(result["paired"])


def test_current_system_must_come_from_the_plans_outer_folds(setup):
    train, features, plan, config, root = setup
    current = pd.read_parquet(root / "current" / "oof.parquet")
    current["fold"] = (current["fold"] + 1) % 5
    with pytest.raises(ValueError, match="outer folds"):
        lora_compare.compare(train, features, plan, current, {"control": root / "control"}, root / "rotated",
                             config, samples=10)


def test_inner_predictions_must_cover_the_inner_rows(setup):
    train, features, plan, config, root = setup
    _, inner, _ = lora_compare.load_runs(root / "control", train)
    current = {0: (np.arange(3), np.full((3, len(LABELS)), 1 / len(LABELS)), 0.25)}
    with pytest.raises(ValueError, match="cover"):
        lora_compare.choose_weight(train, current, inner, 0, 0)
