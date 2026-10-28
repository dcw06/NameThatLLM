"""Frozen Qwen archives, fold-local heads and the nested 18-class CPU-Qwen blend."""

import json

import numpy as np
import pandas as pd
import pytest

from ntl import ensemble, neural
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.modeling import BaselineConfig
from ntl.smoke import synthetic_frames


@pytest.fixture
def official(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    path = save_fold_manifest(manifest, metadata, tmp_path / "folds")
    features = np.random.default_rng(2026).normal(size=(len(train), 8)).astype(np.float32)
    features[:, 0] = np.arange(len(train))
    reference = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    reference["fold"] = reference["id"].map(manifest.set_index("id")["fold"])
    reference["predicted_model"] = reference["true_model"].where(reference.index % 3 != 0, LABELS[0])
    return train, features, path, reference


def small_config(**changes):
    values = {"cpu": BaselineConfig(char_max_features=500, word_max_features=200, max_iter=300),
              "bootstrap_samples": 50, **changes}
    return ensemble.BlendConfig(**values)


def test_blend_refits_both_models_inside_inner_and_outer_training_rows(official, tmp_path, monkeypatch):
    train, features, path, reference = official
    cpu_fits, qwen_fits = [], []
    cpu_original, qwen_original = ensemble._cpu_probabilities, ensemble._qwen_probabilities

    def cpu(fit_texts, fit_labels, texts, config):
        cpu_fits.append((set(fit_texts), set(texts)))
        return cpu_original(fit_texts, fit_labels, texts, config)

    def qwen(fit_features, fit_targets, rows, config):
        qwen_fits.append((set(fit_features[:, 0].tolist()), set(rows[:, 0].tolist())))
        return qwen_original(fit_features, fit_targets, rows, config)

    monkeypatch.setattr(ensemble, "_cpu_probabilities", cpu)
    monkeypatch.setattr(ensemble, "_qwen_probabilities", qwen)
    output = tmp_path / "blend"
    comparison = ensemble.evaluate_blend(train, features, path, output, small_config(), reference)
    folds = pd.read_csv(path, dtype={"id": str}).set_index("id")["fold"]
    assignments = train["id"].map(folds).to_numpy()
    # Per outer fold: three inner fits of each model, then one outer fit of each.
    assert len(cpu_fits) == len(qwen_fits) == 20
    for fold in range(5):
        fit_rows, valid_rows = set(np.flatnonzero(assignments != fold)), set(np.flatnonzero(assignments == fold))
        fit_texts, valid_texts = set(train["text"].iloc[list(fit_rows)]), set(train["text"].iloc[list(valid_rows)])
        for fitted, scored in cpu_fits[fold * 4:fold * 4 + 3]:
            assert fitted < fit_texts and scored < fit_texts and not fitted & scored
        for fitted, scored in qwen_fits[fold * 4:fold * 4 + 3]:
            assert fitted < fit_rows and scored < fit_rows and not fitted & scored
        assert cpu_fits[fold * 4 + 3] == (fit_texts, valid_texts)
        assert qwen_fits[fold * 4 + 3] == (fit_rows, valid_rows)
    oof = pd.read_parquet(output / "oof.parquet")
    assert len(oof) == len(train) and oof["id"].is_unique
    assert {f"blend::{label}" for label in LABELS} <= set(oof.columns)
    for name in ("cpu", "qwen", "blend"):
        np.testing.assert_allclose(oof[[f"{name}::{label}" for label in LABELS]].sum(axis=1), 1.0)
    assert set(comparison["systems"]) == {"cpu", "qwen", "blend", "recorded_reference"}
    assert {"blend_vs_cpu", "qwen_vs_cpu", "blend_vs_recorded_reference"} <= set(comparison["paired"])
    assert set(comparison["gate"]) >= {"passed", "zero_weight_folds", "folds_blend_beats_same_environment_cpu"}
    nested = json.loads((output / "inner_cv.json").read_text())
    for outer in nested:
        excluded = set(train.loc[assignments == outer["outer_fold"], "prompt_id"])
        assert all(not (set(split["training_prompts"]) | set(split["validation_prompts"])) & excluded
                   for split in outer["inner_splits"])
    with pytest.raises(ValueError, match="new output directory"):
        ensemble.evaluate_blend(train, features, path, output, small_config())


def test_weight_ties_keep_the_cpu_only_system(monkeypatch):
    uniform = lambda *args: np.full((len(args[2]), len(LABELS)), 1 / len(LABELS))  # noqa: E731
    monkeypatch.setattr(ensemble, "_cpu_probabilities", uniform)
    monkeypatch.setattr(ensemble, "_qwen_probabilities", uniform)
    targets = np.tile(np.arange(len(LABELS)), 6)
    prompts = np.repeat([f"prompt-{index}" for index in range(6)], len(LABELS))
    texts = np.asarray([f"text {index}" for index in range(len(targets))], dtype=object)
    weight, record = ensemble._select_weight(texts, np.asarray(LABELS)[targets], np.zeros((len(targets), 2)),
                                             targets, prompts, small_config(weight_grid=(1.0, 0.5, 0.0)), 0)
    assert weight == 0.0 and len(record["candidates"]) == 3
    assert "outer_balanced_accuracy" not in record


def test_prompt_bootstrap_reports_balanced_accuracy_change_and_paired_counts():
    truth = np.tile(np.arange(len(LABELS)), 4)
    prompts = np.repeat(["a", "b", "c", "d"], len(LABELS))
    first = truth.copy()
    first[:5] = (truth[:5] + 1) % len(LABELS)
    same = ensemble.prompt_bootstrap(truth, first, first, prompts, 100, 0)
    assert same["balanced_accuracy_delta"] == 0 and same["delta_95_interval"] == [0.0, 0.0]
    better = ensemble.prompt_bootstrap(truth, first, truth, prompts, 100, 0)
    assert better["balanced_accuracy_delta"] == pytest.approx(5 / len(truth))
    assert better["second_only_correct"] == 5 and better["first_only_correct"] == 0
    assert better["delta_95_interval"][0] >= 0 and better["fraction_delta_at_most_zero"] < 1


def test_prompt_bootstrap_counts_exact_ties_as_zero_despite_float_rounding():
    # Three classes of ten passages in two prompts; every changed passage is in prompt "a".
    # The second system gains 1 and 2 passages in two classes and loses 3 in the third, so
    # every resample changes per-class recall by k/10 + 2k/10 - 3k/10 = 0 exactly, which
    # floating point alone would report as up to about 4e-17.
    truth = np.repeat([0, 1, 2], 10)
    prompts = np.tile(np.repeat(["a", "b"], 5), 3)
    first, second = truth.copy(), truth.copy()
    first[[0, 10, 11]] = 2
    second[[20, 21, 22]] = 0
    result = ensemble.prompt_bootstrap(truth, first, second, prompts, 1000, 0)
    assert result["balanced_accuracy_delta"] == 0.0
    assert result["delta_95_interval"] == [0.0, 0.0] and result["fraction_delta_at_most_zero"] == 1.0
    assert result["second_only_correct"] == result["first_only_correct"] == 3


def test_archive_records_and_enforces_the_frozen_recipe(official, tmp_path, monkeypatch):
    train, features, _, _ = official
    config = neural.FrozenConfig()
    wide = np.random.default_rng(0).normal(size=(len(train), 4096 * 2)).astype(np.float32)
    monkeypatch.setattr(neural, "extract_features", lambda texts, path, config: (wide, {"rows": len(texts)}))
    path = tmp_path / "features.npz"
    saved = ensemble.extract_archive(train, "unused", path, config)
    assert saved["feature_options"]["pool_layers"] == [7, 22]
    assert saved["feature_options"]["quantization"] == "nf4"
    assert json.loads(path.with_suffix(".json").read_text())["train_fingerprint"] == saved["train_fingerprint"]
    loaded, _ = ensemble.load_archive(path, train.sample(frac=1, random_state=3), config)
    assert loaded.shape == wide.shape
    with pytest.raises(ValueError, match="pool_layers"):
        ensemble.load_archive(path, train, neural.FrozenConfig(pool_layers=(8, 22)))
    with pytest.raises(ValueError, match="already exists"):
        ensemble.extract_archive(train, "unused", path, config)


def test_qwen_oof_scores_every_row_once_with_fold_local_heads(official, tmp_path):
    train, features, path, _ = official
    metrics = ensemble.qwen_oof(train, features, path, tmp_path / "oof", neural.FrozenConfig())
    oof = pd.read_parquet(tmp_path / "oof" / "oof.parquet")
    assert len(oof) == len(train) and oof["id"].is_unique
    np.testing.assert_allclose(oof[[f"probability::{label}" for label in LABELS]].sum(axis=1), 1.0)
    assert len(metrics["fold_balanced_accuracy"]) == 5 and set(metrics["recall"]) == set(LABELS)


def fake_features(texts, path, config):
    # Deterministic per-text vectors so refits and predictions are reproducible.
    rows = [np.random.default_rng(abs(hash(text)) % 2**32).normal(size=4096 * 2) for text in texts]
    return np.asarray(rows, dtype=np.float32), {"rows": len(texts)}


def test_freeze_fit_and_predict_use_the_frozen_recipe_on_every_row(official, tmp_path, monkeypatch):
    train, features, _, _ = official
    monkeypatch.setattr(neural, "extract_features", fake_features)
    config = small_config(qwen=neural.FrozenConfig())
    recipe = ensemble.freeze_recipe(train, features, config)
    assert recipe["weight"] in config.weight_grid
    assert recipe["weight_selection"]["outer_fold"] == ensemble.FINAL_SELECTION_SEED_OFFSET
    fitted = ensemble.fit_final(train, "unused", recipe, tmp_path / "model", device="cpu")
    assert fitted["n_train"] == len(train) and fitted["weight"] == recipe["weight"]
    test = train[["id", "text"]].sample(n=7, random_state=1).reset_index(drop=True)
    test["id"] = [f"hidden-{index}" for index in range(len(test))]
    submission = ensemble.predict_final(tmp_path / "model", test, "unused", tmp_path / "submission.parquet")
    assert submission["id"].tolist() == test["id"].tolist()
    assert set(submission["model"]) <= set(LABELS)
    assert json.loads((tmp_path / "submission_profile.json").read_text())["rows"] == 7
    with pytest.raises(ValueError, match="must not be overwritten"):
        ensemble.fit_final(train, "unused", recipe, tmp_path / "model")
    with pytest.raises(ValueError, match="Training data differ"):
        ensemble.fit_final(train.iloc[1:], "unused", recipe, tmp_path / "other")
    for broken in ({**recipe, "weight": 1.5}, {**recipe, "labels": list(reversed(LABELS))},
                   {**recipe, "feature_options": {**recipe["feature_options"], "pool_layers": [8, 22]}}):
        with pytest.raises(ValueError):
            ensemble.load_recipe(broken)
