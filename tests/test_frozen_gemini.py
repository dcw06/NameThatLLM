"""Nested readout leakage, immutable feature provenance and Gemini reporting."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ntl import frozen_gemini as frozen
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.gemini import GEMINI_LABELS, prepare_gemini_evaluation
from ntl.smoke import synthetic_frames


@pytest.fixture
def cohort(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    path = save_fold_manifest(manifest, metadata, tmp_path / "folds")
    baseline = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    baseline["predicted_model"] = baseline["true_model"]
    baseline["fold"] = baseline["id"].map(manifest.set_index("id")["fold"])
    index = baseline.index[baseline["true_model"].isin(GEMINI_LABELS)][0]
    baseline.loc[index, "predicted_model"] = "gpt-5.5"
    frame, protocol = prepare_gemini_evaluation(train, baseline, path)
    rng = np.random.default_rng(2026)
    features = rng.normal(size=(len(frame), 6)).astype(np.float32)
    features[:, 0] = np.arange(len(frame))
    return frame, protocol, features


def provenance():
    return {"model_id": "fixture/frozen-encoder", "model_revision": "a" * 40,
            "encoder_frozen": True,
            "feature_options": {"pooling": "mean", "max_length": 512, "normalization": "none"}}


def test_feature_archive_aligns_reordered_rows_and_detects_tampering(cohort, tmp_path):
    frame, _, features = cohort
    order = np.arange(len(frame))[::-1]
    path = tmp_path / "features.npz"
    frozen.save_feature_archive(path, frame.iloc[order]["id"].tolist(), features[order], provenance(),
                                texts=frame.iloc[order]["text"].tolist())
    loaded, metadata = frozen.load_feature_archive(path, frame)
    np.testing.assert_array_equal(loaded, features)
    assert metadata["aligned_embeddings_sha256"] == frozen.embeddings_sha256(features)
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["embeddings"] = payload["embeddings"].copy()
    payload["embeddings"][0, 0] += 1
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="array hash"):
        frozen.load_feature_archive(path, frame)


def test_feature_archive_rejects_wrong_text_ids_revision_and_nonfinite_values(cohort, tmp_path):
    frame, _, features = cohort
    path = tmp_path / "features.npz"
    frozen.save_feature_archive(path, frame["id"].tolist(), features, provenance(), texts=frame["text"].tolist())
    changed = frame.copy()
    changed.loc[0, "text"] += "changed"
    with pytest.raises(ValueError, match="input-text"):
        frozen.load_feature_archive(path, changed)
    changed = frame.copy()
    changed.loc[0, "id"] = "unrelated-id"
    with pytest.raises(ValueError, match="exactly cover"):
        frozen.load_feature_archive(path, changed)
    bad_metadata = provenance() | {"model_revision": "main"}
    with pytest.raises(ValueError, match="immutable"):
        frozen.save_feature_archive(path, frame["id"], features, bad_metadata, texts=frame["text"])
    broken = features.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        frozen.save_feature_archive(path, frame["id"], broken, provenance(), texts=frame["text"])


def test_nested_scalers_fit_only_inner_or_outer_training_rows(cohort, tmp_path, monkeypatch):
    frame, protocol, features = cohort
    scaler_inputs = []

    class RecordingScaler(StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            scaler_inputs.append(set(X[:, 0].tolist()))
            return super().fit(X, y, sample_weight=sample_weight)

    monkeypatch.setattr(frozen, "StandardScaler", RecordingScaler)
    config = frozen.FrozenGeminiConfig(C_grid=(0.01, 0.1), inner_folds=3)
    metrics = frozen.evaluate_embeddings(frame, features, tmp_path / "report", config,
                                         protocol_metadata=protocol)
    # Per outer fold: 2 C values * 3 inner fits followed by one outer refit.
    assert len(scaler_inputs) == 35
    for fold in range(5):
        outer_fit_ids = set(features[frame["fold"] != fold, 0].tolist())
        outer_valid_ids = set(features[frame["fold"] == fold, 0].tolist())
        fits = scaler_inputs[fold * 7:(fold + 1) * 7]
        assert fits[-1] == outer_fit_ids
        assert all(values.issubset(outer_fit_ids) and not values & outer_valid_ids for values in fits)
        assert all(len(values) < len(outer_fit_ids) for values in fits[:-1])
    nested = json.loads((tmp_path / "report" / "inner_cv.json").read_text())
    for outer in nested:
        excluded = set(frame.loc[frame["fold"] == outer["outer_fold"], "prompt_id"])
        for split in outer["inner_splits"]:
            assert not set(split["training_prompts"]) & set(split["validation_prompts"])
            assert not (set(split["training_prompts"]) | set(split["validation_prompts"])) & excluded
    oof = pd.read_parquet(tmp_path / "report" / "oof.parquet")
    assert len(oof) == len(frame) and oof["id"].is_unique
    assert len(pd.read_csv(tmp_path / "report" / "selected_predictions.csv")) == frame["selected"].sum()
    assert metrics["all_true_gemini"]["rows"] == len(frame)
    assert metrics["correctly_routed_gemini"]["rows"] == len(frame) - 1
    # Four of five outer fits include the CPU router's false-negative row.
    assert sum(item["outer_train_rows"] > item["outer_train_selected_rows"] for item in nested) == 4


def test_regularization_selection_uses_only_inner_scores_and_ties_choose_smaller_C(monkeypatch):
    # A classifier with equal scores for every C makes the deterministic tie rule observable.
    class ConstantClassifier:
        classes_ = np.arange(4)

    class ConstantPipeline:
        named_steps = {"classifier": ConstantClassifier()}

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.full((len(X), 4), 0.25)

    monkeypatch.setattr(frozen, "_pipeline", lambda C, config: ConstantPipeline())
    features = np.zeros((24, 2), dtype=np.float32)
    targets = np.tile(np.arange(4), 6)
    groups = np.repeat([f"prompt-{i}" for i in range(6)], 4)
    chosen, result = frozen._select_regularization(features, targets, groups,
                                                  frozen.FrozenGeminiConfig(C_grid=(10, .1, 1)), 0)
    assert chosen == 0.1
    assert all(candidate["mean_inner_balanced_accuracy"] == 0.25 for candidate in result["candidates"])
    assert "outer_validation_balanced_accuracy" not in result


def test_probability_columns_are_aligned_to_canonical_label_indices():
    class Classifier:
        classes_ = np.array([3, 1, 0, 2])

    class Estimator:
        named_steps = {"classifier": Classifier()}

        def predict_proba(self, features):
            return np.array([[.1, .2, .3, .4]])

    values = frozen._aligned_probabilities(Estimator(), np.zeros((1, 2)))
    np.testing.assert_array_equal(values, [[.3, .2, .4, .1]])
