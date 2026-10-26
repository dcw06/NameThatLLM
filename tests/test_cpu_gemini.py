"""CPU specialist/blend leakage, tie rules and exact end-to-end substitution."""

import json

import numpy as np
import pandas as pd
import pytest

from ntl import cpu_gemini
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.gemini import GEMINI_LABELS, prepare_gemini_evaluation
from ntl.modeling import BaselineConfig
from ntl.smoke import synthetic_frames


@pytest.fixture
def cohort(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    path = save_fold_manifest(manifest, metadata, tmp_path / "folds")
    baseline = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    baseline["predicted_model"] = baseline["true_model"]
    baseline["fold"] = baseline["id"].map(manifest.set_index("id")["fold"])
    gemini_rows = baseline.index[baseline["true_model"].isin(GEMINI_LABELS)]
    baseline.loc[gemini_rows[0], "predicted_model"] = "gpt-5.5"
    baseline.loc[baseline.index[baseline["true_model"] == "kimi-k3"][0], "predicted_model"] = GEMINI_LABELS[2]
    frame, protocol = prepare_gemini_evaluation(train, baseline, path)
    rng = np.random.default_rng(2026)
    features = rng.normal(size=(len(frame), 6)).astype(np.float32)
    return frame, protocol, features, baseline


def small_config():
    return cpu_gemini.SpecialistConfig(
        cpu=BaselineConfig(char_max_features=500, word_max_features=200, max_iter=300),
        weight_grid=(0.0, 0.5, 1.0), bootstrap_samples=50,
    )


def test_specialist_and_blend_fit_only_training_rows_and_score_all_rows(cohort, tmp_path, monkeypatch):
    frame, protocol, features, baseline = cohort
    fits = []
    original = cpu_gemini._cpu_probabilities

    def recording(fit_texts, fit_targets, texts, config):
        fits.append((set(fit_texts), set(texts)))
        return original(fit_texts, fit_targets, texts, config)

    monkeypatch.setattr(cpu_gemini, "_cpu_probabilities", recording)
    output = tmp_path / "report"
    comparison = cpu_gemini.evaluate_specialists(frame, features, baseline, output, small_config(),
                                                 protocol_metadata=protocol)
    # Per outer fold: three inner CPU fits, then one outer fit shared by the control and blend.
    assert len(fits) == 20
    for fold in range(5):
        outer_fit = set(frame.loc[frame["fold"] != fold, "text"])
        outer_valid = set(frame.loc[frame["fold"] == fold, "text"])
        for fitted, scored in fits[fold * 4:(fold + 1) * 4 - 1]:
            assert fitted < outer_fit and scored < outer_fit and not fitted & scored
        assert fits[fold * 4 + 3] == (outer_fit, outer_valid)
    nested = json.loads((output / "inner_cv.json").read_text())
    for outer in nested:
        excluded = set(frame.loc[frame["fold"] == outer["outer_fold"], "prompt_id"])
        for split in outer["inner_splits"]:
            assert not (set(split["training_prompts"]) | set(split["validation_prompts"])) & excluded
    for name in (cpu_gemini.CONTROL, cpu_gemini.BLEND):
        oof = pd.read_parquet(output / name / "oof.parquet")
        assert len(oof) == len(frame) and oof["id"].is_unique
    full = comparison["end_to_end"]
    assert full["rows"] == len(baseline)
    assert full["non_gemini_routed_to_gemini"] == 1 and full["true_gemini_routed_elsewhere"] == 1
    predictions = pd.read_parquet(output / "end_to_end_predictions.parquet")
    assert len(predictions) == len(baseline) and set(predictions["reference"]) <= set(LABELS)
    with pytest.raises(ValueError, match="new output directory"):
        cpu_gemini.evaluate_specialists(frame, features, baseline, output, small_config())


def test_blend_ties_choose_the_control_weight_then_smaller_C(monkeypatch):
    class Constant:
        named_steps = {"classifier": type("Classifier", (), {"classes_": np.arange(4)})()}

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.full((len(X), 4), 0.25)

    monkeypatch.setattr(cpu_gemini, "_pipeline", lambda C, config: Constant())
    monkeypatch.setattr(cpu_gemini, "_cpu_probabilities",
                        lambda fit_texts, fit_targets, texts, config: np.full((len(texts), 4), 0.25))
    targets = np.tile(np.arange(4), 6)
    prompts = np.repeat([f"prompt-{index}" for index in range(6)], 4)
    texts = np.asarray([f"text {index}" for index in range(24)], dtype=object)
    C, weight, record = cpu_gemini._select_blend(texts, np.zeros((24, 2)), targets, prompts,
                                                 cpu_gemini.SpecialistConfig(weight_grid=(1.0, 0.5, 0.0)), 0)
    assert (C, weight) == (0.001, 0.0)
    assert len(record["candidates"]) == 15


def test_end_to_end_replaces_only_correctly_routed_gemini_versions():
    baseline = pd.DataFrame({
        "id": ["a", "b", "c", "d"], "prompt_id": ["p"] * 4, "fold": [0] * 4,
        "true_model": [GEMINI_LABELS[0], GEMINI_LABELS[1], "kimi-k3", GEMINI_LABELS[2]],
        "predicted_model": [GEMINI_LABELS[1], GEMINI_LABELS[1], GEMINI_LABELS[0], "gpt-5.5"],
    })
    versions = pd.DataFrame({"id": ["a", "b", "d"], "selected": [True, True, False],
                             "prediction": [GEMINI_LABELS[0], GEMINI_LABELS[3], GEMINI_LABELS[2]]})
    summary, result = cpu_gemini.end_to_end(baseline, {"specialist": versions})
    assert result["specialist"].tolist() == [GEMINI_LABELS[0], GEMINI_LABELS[3], GEMINI_LABELS[0], "gpt-5.5"]
    assert summary["systems"]["reference"]["correct"] == 1
    assert summary["systems"]["specialist"]["correct"] == 1
    assert summary["non_gemini_routed_to_gemini"] == 1
    with pytest.raises(ValueError, match="exactly the true-Gemini rows routed"):
        cpu_gemini.end_to_end(baseline, {"specialist": versions.assign(selected=[True, False, False])})


def test_paired_bootstrap_resamples_prompts_and_counts_transitions():
    frame = pd.DataFrame({"prompt_id": ["p1", "p1", "p2", "p3"]})
    result = cpu_gemini._paired(frame, np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]), 200, 0)
    assert (result["both_correct"], result["second_only_correct"],
            result["first_only_correct"], result["both_wrong"]) == (1, 1, 1, 1)
    assert result["prompts"] == 3 and result["accuracy_delta"] == 0
    low, high = result["prompt_bootstrap"]["accuracy_delta_95_interval"]
    assert low <= 0 <= high
