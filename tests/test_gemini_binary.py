"""Binary scoring experiment: training-only standardization, whole-prompt folds, saved weights
that reproduce the scores, and the corrected-versus-lost comparison."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ntl import gemini_binary, gemini_markers

PAIR = gemini_markers.PAIR


def _frame(prompts=10, seed=0):
    """Each prompt has one passage per version and one from another model; 3.8 leans on markers."""
    rng = np.random.default_rng(seed)
    rows = []
    for prompt in range(prompts):
        for model in (*PAIR, "kimi-k3"):
            words = ["word"] * int(rng.integers(40, 80))
            if model == PAIR[1] and rng.random() < 0.7:
                words += ["shatter;", "void:"]
            elif model == PAIR[0] and rng.random() < 0.6:
                words += ["merely", "thus"]
            rows.append({"id": f"{prompt}-{model}", "prompt_id": f"p{prompt}", "model": model,
                         "text": " ".join(words) + "."})
    frame = pd.DataFrame(rows)
    folds = frame["prompt_id"].str[1:].astype(int).to_numpy() % 5
    return frame, folds, gemini_markers.feature_table(frame)


def test_each_fold_fits_its_scaler_and_model_on_training_pair_rows_only():
    frame, folds, features = _frame()
    scores, records = gemini_binary.cross_validate(frame, folds, features)
    pair = frame["model"].isin(PAIR).to_numpy()
    assert np.isnan(scores[~pair]).all() and np.isfinite(scores[pair]).all()
    values = features[list(gemini_binary.FEATURES)].to_numpy(float)
    target = (frame["model"] == PAIR[1]).to_numpy().astype(int)
    for record in records:
        fold = record["outer_fold"]
        rows, held = np.flatnonzero(pair & (folds != fold)), np.flatnonzero(pair & (folds == fold))
        assert record["training_rows"] == len(rows) == 16 and record["held_out_rows"] == len(held) == 4
        manual = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, tol=1e-6, random_state=2026))
        np.testing.assert_allclose(scores[held], manual.fit(values[rows], target[rows]).predict_proba(values[held])[:, 1])
        # The saved weights, in either unit, reproduce the held-out scores.
        rates = values[held]
        raw = record["intercept_per_100_words"] + rates @ np.array(list(record["per_100_words"].values()))
        np.testing.assert_allclose(scores[held], 1 / (1 + np.exp(-raw)), rtol=0, atol=1e-12)
        standard = (rates - np.array(list(record["scaler_mean"].values()))) / np.array(
            list(record["scaler_scale"].values()))
        logit = record["intercept"] + standard @ np.array(list(record["standardized"].values()))
        np.testing.assert_allclose(scores[held], 1 / (1 + np.exp(-logit)), rtol=0, atol=1e-12)
    shifted = features.copy()
    held = np.flatnonzero(pair & (folds == 0))
    shifted.loc[held, list(gemini_binary.FEATURES)] += 50  # held-out rows never reach the scaler
    again, records_again = gemini_binary.cross_validate(frame, folds, shifted)
    assert records_again[0]["scaler_mean"] == records[0]["scaler_mean"]
    assert records_again[0]["standardized"] == records[0]["standardized"]


def test_a_prompt_split_across_folds_or_a_single_version_is_refused():
    frame, folds, features = _frame()
    split = folds.copy()
    split[np.flatnonzero(frame["model"] == PAIR[0])[0]] = (folds[0] + 1) % 5
    with pytest.raises(ValueError, match="share a fold"):
        gemini_binary.cross_validate(frame, split, features)
    values, target = np.random.default_rng(1).normal(size=(6, 7)), np.zeros(6, dtype=int)
    with pytest.raises(ValueError, match="Both Gemini versions"):
        gemini_binary.fit(values, target, np.arange(6))


def test_a_convergence_warning_fails_the_fit(monkeypatch):
    values, target = np.random.default_rng(2).normal(size=(20, 7)), np.tile([0, 1], 10)
    original = LogisticRegression.fit

    def warning(self, *args, **kwargs):
        warnings.warn("did not converge", ConvergenceWarning)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", warning)
    with pytest.raises(ConvergenceWarning):
        gemini_binary.fit(values, target, np.arange(20))


def test_comparison_counts_corrections_losses_and_the_threshold():
    target = np.array([0, 0, 1, 1, 1, 0])
    scores = np.array([0.5, 0.7, 0.51, 0.2, 0.9, 0.1])  # exactly 0.5 predicts 3.7
    incumbent = np.array([0.6, 0.4, 0.8, 0.8, 0.9, 0.3])
    result = gemini_binary.compare(target, scores, incumbent)
    assert result["model"]["correct"] == 4 and result["incumbent_forced"]["correct"] == 5
    assert result["model"]["recall"] == {PAIR[0]: 2 / 3, PAIR[1]: 2 / 3}
    assert result["errors_corrected"] == 1 and result["correct_lost"] == 2  # rows 0; rows 1 and 3
    assert result["both_correct"] == 3 and result["both_wrong"] == 0 and result["predicted_3.8"] == 3
    probabilities = np.zeros((2, 18))
    probabilities[:, gemini_markers.PAIR_INDEX] = [[0.2, 0.6], [0.3, 0.1]]
    np.testing.assert_allclose(gemini_binary.forced_share(probabilities), [0.75, 0.25])
