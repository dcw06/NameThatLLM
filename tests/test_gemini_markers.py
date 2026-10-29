"""Marker study: the frozen extractor, isolated fits, the declared selection order and the
pair-mass invariants, on synthetic inputs."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ntl import claude_glm, ensemble, gemini_markers, gemini_pair, lora, neural
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.selection import first_best
from ntl.smoke import synthetic_frames

PAIR_INDEX = gemini_pair.PAIR_INDEX


def test_the_extractor_counts_complete_casefolded_tokens_phrases_and_ascii_punctuation():
    text = "Shatter SHATTER shattered shatter-proof demands demand; Thus: thus. Tears, tears’s."
    result = gemini_markers.extract(text)
    assert result["words"] == 10  # shatter-proof and tears’s are single tokens
    assert result["count_action"] == 3 and result["count_formal"] == 2 and result["count_sensory"] == 1
    assert result["count_semicolons"] == 1 and result["count_colons"] == 1
    assert result["rate_action"] == pytest.approx(100 * 3 / 10) and result["log_words"] == pytest.approx(np.log(10))
    phrases = "not simply. not\tsimply, not simply not, simply not-simply not​simply Not  Simply"
    assert gemini_markers.extract(phrases)["count_qualifying"] == 4
    assert gemini_markers.extract("a；b：c merely merely")["count_semicolons"] == 0  # fullwidth is not converted
    assert gemini_markers.extract("a；b：c merely merely")["count_qualifying"] == 2
    empty = gemini_markers.extract("; :")
    assert empty["words"] == 0 and "log_words" not in empty and "rate_semicolons" not in empty


def test_length_bins_and_the_w_requirement():
    assert [gemini_markers.length_bin(w) for w in (59, 60, 99, 100, 149, 150)] == \
        ["<60", "60-99", "60-99", "100-149", "100-149", ">=150"]
    with pytest.raises(ValueError, match="W > 0"):
        gemini_markers.feature_table(pd.DataFrame({"id": ["a"], "text": ["; :"]}))


def test_fits_use_only_their_fitting_rows_and_fail_on_convergence_warnings(monkeypatch):
    rng = np.random.default_rng(0)
    values, target = rng.normal(size=(40, 3)), np.tile([0, 1], 20)
    values[:, 1] = 0.0  # a constant column is scaled by 1
    values[:, 0] += target
    fit, rows = np.arange(30), np.arange(30, 40)
    scores, record = gemini_markers.fit_scores(values, target, fit, rows, 0.1)
    manual = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, tol=1e-6, random_state=2026))
    np.testing.assert_allclose(scores, manual.fit(values[fit], target[fit]).predict_proba(values[rows])[:, 1])
    assert record["converged"] and record["fit_rows"] == 30
    shifted = values.copy()
    shifted[rows] += 100  # scored rows never enter the scaler or the fit
    moved, _ = gemini_markers.fit_scores(shifted, target, fit, fit[:5], 0.1)
    np.testing.assert_allclose(moved, manual.predict_proba(values[fit[:5]])[:, 1])
    with pytest.raises(ValueError, match="Both Gemini versions"):
        gemini_markers.fit_scores(values, np.zeros(40, dtype=int), fit, rows, 0.1)
    original = LogisticRegression.fit

    def warning(self, *args, **kwargs):
        warnings.warn("did not converge", ConvergenceWarning)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", warning)
    with pytest.raises(ConvergenceWarning):
        gemini_markers.fit_scores(values, target, fit, rows, 0.1)


def test_integration_keeps_mass_untouched_columns_and_returns_the_incumbent_at_zero():
    rng = np.random.default_rng(1)
    incumbent = rng.dirichlet(np.ones(len(LABELS)), size=6)
    scores = np.array([0.1, 0.5, 0.9, 0.3, np.nan, 0.7])
    assert gemini_markers.integrate(incumbent, scores, 0.0) is incumbent
    result = gemini_markers.integrate(incumbent, scores, 0.25)
    mass = incumbent[:, PAIR_INDEX].sum(axis=1)
    eligible = np.isfinite(scores)
    np.testing.assert_allclose(result[eligible, PAIR_INDEX[1]],
                               0.75 * incumbent[eligible, PAIR_INDEX[1]] + 0.25 * mass[eligible] * scores[eligible],
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(result[eligible, PAIR_INDEX[0]],
                               0.75 * incumbent[eligible, PAIR_INDEX[0]]
                               + 0.25 * mass[eligible] * (1 - scores[eligible]), rtol=0, atol=1e-15)
    others = np.setdiff1d(np.arange(len(LABELS)), PAIR_INDEX)
    assert np.array_equal(result[:, others], incumbent[:, others]) and np.array_equal(result[4], incumbent[4])
    zero = incumbent.copy()
    zero[0, PAIR_INDEX] = 0
    zero[0] /= zero[0].sum()
    share, count = gemini_markers.binary_share(zero)
    assert count == 1 and share[0] == 0.5
    assert np.array_equal(gemini_markers.integrate(zero, scores, 0.5)[0, PAIR_INDEX], [0.0, 0.0])


@pytest.fixture
def study(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    truth = ensemble._targets(train)
    rng = np.random.default_rng(3)
    pair = np.isin(truth, PAIR_INDEX)
    texts = train["text"].to_numpy(dtype=object)
    for row in np.flatnonzero(pair):  # 3.8 tends to "thus" and semicolons, imperfectly
        if (truth[row] == PAIR_INDEX[1]) == (rng.random() < 0.7):
            texts[row] = texts[row] + " Thus; it holds."
    train = train.assign(text=texts)
    manifest, metadata = build_fold_manifest(train)
    plan = lora.split_plan(train, save_fold_manifest(manifest, metadata, tmp_path / "folds"), lora.AdapterConfig())
    blocks = {"qwen": rng.normal(size=(len(train), 8)), "styledistance": rng.normal(size=(len(train), 4))}
    blocks["qwen"][:, 0] += 0.3 * truth
    folds = plan["outer_fold"].to_numpy()

    def probabilities(rows):
        logits = rng.normal(size=(len(rows), len(LABELS)))
        logits[np.arange(len(rows)), truth[rows]] += 1.5
        swap = pair[rows] & (rng.random(len(rows)) < 0.5)  # the pair often swaps
        logits[np.ix_(swap, PAIR_INDEX)] = logits[np.ix_(swap, PAIR_INDEX[::-1])]
        exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exponent / exponent.sum(axis=1, keepdims=True)

    current = probabilities(np.arange(len(train)))
    frames = []
    for fold in range(5):
        column = plan[f"inner_{fold}"].to_numpy()
        for inner_fold in (0, 1):
            rows = np.flatnonzero(column == inner_fold)
            frame = pd.DataFrame(probabilities(rows), columns=claude_glm.BLEND)
            frame.insert(0, "groups_sha256", claude_glm.groups_sha256(train))
            frame.insert(0, "weight", 0.25)
            frame.insert(0, "inner_fold", inner_fold)
            frame.insert(0, "outer_fold", fold)
            frame.insert(0, "id", train["id"].to_numpy()[rows])
            frames.append(frame)
    cache = pd.concat(frames, ignore_index=True)
    frozen = neural.FrozenConfig()
    locked = np.empty_like(current)
    for fold in range(5):  # the complete incumbent, built as the lock records it
        _, s, _ = gemini_pair.inner_incumbents(train, blocks, plan, cache, fold, frozen)
        rows, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        specialist = claude_glm.specialist_probabilities(blocks, gemini_pair.INCUMBENT, truth, fit, rows, frozen)
        locked[rows] = claude_glm.redistribute(current[rows], specialist, s)
    return {"train": train, "plan": plan, "blocks": blocks, "current": current, "cache": cache, "locked": locked,
            "features": gemini_markers.feature_table(train), "frozen": frozen}


def test_a_fold_selects_c_then_a_on_inner_rows_and_applies_them_to_the_locked_incumbent(study):
    p = study
    truth, folds = ensemble._targets(p["train"]), p["plan"]["outer_fold"].to_numpy()
    target = (truth == PAIR_INDEX[1]).astype(int)
    for fold in (0, 3):
        result = gemini_markers.run_fold(p["train"], p["blocks"], p["features"], p["plan"], p["current"], p["cache"],
                                         p["locked"], fold, p["frozen"])
        record = result["record"]
        for name in gemini_markers.MODELS:
            means = {C: np.mean(record["inner_binary_balanced_accuracy"][name][f"{C:g}"])
                     for C in gemini_markers.C_GRID}
            assert record["selected_C"][name] == first_best(gemini_markers.C_GRID, means.__getitem__)
        a_means = {a: record["a_inner_mean_balanced_accuracy"][f"{a:g}"] for a in gemini_markers.A_GRID}
        assert record["selected_a"] == first_best(gemini_markers.A_GRID, a_means.__getitem__)
        assert record["incumbent_lock_max_abs_difference"] <= 1e-12
        rows, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        assert np.array_equal(result["rows"], rows)
        for partition in record["inner_partitions"]:
            assert partition["fit_pair_rows"] + partition["valid_pair_rows"] == np.isin(truth[fit], PAIR_INDEX).sum()
        # The outer style head, refit independently on the outer fitting pair rows only.
        pair_fit = fit[np.isin(truth[fit], PAIR_INDEX)]
        values = p["features"][list(gemini_markers.MODELS["style"])].to_numpy(float)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=record["selected_C"]["style"], max_iter=2000,
                                                                   tol=1e-6, random_state=2026))
        scores = model.fit(values[pair_fit], target[pair_fit]).predict_proba(values[rows])[:, 1]
        np.testing.assert_allclose(result["style_all"], scores, rtol=0, atol=1e-12)
        expected = gemini_markers.integrate(p["locked"][rows], scores, record["selected_a"])
        np.testing.assert_allclose(result["candidate"], expected, rtol=0, atol=1e-12)
        if record["selected_a"] == 0:
            assert np.array_equal(result["candidate"], p["locked"][rows])
        assert all(item["converged"] for item in record["fits"])


def test_ties_choose_the_smaller_c_and_a_zero(study, monkeypatch):
    p = study
    monkeypatch.setattr(gemini_markers, "_binary_ba", lambda target, scores: 0.5)
    monkeypatch.setattr(gemini_markers, "_ba", lambda truth, probabilities: 0.4)
    record = gemini_markers.run_fold(p["train"], p["blocks"], p["features"], p["plan"], p["current"], p["cache"],
                                     p["locked"], 1, p["frozen"])["record"]
    assert set(record["selected_C"].values()) == {0.01} and record["selected_a"] == 0.0


def test_a_lock_that_the_rebuild_does_not_reproduce_stops_the_run(study):
    p = study
    wrong = p["locked"].copy()
    rows = np.flatnonzero(p["plan"]["outer_fold"].to_numpy() == 2)
    wrong[rows[0]] = wrong[rows[0]][::-1]
    with pytest.raises(ValueError, match="does not match its lock"):
        gemini_markers.run_fold(p["train"], p["blocks"], p["features"], p["plan"], p["current"], p["cache"], wrong, 2,
                                p["frozen"])


def test_reports_separate_binary_and_system_outcomes():
    target = np.tile([0, 1], 10)
    scores, reference = np.where(target == 1, 0.8, 0.3), np.full(20, 0.6)
    metrics = gemini_markers.binary_metrics(target, scores, reference)
    assert metrics["balanced_accuracy"] == 1.0 and metrics["versus_incumbent_forced"] == {"gained": 10, "lost": 0}
    assert metrics["brier"] == pytest.approx(np.mean((scores - target) ** 2))
    bins = gemini_markers._bin_report(target, np.array(["<60"] * 10 + ["60-99"] * 10), np.arange(20) // 2,
                                      {"x": scores})
    assert bins["<60"]["balanced_accuracy"]["x"] == 1.0 and bins["100-149"]["balanced_accuracy"]["x"] is None
    truth = np.repeat(np.arange(len(LABELS)), 10)
    folds, groups = np.tile(np.arange(5), len(truth) // 5), np.arange(len(truth)) // 3
    incumbent = np.eye(len(LABELS))[truth] * 0.6 + 0.4 / len(LABELS)
    candidate = incumbent.copy()
    row = np.flatnonzero(truth == PAIR_INDEX[0])[0]
    candidate[row, [PAIR_INDEX[0], PAIR_INDEX[1]]] = candidate[row, [PAIR_INDEX[1], PAIR_INDEX[0]]]
    report = gemini_markers.system_report(truth, folds, groups, incumbent, candidate, "prompt")
    assert report["swaps"][f"{LABELS[PAIR_INDEX[0]]}->{LABELS[PAIR_INDEX[1]]}"]["introduced"] == 1
    assert report["changes"]["lost"] == 1 and not report["gate"]["passes"]
    expected = balanced_accuracy_score(truth, candidate.argmax(axis=1)) - balanced_accuracy_score(truth, truth)
    assert report["gate"]["delta"] == pytest.approx(expected)
    same = gemini_markers.system_report(truth, folds, groups, incumbent, incumbent, "topic")
    assert same["gate"]["delta"] == 0.0 and same["gate"]["zero_folds"] == 5 and not same["gate"]["passes"]
