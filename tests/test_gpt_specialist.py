"""GPT specialist: fit on training-partition GPT rows only, the declared selection order, the
pair-mass formula on the five GPT columns, and an incumbent option equal to the locked vectors."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ntl import claude_glm, ensemble, gemini_pair, gpt_specialist, lora, neural
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.selection import first_best
from ntl.smoke import synthetic_frames

GPT_INDEX = gpt_specialist.GPT_INDEX


@pytest.fixture
def setup(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    plan = lora.split_plan(train, save_fold_manifest(manifest, metadata, tmp_path / "folds"), lora.AdapterConfig())
    rng = np.random.default_rng(6)
    truth = ensemble._targets(train)
    blocks = {"qwen": rng.normal(size=(len(train), 8)), "styledistance": rng.normal(size=(len(train), 4))}
    blocks["qwen"][:, 0] += 0.3 * truth
    gpt = np.isin(truth, GPT_INDEX)
    blocks["qwen"][gpt, 1] += 0.8 * truth[gpt]  # the GPT versions are partly separable
    blocks["styledistance"][:, 0] += 0.3 * np.isin(truth, claude_glm.INDEX) * truth
    folds = plan["outer_fold"].to_numpy()

    def probabilities(rows):
        logits = rng.normal(size=(len(rows), len(LABELS)))
        logits[np.arange(len(rows)), truth[rows]] += 1.5
        shuffle = gpt[rows] & (rng.random(len(rows)) < 0.6)  # GPT rows often take a sibling's score
        for row in np.flatnonzero(shuffle):
            logits[row, GPT_INDEX] = rng.permutation(logits[row, GPT_INDEX])
        exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exponent / exponent.sum(axis=1, keepdims=True)

    current = pd.DataFrame(probabilities(np.arange(len(train))), columns=claude_glm.BLEND)
    current.insert(0, "fold", folds)
    current.insert(0, "true_model", train["model"].to_numpy())
    current.insert(0, "prompt_id", train["prompt_id"].to_numpy())
    current.insert(0, "id", train["id"].to_numpy())
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
    base = current[claude_glm.BLEND].to_numpy()
    locked = np.empty_like(base)
    for fold in range(5):  # the complete incumbent, built as its lock records it
        _, s, _ = gemini_pair.inner_incumbents(train, blocks, plan, cache, fold, frozen)
        rows, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        claude = claude_glm.specialist_probabilities(blocks, gemini_pair.INCUMBENT, truth, fit, rows, frozen)
        locked[rows] = claude_glm.redistribute(base[rows], claude, s)
    return {"train": train, "plan": plan, "blocks": blocks, "current": current, "cache": cache, "locked": locked,
            "frozen": frozen, "root": tmp_path}


def _evaluate(setup, name="out", **kwargs):
    return gpt_specialist.evaluate(setup["train"], setup["blocks"], setup["plan"], setup["current"], setup["cache"],
                                   setup["locked"], setup["root"] / name, samples=50, **kwargs)


def test_the_class_index_keeps_the_claude_glm_default_and_fits_only_gpt_training_rows(setup):
    p = setup
    truth, folds = ensemble._targets(p["train"]), p["plan"]["outer_fold"].to_numpy()
    fit, rows = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    default =claude_glm.specialist_probabilities(p["blocks"], "qwen", truth, fit, rows, p["frozen"])
    explicit = claude_glm.specialist_probabilities(p["blocks"], "qwen", truth, fit, rows, p["frozen"], claude_glm.INDEX)
    assert np.array_equal(default, explicit) and default.shape[1] == 7
    gpt = gpt_specialist.specialist(p["blocks"], "qwen", truth, fit, rows, p["frozen"])
    training = fit[np.isin(truth[fit], GPT_INDEX)]
    targets = np.searchsorted(GPT_INDEX, truth[training])
    manual = make_pipeline(StandardScaler(), LogisticRegression(C=0.005, max_iter=2000, random_state=2026))
    np.testing.assert_allclose(gpt, manual.fit(p["blocks"]["qwen"][training], targets)
                               .predict_proba(p["blocks"]["qwen"][rows]), rtol=0, atol=1e-10)
    with pytest.raises(ValueError, match="Every specialist class"):
        gpt_specialist.specialist(p["blocks"], "qwen", truth, fit[~np.isin(truth[fit], GPT_INDEX[:1])], rows,
                                  p["frozen"])


def test_selection_follows_the_declared_order_and_the_incumbent_option_is_exact(setup):
    p = setup
    result = _evaluate(p)
    truth, folds = ensemble._targets(p["train"]), p["plan"]["outer_fold"].to_numpy()
    grid = gpt_specialist.options()
    saved = np.load(p["root"] / "out" / "probabilities.npz")
    for item in result["selection"]:
        means = item["inner_mean_balanced_accuracy"]
        assert item["selected"] == gpt_specialist.option_name(
            first_best(grid, lambda option: means[gpt_specialist.option_name(option)]))
        assert all(means[name] == pytest.approx(np.mean(values)) for name, values in item["inner_balanced_accuracy"].items())
        for feature_set, name in item["selected_by_feature_set"].items():
            assert name == "incumbent" or name.startswith(f"{feature_set}@")
        rows = folds == item["outer_fold"]
        if item["selected"] == "incumbent":
            assert np.array_equal(saved["candidate"][rows], p["locked"][rows])
        assert item["incumbent_lock_max_abs_difference"] <= 1e-12
    predictions = saved["candidate"].argmax(axis=1)
    assert result["systems"]["candidate"]["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(truth, predictions))
    first = p["locked"].argmax(axis=1)
    assert result["candidate_vs_incumbent"]["gained"] == int(((first != truth) & (predictions == truth)).sum())
    assert result["candidate_vs_incumbent"]["lost"] == int(((first == truth) & (predictions != truth)).sum())
    promotion = result["promotion"]
    assert promotion["passes"] == (promotion["delta"] >= 0.005 and promotion["interval"][0] > 0)
    gpt = np.isin(truth, GPT_INDEX)
    forced = balanced_accuracy_score(np.searchsorted(GPT_INDEX, truth[gpt]), p["locked"][gpt][:, GPT_INDEX].argmax(axis=1))
    assert result["five_way_balanced_accuracy"]["incumbent_forced"] == pytest.approx(forced)
    with pytest.raises(ValueError, match="new output directory"):
        _evaluate(p)


def test_a_selected_option_applies_its_own_outer_specialist_through_the_formula(setup, monkeypatch):
    p = setup
    # Force the last option, full weight on Qwen + StyleDistance, so the integration is exercised.
    monkeypatch.setattr(gpt_specialist, "first_best", lambda options, score: list(options)[-1])
    result = _evaluate(p, name="forced")
    truth, folds = ensemble._targets(p["train"]), p["plan"]["outer_fold"].to_numpy()
    saved = np.load(p["root"] / "forced" / "probabilities.npz")
    others = np.setdiff1d(np.arange(len(LABELS)), GPT_INDEX)
    assert np.array_equal(saved["candidate"][:, others], p["locked"][:, others])
    for item in result["selection"]:
        assert item["selected"] == "qwen+styledistance@w=1"
        rows, fit = np.flatnonzero(folds == item["outer_fold"]), np.flatnonzero(folds != item["outer_fold"])
        q = claude_glm.specialist_probabilities(p["blocks"], "qwen+styledistance", truth, fit, rows, p["frozen"],
                                                GPT_INDEX)
        expected = p["locked"][rows].copy()
        mass = expected[:, GPT_INDEX].sum(axis=1, keepdims=True)
        expected[:, GPT_INDEX] = mass * q  # w = 1
        np.testing.assert_allclose(saved["candidate"][rows], expected, rtol=0, atol=1e-12)
    assert (saved["candidate"].argmax(axis=1) != p["locked"].argmax(axis=1)).any()
    gpt = np.isin(truth, GPT_INDEX)
    target = np.searchsorted(GPT_INDEX, truth[gpt])
    for name, values in (("incumbent_forced", p["locked"]), ("candidate_forced", saved["candidate"])):
        assert result["five_way_balanced_accuracy"][name] == pytest.approx(
            balanced_accuracy_score(target, values[gpt][:, GPT_INDEX].argmax(axis=1)))
    assert result["five_way_balanced_accuracy"]["incumbent_forced"] != \
        result["five_way_balanced_accuracy"]["candidate_forced"]
    topic = _evaluate(p, name="topic", gate="topic")
    changes = topic["promotion"]["fold_changes"]
    assert topic["promotion"]["passes"] == (topic["promotion"]["delta"] > 0 and sum(c >= 0 for c in changes) >= 3)


def test_a_lock_the_rebuild_does_not_reproduce_stops_the_run(setup):
    p = setup
    wrong = p["locked"].copy()
    wrong[0] = wrong[0][::-1]
    with pytest.raises(ValueError, match="does not match its lock"):
        gpt_specialist.evaluate(p["train"], p["blocks"], p["plan"], p["current"], p["cache"], wrong,
                                p["root"] / "wrong", samples=20)


def test_confusion_and_routing_counts():
    truth = np.array([0, 0, 1, 2, 5, 6])
    first = np.array([0, 3, 7, 2, 1, 6])
    second = np.array([1, 3, 1, 9, 5, 0])
    confusion = gpt_specialist.gpt_confusion(truth, first)
    assert confusion["gpt-5.5"] == {"gpt-5.5": 1, "gpt-5.6-luna": 0, "gpt-5.6-sol": 0, "gpt-5.6-terra": 1,
                                    "gpt-6-astra": 0, "other": 0}
    assert confusion["gpt-5.6-luna"]["other"] == 1
    moves = gpt_specialist.routing(truth, first, second)
    assert moves["gpt-5.6-luna"] == {"gpt_to_other": 0, "other_to_gpt": 1}
    assert moves["gpt-5.6-sol"] == {"gpt_to_other": 1, "other_to_gpt": 0}
    assert moves["gemini-3.1-pro-preview"] == {"gpt_to_other": 1, "other_to_gpt": 0}
    assert moves["gemini-3.6-flash"] == {"gpt_to_other": 0, "other_to_gpt": 1}
