"""Gemini pair specialist: the incumbent option is the incumbent exactly; every choice is made on inner rows."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ntl import claude_glm, ensemble, gemini_pair, lora, neural
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.modeling import BaselineConfig, build_pipeline
from ntl.smoke import synthetic_frames

PAIR_INDEX = [LABELS.index("gemini-3.7-flash"), LABELS.index("gemini-3.8-flash")]


def _probabilities(rng, truth, signal):
    logits = rng.normal(size=(len(truth), len(LABELS)))
    logits[np.arange(len(truth)), truth] += signal
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def _confused(rng, truth, signal):
    """Probabilities that often swap the pair, so the specialist has errors to recover."""
    probabilities = _probabilities(rng, truth, signal)
    swap = np.isin(truth, PAIR_INDEX) & (rng.random(len(truth)) < 0.5)
    probabilities[np.ix_(swap, PAIR_INDEX)] = probabilities[np.ix_(swap, PAIR_INDEX[::-1])]
    return probabilities


@pytest.fixture
def setup(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    # One text marker for both versions, so no CPU specialist separates the pair perfectly
    # and the chosen configuration, C and a all matter.
    pair_rows = train["model"].isin(["gemini-3.7-flash", "gemini-3.8-flash"])
    train.loc[pair_rows, "text"] = train.loc[pair_rows, "text"].str.replace("ClassMarker08", "ClassMarker07")
    manifest, metadata = build_fold_manifest(train)
    plan = lora.split_plan(train, save_fold_manifest(manifest, metadata, tmp_path / "folds"), lora.AdapterConfig())
    rng = np.random.default_rng(5)
    truth = ensemble._targets(train)
    blocks = {"qwen": rng.normal(size=(len(train), 8)), "styledistance": rng.normal(size=(len(train), 4))}
    blocks["qwen"][:, 0] += 0.3 * truth
    blocks["qwen"][:, 1] += 1.0 * (truth == PAIR_INDEX[1])  # Qwen partly separates the pair
    blocks["styledistance"][:, 0] += 0.3 * np.isin(truth, claude_glm.INDEX) * truth
    folds = plan.set_index("id").loc[train["id"], "outer_fold"].to_numpy()
    current = pd.DataFrame(_confused(rng, truth, 2.0), columns=claude_glm.BLEND)
    current.insert(0, "fold", folds)
    current.insert(0, "true_model", train["model"].to_numpy())
    current.insert(0, "prompt_id", train["prompt_id"].to_numpy())
    current.insert(0, "id", train["id"].to_numpy())
    frames = []
    for fold in range(5):
        column = plan[f"inner_{fold}"].to_numpy()
        for inner_fold in (0, 1):
            rows = np.flatnonzero(column == inner_fold)
            frame = pd.DataFrame(_confused(rng, truth[rows], 2.0), columns=claude_glm.BLEND)
            frame.insert(0, "groups_sha256", claude_glm.groups_sha256(train))
            frame.insert(0, "weight", 0.25)
            frame.insert(0, "inner_fold", inner_fold)
            frame.insert(0, "outer_fold", fold)
            frame.insert(0, "id", train["id"].to_numpy()[rows])
            frames.append(frame)
    return {"train": train, "plan": plan, "blocks": blocks, "current": current,
            "cache": pd.concat(frames, ignore_index=True), "root": tmp_path}


def _evaluate(setup, name="pair"):
    return gemini_pair.evaluate(setup["train"], setup["blocks"], setup["plan"], setup["current"], setup["cache"],
                                setup["root"] / name, samples=20)


def _manual_pair(train, blocks, truth, fit, rows, configuration, C):
    """The specialist fit independently: true-pair training rows only, 3.8 as the positive class."""
    fit = [row for row in fit if truth[row] in PAIR_INDEX]
    target = (truth[fit] == PAIR_INDEX[1]).astype(int)
    if configuration == "cpu":
        model, inputs = build_pipeline(BaselineConfig(C=C)), train["text"].to_numpy(dtype=object)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=2026))
        inputs = blocks["qwen"]
    model.fit(inputs[fit], target)
    return model.predict_proba(inputs[rows])


def _apply(base, q, weight):
    result = base.copy()
    mass = base[:, PAIR_INDEX].sum(axis=1, keepdims=True)
    result[:, PAIR_INDEX] = (1 - weight) * base[:, PAIR_INDEX] + weight * mass * q
    return result


def test_the_incumbent_option_is_the_incumbent_exactly(setup):
    reference = claude_glm.evaluate(setup["train"], setup["blocks"], setup["plan"], setup["current"],
                                    setup["cache"], setup["root"] / "reference", samples=20)
    result = _evaluate(setup)
    old = pd.read_parquet(setup["root"] / "reference" / "predictions.parquet")
    new = pd.read_parquet(setup["root"] / "pair" / "predictions.parquet")
    assert (new["prediction::incumbent"] == old["prediction::current+qwen+styledistance"]).all()
    assert [item["s"] for item in result["selection"]] == \
        [item["weight"] for item in reference["selection"]["qwen+styledistance"]]
    assert result["option_balanced_accuracy"]["incumbent"] == pytest.approx(
        result["systems"]["incumbent"]["balanced_accuracy"])
    assert any(item["selected"] != "incumbent" for item in result["selection"])
    for item in result["selection"]:
        rows = (new["outer_fold"] == item["outer_fold"]).to_numpy()
        if item["selected"] == "incumbent":
            assert (new.loc[rows, "prediction::candidate"] == new.loc[rows, "prediction::incumbent"]).all()


def test_inner_choices_and_outer_predictions_recompute_independently(setup):
    train, plan, blocks, cache = setup["train"], setup["plan"], setup["blocks"], setup["cache"]
    result = _evaluate(setup)
    predictions = pd.read_parquet(setup["root"] / "pair" / "predictions.parquet").set_index("id")
    truth, folds, frozen = ensemble._targets(train), plan["outer_fold"].to_numpy(), neural.FrozenConfig()
    current = setup["current"].set_index("id").loc[train["id"], claude_glm.BLEND].to_numpy()
    position = {row: index for index, row in enumerate(train["id"])}
    names = [gemini_pair.option_name(option) for option in gemini_pair.options()]
    incumbent, candidate = np.empty_like(current), np.empty_like(current)
    for item in result["selection"]:
        fold, s = item["outer_fold"], item["s"]
        column = plan[f"inner_{fold}"].to_numpy()
        for inner_fold in (0, 1):
            frame = cache[(cache["outer_fold"] == fold) & (cache["inner_fold"] == inner_fold)]
            valid = np.array([position[row] for row in frame["id"]])
            fit = np.flatnonzero((folds != fold) & (column != inner_fold))
            specialist = claude_glm.specialist_probabilities(blocks, "qwen+styledistance", truth, fit, valid, frozen)
            base = claude_glm.redistribute(frame[claude_glm.BLEND].to_numpy(), specialist, s)
            scores = item["inner_balanced_accuracy"]
            assert scores["incumbent"][inner_fold] == pytest.approx(
                balanced_accuracy_score(truth[valid], base.argmax(axis=1)))
            for configuration, C in gemini_pair.SPECIALISTS:
                q = _manual_pair(train, blocks, truth, fit, valid, configuration, C)
                for weight in gemini_pair.WEIGHTS:
                    expected = balanced_accuracy_score(truth[valid], _apply(base, q, weight).argmax(axis=1))
                    name = gemini_pair.option_name((configuration, C, weight))
                    assert scores[name][inner_fold] == pytest.approx(expected)
        means = item["inner_mean_balanced_accuracy"]  # the first option, in tie order, at the best mean
        assert all(means[name] == pytest.approx(np.mean(item["inner_balanced_accuracy"][name])) for name in names)
        assert item["selected"] == next(name for name in names if means[name] >= max(means.values()) - 1e-12)
        best = next(name for name in names[1:] if means[name] >= max(means[key] for key in names[1:]) - 1e-12)
        shown = item["selected"] if item["selected"] != "incumbent" else best
        assert item["reported_specialist"] == shown.rsplit("@a=", 1)[0]
        rows, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        specialist = claude_glm.specialist_probabilities(blocks, "qwen+styledistance", truth, fit, rows, frozen)
        incumbent[rows] = claude_glm.redistribute(current[rows], specialist, s)
        option = dict(zip(names, gemini_pair.options()))[item["selected"]]
        candidate[rows] = incumbent[rows] if option is None else _apply(
            incumbent[rows], _manual_pair(train, blocks, truth, fit, rows, *option[:2]), option[2])
        assert (predictions.loc[train["id"].to_numpy()[rows], "prediction::candidate"]
                == np.asarray(LABELS)[candidate[rows].argmax(axis=1)]).all()
    pair = np.isin(truth, PAIR_INDEX)
    target = (truth[pair] == PAIR_INDEX[1]).astype(int)
    binary = result["binary"]
    assert binary["rows"] == pair.sum()
    for name, values in (("incumbent_forced", incumbent), ("candidate_forced", candidate)):
        assert binary[name] == pytest.approx(
            balanced_accuracy_score(target, values[pair][:, PAIR_INDEX].argmax(axis=1)))


def test_ties_keep_the_incumbent(setup, monkeypatch):
    monkeypatch.setattr(gemini_pair, "_ba", lambda truth, probabilities: 0.5)  # every option ties
    result = _evaluate(setup)
    assert all(item["selected"] == "incumbent" and item["s"] == 0 for item in result["selection"])
    assert all(item["reported_specialist"] == "cpu@C=0.1" for item in result["selection"])


def test_reports_match_the_saved_predictions(setup):
    result = _evaluate(setup)
    frame = pd.read_parquet(setup["root"] / "pair" / "predictions.parquet")
    q = pd.read_parquet(setup["root"] / "pair" / "pair-probabilities.parquet")
    for item in result["selection"]:  # each fold reports its own specialist's probabilities
        rows = (frame["outer_fold"] == item["outer_fold"]).to_numpy()
        column = f"::{gemini_pair.PAIR[1]}"
        assert (q.loc[rows, "reported" + column] == q.loc[rows, item["reported_specialist"] + column]).all()
    true, first, second = (frame[column].to_numpy() for column in
                           ("true_model", "prediction::incumbent", "prediction::candidate"))
    systems, paired, changes = result["systems"], result["paired"], result["changes"]
    delta = systems["candidate"]["balanced_accuracy"] - systems["incumbent"]["balanced_accuracy"]
    assert paired["balanced_accuracy_delta"] == pytest.approx(delta)  # candidate minus incumbent
    assert systems["candidate"]["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(true, second))
    gained, lost = (first != true) & (second == true), (first == true) & (second != true)
    assert (changes["gained"], changes["lost"]) == (gained.sum(), lost.sum())
    assert (paired["second_only_correct"], paired["first_only_correct"]) == (gained.sum(), lost.sum())
    outside = ~np.isin(true, gemini_pair.PAIR)
    assert (changes["gained_outside_pair"], changes["lost_outside_pair"]) == (
        (gained & outside).sum(), (lost & outside).sum())
    for label in LABELS:
        assert changes["by_class"][label] == {"gained": (gained & (true == label)).sum(),
                                              "lost": (lost & (true == label)).sum()}
    for a, b in (gemini_pair.PAIR, gemini_pair.PAIR[::-1]):
        swaps = result["swaps"][f"{a}->{b}"]
        rows = true == a
        assert swaps == {"incumbent": (first[rows] == b).sum(), "candidate": (second[rows] == b).sum(),
                         "corrected": ((first == b) & (second == a) & rows).sum(),
                         "introduced": ((first == a) & (second == b) & rows).sum()}
    for label in gemini_pair.GEMINI:
        assert result["gemini_recall"][label] == {"incumbent": systems["incumbent"]["recall"][label],
                                                  "candidate": systems["candidate"]["recall"][label]}
    pair = np.isin(true, gemini_pair.PAIR)
    target = (true[pair] == gemini_pair.PAIR[1]).astype(int)
    for name, column in [("specialist", "reported")] + [(key, key) for key in result["binary"]["fixed_specialists"]]:
        positive = q[f"{column}::{gemini_pair.PAIR[1]}"].to_numpy()[pair]
        expected = balanced_accuracy_score(target, (positive > 1 - positive).astype(int))
        value = result["binary"][name] if name == "specialist" else result["binary"]["fixed_specialists"][name]
        assert value == pytest.approx(expected)


def test_integration_moves_only_the_pair_and_zero_is_exact():
    rng = np.random.default_rng(3)
    incumbent = rng.dirichlet(np.ones(len(LABELS)), size=6)
    q = rng.dirichlet(np.ones(2), size=6)
    pair = {("cpu", 1.0): q}
    assert gemini_pair.integrated(incumbent, pair, None) is incumbent
    zero = claude_glm.redistribute(incumbent, q, 0.0, gemini_pair.PAIR_INDEX)
    assert np.array_equal(zero, incumbent)
    quarter = gemini_pair.integrated(incumbent, pair, ("cpu", 1.0, 0.25))
    np.testing.assert_allclose(quarter, _apply(incumbent, q, 0.25))  # a is the specialist's weight
    others = np.setdiff1d(np.arange(len(LABELS)), PAIR_INDEX)
    assert np.array_equal(quarter[:, others], incumbent[:, others])
    np.testing.assert_allclose(quarter.sum(axis=1), 1.0)
    np.testing.assert_allclose(quarter[:, PAIR_INDEX].sum(axis=1), incumbent[:, PAIR_INDEX].sum(axis=1))


def test_options_follow_the_declared_tie_order():
    names = [gemini_pair.option_name(option) for option in gemini_pair.options()]
    assert len(names) == 25 and len(set(names)) == 25
    assert names[:8] == ["incumbent", "cpu@C=0.1@a=0.25", "cpu@C=1@a=0.25", "cpu@C=4@a=0.25",
                         "qwen@C=0.005@a=0.25", "qwen@C=0.05@a=0.25", "qwen@C=0.5@a=0.25", "cpu@C=0.1@a=0.5"]
    assert names[-1] == "qwen@C=0.5@a=1"


def test_inputs_are_checked(setup):
    stale = setup["cache"].assign(groups_sha256="other")
    with pytest.raises(ValueError, match="different row grouping"):
        gemini_pair.evaluate(setup["train"], setup["blocks"], setup["plan"], setup["current"], stale,
                             setup["root"] / "a", samples=10)
    (setup["root"] / "b").mkdir()
    (setup["root"] / "b" / "comparison.json").write_text("{}")
    with pytest.raises(ValueError, match="overwritten"):
        gemini_pair.evaluate(setup["train"], setup["blocks"], setup["plan"], setup["current"], setup["cache"],
                             setup["root"] / "b", samples=10)
    truth = ensemble._targets(setup["train"])
    texts = setup["train"]["text"].to_numpy(dtype=object)
    only = np.flatnonzero(truth == PAIR_INDEX[0])
    with pytest.raises(ValueError, match="Both Gemini versions"):
        gemini_pair.pair_probabilities(texts, setup["blocks"]["qwen"], truth, only, np.arange(3), "qwen", 0.05, 0)
    with pytest.raises(ValueError, match="Unknown configuration"):
        gemini_pair.pair_probabilities(texts, setup["blocks"]["qwen"], truth, np.arange(len(truth)), np.arange(3),
                                       "tfidf", 1.0, 0)
