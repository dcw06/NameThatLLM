"""18-class head: b = 0 is the complete incumbent exactly; s and b are chosen on inner rows only."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from ntl import claude_glm, ensemble, lora, neural, styledistance_head
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.smoke import synthetic_frames


def _probabilities(rng, truth, signal):
    logits = rng.normal(size=(len(truth), len(LABELS)))
    logits[np.arange(len(truth)), truth] += signal
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


@pytest.fixture
def setup(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    plan = lora.split_plan(train, save_fold_manifest(manifest, metadata, tmp_path / "folds"), lora.AdapterConfig())
    rng = np.random.default_rng(11)
    truth = ensemble._targets(train)
    blocks = {"qwen": rng.normal(size=(len(train), 8)), "styledistance": rng.normal(size=(len(train), len(LABELS)))}
    blocks["qwen"][:, 0] += 0.3 * truth
    blocks["styledistance"][np.arange(len(train)), truth] += 3.0  # StyleDistance separates every class
    # An imperfect current system, so both the specialist's s and the head's b matter.
    folds = plan.set_index("id").loc[train["id"], "outer_fold"].to_numpy()
    current = pd.DataFrame(_probabilities(rng, truth, 1.0), columns=claude_glm.BLEND)
    current.insert(0, "fold", folds)
    current.insert(0, "true_model", train["model"].to_numpy())
    current.insert(0, "prompt_id", train["prompt_id"].to_numpy())
    current.insert(0, "id", train["id"].to_numpy())
    frames = []
    for fold in range(5):
        column = plan[f"inner_{fold}"].to_numpy()
        for inner_fold in (0, 1):
            rows = np.flatnonzero(column == inner_fold)
            frame = pd.DataFrame(_probabilities(rng, truth[rows], 1.0), columns=claude_glm.BLEND)
            frame.insert(0, "groups_sha256", claude_glm.groups_sha256(train))
            frame.insert(0, "weight", 0.25)
            frame.insert(0, "inner_fold", inner_fold)
            frame.insert(0, "outer_fold", fold)
            frame.insert(0, "id", train["id"].to_numpy()[rows])
            frames.append(frame)
    return {"train": train, "plan": plan, "blocks": blocks, "current": current,
            "cache": pd.concat(frames, ignore_index=True), "root": tmp_path}


def test_b_zero_is_the_incumbent_and_every_fold_applies_its_choice(setup):
    reference = claude_glm.evaluate(setup["train"], setup["blocks"], setup["plan"], setup["current"],
                                    setup["cache"], setup["root"] / "reference", samples=20)
    result = styledistance_head.evaluate(setup["train"], setup["blocks"], setup["plan"], setup["current"],
                                         setup["cache"], setup["root"] / "head", samples=20)
    old = pd.read_parquet(setup["root"] / "reference" / "predictions.parquet")
    new = pd.read_parquet(setup["root"] / "head" / "predictions.parquet")
    assert (new["prediction::incumbent"] == old["prediction::current+qwen+styledistance"]).all()
    assert [item["s"] for item in result["selection"]] == \
        [item["weight"] for item in reference["selection"]["qwen+styledistance"]]
    for system in styledistance_head.HEADS:
        assert result["weight_balanced_accuracy"][system]["0.0"] == pytest.approx(
            result["systems"]["incumbent"]["balanced_accuracy"])
    chosen = [item["b"]["candidate"] for item in result["selection"]]
    assert any(weight > 0 for weight in chosen)
    for item in result["selection"]:
        rows = (new["outer_fold"] == item["outer_fold"]).to_numpy()
        for system in styledistance_head.HEADS:
            means = item["inner_mean_balanced_accuracy"][system]
            assert means[str(item["b"][system])] == max(means.values())
            if item["b"][system] == 0:  # an unchanged fold keeps the incumbent's predictions
                assert (new.loc[rows, f"prediction::{system}"] == new.loc[rows, "prediction::incumbent"]).all()
    heads = np.load(setup["root"] / "head" / "head-qwen+styledistance.npy")
    np.testing.assert_allclose(heads.sum(axis=1), 1.0)


def test_inner_choices_and_outer_predictions_recompute_independently(setup):
    """Inner fits see only the inner training partition, b follows the declared tie rule,
    and each outer fold applies its b to the incumbent refit on the outer training rows."""
    train, plan, blocks, cache = setup["train"], setup["plan"], setup["blocks"], setup["cache"]
    result = styledistance_head.evaluate(train, blocks, plan, setup["current"], cache, setup["root"] / "head",
                                         samples=20)
    predictions = pd.read_parquet(setup["root"] / "head" / "predictions.parquet").set_index("id")
    truth, folds, frozen = ensemble._targets(train), plan["outer_fold"].to_numpy(), neural.FrozenConfig()
    current = setup["current"].set_index("id").loc[train["id"], claude_glm.BLEND].to_numpy()
    position = {row: index for index, row in enumerate(train["id"])}
    for item in result["selection"]:
        fold, s = item["outer_fold"], item["s"]
        column = plan[f"inner_{fold}"].to_numpy()
        for inner_fold in (0, 1):
            frame = cache[(cache["outer_fold"] == fold) & (cache["inner_fold"] == inner_fold)]
            valid = np.array([position[row] for row in frame["id"]])
            fit = np.flatnonzero((folds != fold) & (column != inner_fold))
            specialist = claude_glm.specialist_probabilities(blocks, "qwen+styledistance", truth, fit, valid, frozen)
            incumbent = claude_glm.redistribute(frame[claude_glm.BLEND].to_numpy(), specialist, s)
            for system, name in styledistance_head.HEADS.items():
                head = styledistance_head.head_probabilities(blocks, name, truth, fit, valid, frozen)
                for weight in styledistance_head.WEIGHTS:
                    expected = balanced_accuracy_score(truth[valid],
                                                       ((1 - weight) * incumbent + weight * head).argmax(axis=1))
                    assert item["inner_balanced_accuracy"][system][str(weight)][inner_fold] == pytest.approx(expected)
        for system in styledistance_head.HEADS:  # the smallest b whose inner mean is the best
            means = item["inner_mean_balanced_accuracy"][system]
            assert item["b"][system] == min(float(key) for key, value in means.items()
                                            if value >= max(means.values()) - 1e-12)
        rows, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        specialist = claude_glm.specialist_probabilities(blocks, "qwen+styledistance", truth, fit, rows, frozen)
        incumbent = claude_glm.redistribute(current[rows], specialist, s)
        for system, name in styledistance_head.HEADS.items():
            head = styledistance_head.head_probabilities(blocks, name, truth, fit, rows, frozen)
            weight = item["b"][system]
            expected = np.asarray(LABELS)[((1 - weight) * incumbent + weight * head).argmax(axis=1)]
            assert (predictions.loc[train["id"].to_numpy()[rows], f"prediction::{system}"] == expected).all()
    # The fixture's control head ties the incumbent at several b, so the tie rule is exercised.
    assert any(len(set(item["inner_mean_balanced_accuracy"]["control"].values())) < len(styledistance_head.WEIGHTS)
               for item in result["selection"])


def test_reports_where_the_change_lands(setup):
    result = styledistance_head.evaluate(setup["train"], setup["blocks"], setup["plan"], setup["current"],
                                         setup["cache"], setup["root"] / "head", samples=20)
    frame = pd.read_parquet(setup["root"] / "head" / "predictions.parquet")
    true = frame["true_model"].to_numpy()
    incumbent = frame["prediction::incumbent"].to_numpy()
    for system in styledistance_head.HEADS:
        predicted = frame[f"prediction::{system}"].to_numpy()
        systems = result["systems"]
        delta = systems[system]["balanced_accuracy"] - systems["incumbent"]["balanced_accuracy"]
        paired, changes = result["paired"][f"{system}_vs_incumbent"], result["changes"][system]
        assert paired["balanced_accuracy_delta"] == pytest.approx(delta)  # second minus first
        assert systems[system]["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(true, predicted))
        gained, lost = (incumbent != true) & (predicted == true), (incumbent == true) & (predicted != true)
        assert (changes["gained"], changes["lost"]) == (gained.sum(), lost.sum())
        assert (paired["second_only_correct"], paired["first_only_correct"]) == (gained.sum(), lost.sum())
        for label in LABELS:
            assert changes["by_class"][label] == {"gained": gained[true == label].sum(),
                                                  "lost": lost[true == label].sum()}
        contributions = result["contributions"][system]
        assert sum(contributions["by_class"].values()) == pytest.approx(delta)
        assert contributions["gpt_5_6"] == pytest.approx(
            sum(contributions["by_class"][label] for label in styledistance_head.GPT))
    for name, confusion in result["gpt_confusion"].items():
        predicted = frame[f"prediction::{name}"].to_numpy()
        within = 0
        for label in styledistance_head.GPT:
            rows = true == label
            for other in styledistance_head.GPT:
                count = int((predicted[rows] == other).sum())
                assert confusion["matrix"][label][other] == count
                within += count if other != label else 0
            assert confusion["matrix"][label]["other"] == int((~np.isin(predicted[rows], styledistance_head.GPT)).sum())
        assert confusion["within_family_errors"] == within
        assert confusion["within_family_error_share"] == within / np.isin(true, styledistance_head.GPT).sum()


def test_blending_is_a_convex_combination_with_an_exact_zero():
    rng = np.random.default_rng(1)
    incumbent, head = rng.dirichlet(np.ones(18), size=4), rng.dirichlet(np.ones(18), size=4)
    assert styledistance_head.blended(incumbent, head, 0.0) is incumbent
    quarter = styledistance_head.blended(incumbent, head, 0.25)
    np.testing.assert_allclose(quarter, 0.75 * incumbent + 0.25 * head)  # b is the head's weight
    np.testing.assert_allclose(quarter.sum(axis=1), 1.0)


def test_inputs_are_checked(setup):
    stale = setup["cache"].assign(groups_sha256="other")
    with pytest.raises(ValueError, match="different row grouping"):
        styledistance_head.evaluate(setup["train"], setup["blocks"], setup["plan"], setup["current"], stale,
                                    setup["root"] / "a", samples=10)
    (setup["root"] / "b").mkdir()
    (setup["root"] / "b" / "comparison.json").write_text("{}")
    with pytest.raises(ValueError, match="overwritten"):
        styledistance_head.evaluate(setup["train"], setup["blocks"], setup["plan"], setup["current"],
                                    setup["cache"], setup["root"] / "b", samples=10)
    truth = ensemble._targets(setup["train"])
    fit = np.flatnonzero(truth != 0)  # one class missing from the training rows
    with pytest.raises(ValueError, match="complete canonical label order"):
        styledistance_head.head_probabilities(setup["blocks"], "qwen+styledistance", truth, fit, np.arange(5),
                                              neural.FrozenConfig())
