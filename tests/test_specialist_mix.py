"""Added-encoder specialists: a = 0 is the incumbent exactly; encoder and a are chosen on inner rows only."""

import numpy as np
import pandas as pd
import pytest

from ntl import claude_glm, ensemble, lora, specialist_mix
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.frozen_gemini import save_feature_archive
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
    rng = np.random.default_rng(7)
    truth = ensemble._targets(train)
    group = np.isin(truth, claude_glm.INDEX)
    blocks = {"qwen": rng.normal(size=(len(train), 8)), "styledistance": rng.normal(size=(len(train), 4)),
              "luar": rng.normal(size=(len(train), 3))}
    blocks["qwen"][:, 0] += 0.3 * truth
    blocks["styledistance"][:, 0] += 0.3 * group * truth
    blocks["luar"][:, 0] += group * truth  # the added encoder carries the strongest group signal
    # An imperfect current system, so the specialist's weight s matters.
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


def test_a_zero_is_the_incumbent_and_every_fold_applies_its_choice(setup):
    reference = claude_glm.evaluate(setup["train"], {name: setup["blocks"][name] for name in ("qwen", "styledistance")},
                                    setup["plan"], setup["current"], setup["cache"], setup["root"] / "reference",
                                    samples=20)
    result = specialist_mix.evaluate(setup["train"], setup["blocks"], ("luar",), setup["plan"], setup["current"],
                                     setup["cache"], setup["root"] / "mix", samples=20)
    old = pd.read_parquet(setup["root"] / "reference" / "predictions.parquet")
    new = pd.read_parquet(setup["root"] / "mix" / "predictions.parquet")
    assert (new["prediction::incumbent"] == old["prediction::current+qwen+styledistance"]).all()
    assert [item["s"] for item in result["selection"]] == \
        [item["weight"] for item in reference["selection"]["qwen+styledistance"]]
    assert result["option_balanced_accuracy"]["reference"] == pytest.approx(
        result["systems"]["incumbent"]["balanced_accuracy"])
    assert any(s > 0 for s in (item["s"] for item in result["selection"]))
    names = {specialist_mix.option_name(*option) for option in specialist_mix.options(("luar",))}
    assert names == {"reference", "luar@0.25", "luar@0.5", "luar@1"}
    for item in result["selection"]:
        assert item["selected"] in names and set(item["inner_mean_balanced_accuracy"]) == names
        rows = (new["outer_fold"] == item["outer_fold"]).to_numpy()
        if item["selected"] == "reference":  # an unchanged fold keeps the incumbent's predictions
            assert (new.loc[rows, "prediction::candidate"] == new.loc[rows, "prediction::incumbent"]).all()
    assert set(result["confusions"]["candidate"]) == {f"{a}->{b}" for a, b in specialist_mix.CONFUSIONS}
    assert result["changes"]["gained_outside_specialist"] <= result["changes"]["gained"]


def test_mixing_is_a_convex_combination_with_an_exact_zero():
    rng = np.random.default_rng(1)
    existing = rng.dirichlet(np.ones(7), size=4)
    augmented = {"luar": rng.dirichlet(np.ones(7), size=4)}
    assert specialist_mix.mixed(existing, augmented, None, 0.0) is existing
    assert specialist_mix.mixed(existing, augmented, "luar", 0.0) is existing
    half = specialist_mix.mixed(existing, augmented, "luar", 0.5)
    np.testing.assert_allclose(half, (existing + augmented["luar"]) / 2)
    np.testing.assert_allclose(half.sum(axis=1), 1.0)


def test_inputs_are_checked(setup):
    with pytest.raises(ValueError, match="incumbent's"):
        specialist_mix.evaluate(setup["train"], setup["blocks"], ("styledistance",), setup["plan"], setup["current"],
                                setup["cache"], setup["root"] / "a", samples=10)
    with pytest.raises(ValueError, match="incumbent's"):
        specialist_mix.evaluate(setup["train"], setup["blocks"], ("missing",), setup["plan"], setup["current"],
                                setup["cache"], setup["root"] / "b", samples=10)
    stale = setup["cache"].assign(groups_sha256="other")
    with pytest.raises(ValueError, match="different row grouping"):
        specialist_mix.evaluate(setup["train"], setup["blocks"], ("luar",), setup["plan"], setup["current"], stale,
                                setup["root"] / "c", samples=10)
    (setup["root"] / "d").mkdir()
    (setup["root"] / "d" / "comparison.json").write_text("{}")
    with pytest.raises(ValueError, match="overwritten"):
        specialist_mix.evaluate(setup["train"], setup["blocks"], ("luar",), setup["plan"], setup["current"],
                                setup["cache"], setup["root"] / "d", samples=10)


def test_ties_keep_the_reference_then_the_smaller_a_then_the_first_encoder():
    grid = specialist_mix.options(("luar-mud", "lusr"))
    assert [specialist_mix.option_name(*option) for option in grid] == [
        "reference", "luar-mud@0.25", "lusr@0.25", "luar-mud@0.5", "lusr@0.5", "luar-mud@1", "lusr@1"]


def test_encoder_names_are_unique_and_archives_must_be_the_pinned_encoder(setup):
    blocks = dict(setup["blocks"], **{"styledistance+luar": setup["blocks"]["luar"]})
    for encoders in (("styledistance+luar",), ("luar", "luar")):
        with pytest.raises(ValueError, match="once"):
            specialist_mix.evaluate(setup["train"], blocks, encoders, setup["plan"], setup["current"],
                                    setup["cache"], setup["root"] / "names", samples=10)
    train = setup["train"]
    metadata = {"model_id": "rrivera1849/LUSR", "model_revision": specialist_mix.ENCODERS["lusr"]["model_revision"],
                "encoder_frozen": True, "feature_options": {"pooling": "mean", "max_length": 512, "normalization": "l2"}}
    path = setup["root"] / "lusr.npz"
    save_feature_archive(path, train["id"].tolist(), setup["blocks"]["luar"].astype(np.float32), metadata,
                         texts=train["text"].tolist())
    assert specialist_mix.load_encoder("lusr", path, train).shape == setup["blocks"]["luar"].shape
    with pytest.raises(ValueError, match="pinned revision"):
        specialist_mix.load_encoder("luar-mud", path, train)  # LUSR features under LUAR's name
    with pytest.raises(ValueError, match="Unknown encoder"):
        specialist_mix.load_encoder("luar", path, train)
