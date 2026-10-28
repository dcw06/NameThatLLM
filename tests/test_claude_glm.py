"""Claude-GLM specialist: mass-preserving redistribution, specialist-only training, inner-only weights."""

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score
from test_lora import tiny  # noqa: F401  (shared tiny-Qwen fixture)

from ntl import claude_glm, ensemble, lora, lora_final, neural
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.modeling import BaselineConfig, aligned_probabilities
from ntl.smoke import synthetic_frames


def test_redistribution_keeps_other_classes_and_s_zero_is_the_current_system():
    rng = np.random.default_rng(0)
    current = rng.dirichlet(np.ones(len(LABELS)), size=6)
    specialist = rng.dirichlet(np.ones(len(claude_glm.SPECIALIST)), size=6)
    np.testing.assert_array_equal(claude_glm.redistribute(current, specialist, 0.0), current)
    for weight in (0.25, 1.0):
        result = claude_glm.redistribute(current, specialist, weight)
        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        others = np.setdiff1d(np.arange(len(LABELS)), claude_glm.INDEX)
        np.testing.assert_array_equal(result[:, others], current[:, others])
        np.testing.assert_allclose(result[:, claude_glm.INDEX].sum(axis=1), current[:, claude_glm.INDEX].sum(axis=1))
    mass = current[:, claude_glm.INDEX].sum(axis=1, keepdims=True)
    np.testing.assert_allclose(claude_glm.redistribute(current, specialist, 1.0)[:, claude_glm.INDEX], mass * specialist)


def test_candidate_scales_only_the_appended_block():
    head = claude_glm.specialist_head((8, 2), neural.FrozenConfig())
    weights = head.named_steps["balance"].kw_args["weights"]
    np.testing.assert_allclose(weights, [1.0] * 8 + [2.0] * 2)  # sqrt(8 / 2): equal total variance
    control = claude_glm.specialist_head((8,), neural.FrozenConfig())
    np.testing.assert_array_equal(control.named_steps["balance"].kw_args["weights"], np.ones(8))


@pytest.fixture
def setup(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    manifest_path = save_fold_manifest(manifest, metadata, tmp_path / "folds")
    rng = np.random.default_rng(3)
    truth = ensemble._targets(train)
    qwen = rng.normal(size=(len(train), 8)).astype(np.float32)
    qwen[:, 0] += truth  # signal for every head
    styledistance = rng.normal(size=(len(train), 4)).astype(np.float32)
    styledistance[:, 0] += np.isin(truth, claude_glm.INDEX) * truth  # extra signal inside the group
    config = ensemble.BlendConfig(cpu=BaselineConfig(char_max_features=300, word_max_features=100, max_iter=200),
                                  bootstrap_samples=20)
    ensemble.evaluate_blend(train, qwen, manifest_path, tmp_path / "current", config)
    plan = lora.split_plan(train, manifest_path, lora.AdapterConfig())
    cache = pd.concat([claude_glm.build_current_inner(train, qwen, plan, fold, config) for fold in range(5)],
                      ignore_index=True)
    current = pd.read_parquet(tmp_path / "current" / "oof.parquet")
    return {"train": train, "blocks": {"qwen": qwen, "styledistance": styledistance}, "plan": plan,
            "current": current, "cache": cache, "root": tmp_path}


def test_rows_outside_the_group_never_reach_the_specialist(setup):
    train, blocks = setup["train"], setup["blocks"]
    truth = ensemble._targets(train)
    prompts = train["prompt_id"].to_numpy()
    first_prompts = np.isin(prompts, np.unique(prompts)[:6])
    fit, rows = np.flatnonzero(first_prompts), np.flatnonzero(~first_prompts)
    outside = np.setdiff1d(fit, fit[np.isin(truth[fit], claude_glm.INDEX)])
    changed = {name: values.copy() for name, values in blocks.items()}
    for values in changed.values():
        values[outside] = 1e3
    for name in claude_glm.FEATURE_SETS:
        first = claude_glm.specialist_probabilities(blocks, name, truth, fit, rows, neural.FrozenConfig())
        second = claude_glm.specialist_probabilities(changed, name, truth, fit, rows, neural.FrozenConfig())
        np.testing.assert_array_equal(first, second)
        assert first.shape == (len(rows), len(claude_glm.SPECIALIST))


def test_specialists_fit_only_training_splits_and_every_row_is_scored(setup, monkeypatch):
    train = setup["train"]
    plan = setup["plan"].set_index("id").loc[train["id"]]
    folds = plan["outer_fold"].to_numpy()
    splits = []  # every legitimate (fit, scored) pair: outer folds, then their inner folds
    for fold in range(5):
        splits.append((set(np.flatnonzero(folds != fold)), set(np.flatnonzero(folds == fold))))
        column = plan[f"inner_{fold}"].to_numpy()
        for inner_fold in (0, 1):
            splits.append((set(np.flatnonzero((folds != fold) & (column != inner_fold))),
                           set(np.flatnonzero(column == inner_fold))))
    calls, original = [], claude_glm.specialist_probabilities

    def recorded(blocks, feature_set, truth, fit, rows, frozen):
        calls.append((set(fit), set(rows)))
        return original(blocks, feature_set, truth, fit, rows, frozen)

    monkeypatch.setattr(claude_glm, "specialist_probabilities", recorded)
    result = claude_glm.evaluate(train, setup["blocks"], setup["plan"], setup["current"], setup["cache"],
                                 setup["root"] / "compare", samples=20)
    assert len(calls) == len(claude_glm.FEATURE_SETS) * len(splits)
    assert all(call in splits for call in calls)
    current = setup["current"].set_index("id").loc[train["id"]]
    assert result["systems"]["current"]["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(
        ensemble._targets(train), current[claude_glm.BLEND].to_numpy().argmax(axis=1)))
    predictions = pd.read_parquet(setup["root"] / "compare" / "predictions.parquet")
    assert len(predictions) == len(train) and predictions["id"].is_unique
    for name, items in result["selection"].items():
        specialist = np.load(setup["root"] / "compare" / f"specialist-{name}.npy")
        for item in items:
            assert item["weight"] in claude_glm.WEIGHTS
            rows = (predictions["outer_fold"] == item["outer_fold"]).to_numpy()
            expected = claude_glm.redistribute(current[claude_glm.BLEND].to_numpy()[rows], specialist[rows],
                                               item["weight"]).argmax(axis=1)
            assert (predictions.loc[rows, f"prediction::current+{name}"] == np.asarray(LABELS)[expected]).all()
            if item["weight"] == 0:  # s = 0 is the current system exactly
                assert (predictions.loc[rows, f"prediction::current+{name}"]
                        == predictions.loc[rows, "prediction::current"]).all()
    assert {"current+qwen_vs_current", "current+qwen+styledistance_vs_current",
            "current+qwen+styledistance_vs_current+qwen"} <= set(result["paired"])


def test_inner_cache_and_current_folds_are_checked(setup):
    train = setup["train"]
    with pytest.raises(ValueError, match="cover each inner fold"):
        claude_glm.evaluate(train, setup["blocks"], setup["plan"], setup["current"], setup["cache"].iloc[1:],
                            setup["root"] / "short", samples=10)
    rotated = setup["current"].assign(fold=(setup["current"]["fold"] + 1) % 5)
    with pytest.raises(ValueError, match="outer folds"):
        claude_glm.evaluate(train, setup["blocks"], setup["plan"], rotated, setup["cache"],
                            setup["root"] / "rotated", samples=10)


def test_regrouping_moves_whole_prompts_and_every_input_must_share_the_grouping(setup):
    train = setup["train"]
    # Topics that the plan's folds keep together: prompts sharing every fold assignment.
    columns = ["outer_fold"] + [f"inner_{fold}" for fold in range(5)]
    keys = setup["plan"].set_index("id").loc[train["id"], columns].astype(str).agg("-".join, axis=1)
    groups = train[["id"]].assign(prompt_id=("topic-" + keys).to_numpy())
    regrouped = claude_glm.regroup(train, groups)
    assert regrouped["prompt_id"].nunique() <= train["prompt_id"].nunique()
    assert claude_glm.groups_sha256(regrouped) != claude_glm.groups_sha256(train)
    split = groups.copy()
    split.loc[split.index[0], "prompt_id"] = "elsewhere"
    with pytest.raises(ValueError, match="together"):
        claude_glm.regroup(train, split)
    with pytest.raises(ValueError, match="every training row"):
        claude_glm.regroup(train, groups.iloc[1:])
    plan = setup["plan"].assign(prompt_id=setup["plan"]["id"].map(regrouped.set_index("id")["prompt_id"]))
    # Prompt-grouped caches and current OOF never serve a regrouped evaluation.
    with pytest.raises(ValueError, match="different row grouping"):
        claude_glm.evaluate(regrouped, setup["blocks"], plan, setup["current"], setup["cache"],
                            setup["root"] / "regrouped", samples=10)
    cache = setup["cache"].assign(groups_sha256=claude_glm.groups_sha256(regrouped))
    with pytest.raises(ValueError, match="same row grouping"):
        claude_glm.evaluate(regrouped, setup["blocks"], plan, setup["current"], cache,
                            setup["root"] / "regrouped", samples=10)


def test_an_existing_comparison_is_never_overwritten(setup):
    (setup["root"] / "done").mkdir()
    (setup["root"] / "done" / "comparison.json").write_text("{}")
    with pytest.raises(ValueError, match="overwritten"):
        claude_glm.evaluate(setup["train"], setup["blocks"], setup["plan"], setup["current"], setup["cache"],
                            setup["root"] / "done", samples=10)


def test_final_path_redistributes_the_fitted_current_system(tiny, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ensemble, "HIDDEN_SIZE", 16)  # the tiny backbone's width
    train, cached, frozen = tiny["train"], tiny["cached"], tiny["frozen"]
    style = np.random.default_rng(1).normal(size=(len(train), 4)).astype(np.float32)
    lookup = dict(zip(train["text"], style, strict=True))
    extracted = []

    def embeddings(texts, model_path, model_key, device="cpu", **_):
        extracted.append(len(texts))
        return np.stack([lookup[text] for text in texts]), {"profile": {"rows": len(texts)}}

    monkeypatch.setattr(claude_glm, "extract_embeddings", embeddings)
    blend = ensemble.BlendConfig(cpu=BaselineConfig(char_max_features=300, word_max_features=100, max_iter=200),
                                 qwen=frozen, bootstrap_samples=10)
    current_recipe = json.loads(json.dumps(ensemble.freeze_recipe(train, cached, blend)))
    ensemble.fit_final(train, tiny["model"], current_recipe, tmp_path / "blend", device="cpu")
    plan = lora_final.final_plan(train, lora.AdapterConfig())
    recipe = claude_glm.freeze_recipe(train, {"qwen": cached, "styledistance": style}, plan,
                                      claude_glm.recipe_sha256(current_recipe), frozen=frozen, blend=blend)
    assert recipe["weight"] in claude_glm.WEIGHTS and len(recipe["current_inner_weights"]) == 2
    recipe["weight"] = 0.5  # exercise the redistribution whatever the tiny data choose
    claude_glm.fit_final(train, tmp_path / "blend", "snapshot", recipe, tmp_path / "specialist", device="cpu")
    with pytest.raises(ValueError, match="overwritten"):
        claude_glm.fit_final(train, tmp_path / "blend", "snapshot", recipe, tmp_path / "specialist", device="cpu")
    test = train[["id", "text"]].sample(n=12, random_state=2).reset_index(drop=True)
    submission = claude_glm.predict_final(tmp_path / "specialist", tmp_path / "blend", test, tiny["model"],
                                          "snapshot", tmp_path / "submission.parquet", device="cpu")
    assert extracted == [len(train), len(test)]
    fitted = joblib.load(tmp_path / "blend" / "blend.joblib")
    specialist = joblib.load(tmp_path / "specialist" / "specialist.joblib")
    qwen, _ = neural.extract_features(test["text"].tolist(), tiny["model"], frozen)
    current = ensemble._blend(aligned_probabilities(fitted["cpu"], test["text"]),
                              neural._probabilities(fitted["qwen_head"], qwen), current_recipe["weight"])
    q = specialist["head"].predict_proba(np.hstack([qwen, np.stack([lookup[text] for text in test["text"]])]))
    expected = np.asarray(LABELS)[claude_glm.redistribute(current, q, 0.5).argmax(axis=1)]
    assert submission["id"].tolist() == test["id"].tolist()
    assert submission.iloc[:, 1].tolist() == expected.tolist()
    # The specialist belongs to one current system: another fitted blend is refused.
    stale = dict(recipe, current_recipe_sha256="0" * 64)
    joblib.dump(dict(specialist, recipe=stale), tmp_path / "specialist" / "specialist.joblib")
    with pytest.raises(ValueError, match="current system differs"):
        claude_glm.predict_final(tmp_path / "specialist", tmp_path / "blend", test, tiny["model"], "snapshot",
                                 tmp_path / "again.parquet", device="cpu")


def test_the_command_line_runs_the_imported_module_so_fitted_heads_load_anywhere():
    # `python -m ntl.claude_glm` must call the importable module's main: a head pickled
    # under __main__ could only be loaded by that same command.
    script = (
        "import runpy, sys\n"
        "import ntl.claude_glm as module\n"
        "module.main = lambda argv=None: print('imported main') or 0\n"
        "sys.argv = ['ntl.claude_glm']\n"
        "try:\n"
        "    runpy.run_module('ntl.claude_glm', run_name='__main__')\n"
        "except SystemExit as stop:\n"
        "    assert stop.code == 0\n"
    )
    source = str(Path(claude_glm.__file__).parents[1])
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                            env={**os.environ, "PYTHONPATH": source})
    assert result.stdout.strip() == "imported main"
    head = claude_glm.specialist_head((2, 1), neural.FrozenConfig())
    assert head.named_steps["balance"].func.__module__ == "ntl.claude_glm"
