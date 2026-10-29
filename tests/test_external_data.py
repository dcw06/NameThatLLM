"""External training data: verified identifiers, topic links that keep validation clean, the
per-class cap, and a blend that reproduces the existing one exactly without external rows."""

import numpy as np
import pandas as pd

from ntl import claude_glm, ensemble, external_data
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, load_fold_manifest, save_fold_manifest
from ntl.modeling import BaselineConfig
from ntl.smoke import synthetic_frames


def test_only_rows_whose_served_model_matches_their_label_are_verified():
    frame = pd.DataFrame({"model": ["glm-5.3", "glm-5.3", "gpt-5.5", "gpt-5.6-luna", "kimi-k3"],
                          "served_model": ["z-ai/glm-5.3", "z-ai/glm-5.3-flash", "", "gpt-5.6-luna", None]})
    assert external_data.verified(frame).tolist() == [True, False, False, True, False]


def _texts():
    train = pd.DataFrame({
        "id": [f"t{i}" for i in range(6)], "prompt_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
        "model": [LABELS[0], LABELS[1]] * 3,
        # A shared word gives competition prompts some cross-prompt similarity, as real passages have.
        "text": ["explain the monty hall door goat car switch probability host", "explain monty hall switching doors goat car odds",
                 "explain photosynthesis chlorophyll sunlight plants glucose leaves", "explain how plants turn sunlight into glucose",
                 "explain volcano magma eruption lava ash crater", "explain lava flows from erupting volcano magma ash"]})
    external = pd.DataFrame({
        "row_id": ["e1", "e2", "e3", "e4"], "prompt_id": ["x1", "x1", "x2", "x3"],
        "model": ["glm-5.3", "glm-5.3", "glm-5.3-flash", "kimi-k3"],
        "text": ["monty hall goat car switch doors host probability", "a quiet harbour at dawn with fishing boats",
                 "the stock market and interest rates", "glaciers carve valleys over millennia slowly"]})
    return train, external


def test_topic_links_tie_whole_external_prompts_to_the_matched_competition_prompt():
    train, external = _texts()
    link, stats = external_data.topic_links(train, external)
    # e1 answers p1's topic, and e2 shares e1's external prompt, so both follow the link.
    assert link.tolist() == ["p1", "p1", "", ""] and stats["linked_rows"] == 2
    assert 0 < stats["tau"] < 1


def test_eligible_rows_respect_links_the_cap_and_the_fixed_order():
    pool = external_data.Pool(ids=np.array(["a", "b", "c", "d", "e"], dtype=object),
                              texts=np.array(list("abcde"), dtype=object),
                              labels=np.array(["glm-5.3"] * 4 + ["kimi-k3"], dtype=object),
                              link=np.array(["p1", "", "", "", ""], dtype=object), rank=np.array([0, 3, 1, 2, 0]))
    labels = np.array(["glm-5.3", "glm-5.3", "kimi-k3", "gpt-5.5"], dtype=object)
    # p1 is not in the fit, so row a is held back; glm-5.3 may add two rows, by rank (c, then d); kimi one.
    assert external_data.eligible_rows(pool, labels, np.array(["p2", "p2", "p3", "p3"])).tolist() == [2, 3, 4]
    # With p1 in the fit, a (rank 0) comes first.
    assert external_data.eligible_rows(pool, labels, np.array(["p1", "p2", "p3", "p3"])).tolist() == [0, 2, 4]
    assert external_data.eligible_rows(pool, labels, np.array(["p1", "p2", "p3", "p3"]), cap=0.5).tolist() == [0]
    assert external_data.eligible_rows(external_data.Pool.empty(), labels, np.array(["p1"] * 4)).size == 0


def test_the_augmented_fit_appends_only_the_eligible_rows(monkeypatch):
    seen = {}
    monkeypatch.setattr(ensemble, "_cpu_probabilities",
                        lambda texts, labels, rows, config: seen.update(texts=list(texts), labels=list(labels)) or 1)
    pool = external_data.Pool(ids=np.array(["a"], dtype=object), texts=np.array(["extra"], dtype=object),
                              labels=np.array(["glm-5.3"], dtype=object), link=np.array([""], dtype=object),
                              rank=np.array([0]))
    usage = []
    cpu = external_data.augmented_cpu(pool, usage)
    cpu(np.array(["one"], dtype=object), np.array(["glm-5.3"], dtype=object), np.array(["p"]), ["x"], None)
    assert seen == {"texts": ["one", "extra"], "labels": ["glm-5.3", "glm-5.3"]}
    assert usage == [{"fit_rows": 1, "external_rows": 1, "by_label": {"glm-5.3": 1}}]
    cpu(np.array(["one"], dtype=object), np.array(["kimi-k3"], dtype=object), np.array(["p"]), ["x"], None)
    assert seen["texts"] == ["one"]  # no GLM row in this fit, so no GLM row may be added


def test_without_external_rows_the_outer_blend_reproduces_the_existing_one(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    path = save_fold_manifest(manifest, metadata, tmp_path / "folds")
    features = np.random.default_rng(2026).normal(size=(len(train), 8)).astype(np.float32)
    config = ensemble.BlendConfig(cpu=BaselineConfig(char_max_features=500, word_max_features=200, max_iter=300),
                                  bootstrap_samples=20)
    ensemble.evaluate_blend(train, features, path, tmp_path / "blend", config)
    existing = pd.read_parquet(tmp_path / "blend" / "oof.parquet").set_index("id")
    folds = train["id"].map(load_fold_manifest(train, path)[0].set_index("id")["fold"]).to_numpy()
    for fold in (0, 3):
        frame, _ = external_data.outer_blend(train, features, folds, fold,
                                             external_data.augmented_cpu(external_data.Pool.empty()), config)
        expected = existing.loc[frame["id"], claude_glm.BLEND].to_numpy()
        assert np.array_equal(frame[claude_glm.BLEND].to_numpy(), expected)
        assert (frame["fold"] == fold).all() and frame["true_model"].tolist() == existing.loc[frame["id"], "true_model"].tolist()


def test_the_default_cpu_hook_is_the_existing_fit():
    assert ensemble._select_weight.__defaults__[-1] is ensemble.default_cpu
