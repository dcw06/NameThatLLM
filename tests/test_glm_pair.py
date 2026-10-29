"""GLM pair: matched external passages, fold-aware exclusions, and control and augmented models
fit on exactly the intended passages."""

import numpy as np
import pandas as pd
import pytest

from ntl import ensemble, glm_pair, lora
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.modeling import BaselineConfig, build_pipeline
from ntl.smoke import synthetic_frames


def _external():
    rows = [  # row_id, prompt, label, source tag, provider, served model
        ("a1", "x1", "glm-5.3", "zaiglm", "Z.AI", "z-ai/glm-5.3"),
        ("a2", "x1", "glm-5.3-flash", "zaiglm", "Z.AI", "z-ai/glm-5.3-flash"),
        ("a3", "x1", "glm-5.3-flash", "zaiglm", "Z.AI", "z-ai/glm-5.3-flash"),  # surplus: x1 keeps one per version
        ("b1", "x2", "glm-5.3", "zaiglm", "Z.AI", "z-ai/glm-5.3"),  # no Flash passage: x2 is not shared
        ("c1", "x3", "glm-5.3", "orglm", "Relace", "z-ai/glm-5.3"),  # other provider
        ("c2", "x3", "glm-5.3-flash", "orglm", "OpenInference", "z-ai/glm-5.3-flash"),
        ("d1", "x4", "glm-5.3", "zaiglm", "Z.AI", "z-ai/glm-5.3"),
        ("d2", "x4", "glm-5.3-flash", "zaiglm", "Z.AI", ""),  # unverified, so x4 is not shared
        ("e1", "x5", "glm-5.3", "zaiglm", "Z.AI", "z-ai/glm-5.3"),
        ("e2", "x5", "glm-5.3-flash", "zaiglm", "Z.AI", "z-ai/glm-5.3-flash"),
    ]
    frame = pd.DataFrame(rows, columns=["row_id", "prompt_id", "model", "source_tag", "provider", "served_model"])
    return frame.assign(text=[f"passage {row_id} words" for row_id in frame["row_id"]])


def test_matched_passages_share_prompts_provider_and_verified_identifiers_in_equal_numbers():
    matched = glm_pair.matched_external(_external(), pd.Series({"e1": "p3", "e2": "p3"}))
    assert matched["row_id"].tolist() == ["a1", "a2", "e1", "e2"]
    assert matched["model"].value_counts().to_dict() == {"glm-5.3": 2, "glm-5.3-flash": 2}
    assert matched["link"].tolist() == ["", "", "p3", "p3"]


def test_linked_passages_are_used_only_when_their_prompt_is_on_the_training_side():
    matched = pd.DataFrame({"link": ["", "p3", "p3", "zz"]})
    prompt_fold = pd.Series({"p1": 0, "p3": 2})
    assert glm_pair.eligible(matched, prompt_fold, 2).tolist() == [True, False, False, False]
    assert glm_pair.eligible(matched, prompt_fold, 0).tolist() == [True, True, True, False]  # unknown links never


@pytest.fixture
def setup(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    plan = lora.split_plan(train, save_fold_manifest(manifest, metadata, tmp_path / "folds"), lora.AdapterConfig())
    prompts = sorted(set(train["prompt_id"]))
    external = pd.DataFrame({
        "row_id": [f"r{i}" for i in range(8)], "prompt_id": ["x1", "x1", "x2", "x2", "x3", "x3", "x4", "x4"],
        "model": ["glm-5.3", "glm-5.3-flash"] * 4, "source_tag": "zaiglm", "provider": "Z.AI",
        "served_model": ["z-ai/glm-5.3", "z-ai/glm-5.3-flash"] * 4,
        "text": ["glm five three external alpha", "glm flash external beta"] * 4})
    links = pd.Series({"r2": prompts[0], "r3": prompts[0]})  # x2 is linked to the first competition prompt
    rng = np.random.default_rng(0)
    locked = rng.dirichlet(np.ones(len(LABELS)), size=len(train))
    return train, plan, glm_pair.matched_external(external, links), locked, tmp_path


def test_the_control_and_augmented_models_are_fit_on_exactly_their_passages(setup):
    train, plan, matched, locked, root = setup
    result = glm_pair.evaluate(train, plan, matched, locked, root / "out")
    scores = pd.read_parquet(root / "out" / "scores.parquet").set_index("id")
    truth = ensemble._targets(train)
    folds = train["id"].map(plan.set_index("id")["outer_fold"]).to_numpy()
    pair = np.isin(truth, glm_pair.PAIR_INDEX)
    target = (truth == glm_pair.PAIR_INDEX[1]).astype(int)
    texts = train["text"].to_numpy(dtype=object)
    linked_fold = folds[(train["prompt_id"] == train["prompt_id"].min()).to_numpy()][0]
    for fold in range(5):
        fit, rows = np.flatnonzero(pair & (folds != fold)), np.flatnonzero(pair & (folds == fold))
        model = build_pipeline(BaselineConfig(seed=2026)).fit(texts[fit], target[fit])
        np.testing.assert_allclose(scores.loc[train["id"].to_numpy()[rows], "p_flash::control"],
                                   model.predict_proba(texts[rows])[:, 1])
        use = (matched["link"] == "").to_numpy() | (fold != linked_fold)
        external = matched[use]
        model = build_pipeline(BaselineConfig(seed=2026)).fit(
            np.concatenate([texts[fit], external["text"].to_numpy(dtype=object)]),
            np.concatenate([target[fit], (external["model"] == "glm-5.3-flash").to_numpy().astype(int)]))
        np.testing.assert_allclose(scores.loc[train["id"].to_numpy()[rows], "p_flash::augmented"],
                                   model.predict_proba(texts[rows])[:, 1])
        assert result["external"][fold]["external_rows"] == int(use.sum())
        assert result["external"][fold]["held_back_linked_rows"] == (2 if fold == linked_fold else 0)
    share = locked[:, glm_pair.PAIR_INDEX[1]] / locked[:, glm_pair.PAIR_INDEX].sum(axis=1)
    assert result["pooled"]["incumbent_forced"] == glm_pair.summary(target[pair], share[pair])
    assert result["gate"]["passes"] == (
        result["pooled"]["augmented"]["balanced_accuracy"] > result["pooled"]["control"]["balanced_accuracy"]
        and result["pooled"]["augmented"]["balanced_accuracy"] > result["pooled"]["incumbent_forced"]["balanced_accuracy"])
    with pytest.raises(ValueError, match="new output directory"):
        glm_pair.evaluate(train, plan, matched, locked, root / "out")


def test_summaries_count_swaps_and_corrections():
    target = np.array([0, 0, 1, 1])
    first, second = np.array([0.9, 0.1, 0.2, 0.8]), np.array([0.1, 0.9, 0.8, 0.8])
    assert glm_pair.summary(target, first)["swaps"] == {"glm-5.3->glm-5.3-flash": 1, "glm-5.3-flash->glm-5.3": 1}
    result = glm_pair.paired(target, first, second, np.array(["a", "a", "b", "b"]))
    assert result["corrected"] == 2 and result["introduced"] == 1  # rows 0 and 2 fixed, row 1 broken
