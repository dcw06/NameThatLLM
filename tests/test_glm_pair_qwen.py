"""GLM pair on Qwen + StyleDistance: both classifiers use the specialist recipe on exactly the
intended passages, and the gate needs an interval above zero."""

import numpy as np
import pandas as pd
import pytest

from ntl import claude_glm, ensemble, glm_pair, glm_pair_qwen, lora, neural
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.smoke import synthetic_frames


@pytest.fixture
def setup(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    plan = lora.split_plan(train, save_fold_manifest(manifest, metadata, tmp_path / "folds"), lora.AdapterConfig())
    rng = np.random.default_rng(1)
    truth = ensemble._targets(train)
    width = sum(glm_pair_qwen.WIDTHS)
    official = rng.normal(size=(len(train), width)).astype(np.float32)
    official[:, 0] += 1.5 * (truth == glm_pair.PAIR_INDEX[1])  # the versions are partly separable
    matched = pd.DataFrame({"row_id": [f"r{i}" for i in range(8)], "prompt_id": np.repeat(["x1", "x2", "x3", "x4"], 2),
                            "model": list(glm_pair.PAIR) * 4, "text": "external",
                            "link": ["", "", train["prompt_id"].min(), train["prompt_id"].min(), "", "", "", ""]})
    external = rng.normal(size=(len(matched), width)).astype(np.float32)
    external[:, 0] += 1.5 * (matched["model"] == glm_pair.PAIR[1]).to_numpy()
    locked = rng.dirichlet(np.ones(len(LABELS)), size=len(train))
    return train, plan, official, matched, external, locked, tmp_path


def test_control_and_augmented_use_the_specialist_recipe_on_their_passages(setup):
    train, plan, official, matched, external, locked, root = setup
    result = glm_pair_qwen.evaluate(train, plan, official, matched, external, locked, root / "out")
    scores = pd.read_parquet(root / "out" / "scores.parquet").set_index("id")
    truth = ensemble._targets(train)
    folds = train["id"].map(plan.set_index("id")["outer_fold"]).to_numpy()
    pair = np.isin(truth, glm_pair.PAIR_INDEX)
    target = (truth == glm_pair.PAIR_INDEX[1]).astype(int)
    linked_fold = folds[(train["prompt_id"] == train["prompt_id"].min()).to_numpy()][0]
    for fold in range(5):
        fit, rows = np.flatnonzero(pair & (folds != fold)), np.flatnonzero(pair & (folds == fold))
        head = claude_glm.specialist_head(glm_pair_qwen.WIDTHS, neural.FrozenConfig()).fit(official[fit], target[fit])
        np.testing.assert_allclose(scores.loc[train["id"].to_numpy()[rows], "p_flash::control"],
                                   head.predict_proba(official[rows])[:, 1], rtol=0, atol=1e-10)
        use = (matched["link"] == "").to_numpy() | (fold != linked_fold)
        head = claude_glm.specialist_head(glm_pair_qwen.WIDTHS, neural.FrozenConfig()).fit(
            np.vstack([official[fit], external[use]]),
            np.concatenate([target[fit], (matched["model"] == glm_pair.PAIR[1]).to_numpy()[use].astype(int)]))
        np.testing.assert_allclose(scores.loc[train["id"].to_numpy()[rows], "p_flash::augmented"],
                                   head.predict_proba(official[rows])[:, 1], rtol=0, atol=1e-10)
        assert result["external"][fold]["external_rows"] == int(use.sum())
    gate = result["gate"]["passes"]
    assert gate == (result["augmented_vs_control"]["interval"][0] > 0
                    and result["pooled"]["augmented"]["balanced_accuracy"]
                    > result["pooled"]["incumbent_forced"]["balanced_accuracy"])


def test_features_must_be_qwen_then_styledistance(setup):
    train, plan, official, matched, external, locked, root = setup
    with pytest.raises(ValueError, match="Qwen"):
        glm_pair_qwen.evaluate(train, plan, official[:, :100], matched, external, locked, root / "bad")
