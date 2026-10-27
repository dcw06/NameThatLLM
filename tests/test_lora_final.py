"""Final adapted model: full-data choice, saved adapters, and scoring consistent with the current system."""

import json

import numpy as np
import pandas as pd
from test_lora import tiny  # noqa: F401  (shared tiny-Qwen fixture)

from ntl import ensemble, lora, lora_final
from ntl.modeling import BaselineConfig


def test_final_pipeline_chooses_on_inner_rows_saves_and_scores_like_the_current_system(
        tiny, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ensemble, "HIDDEN_SIZE", 16)  # the tiny backbone's width
    train, cached, frozen, config = tiny["train"], tiny["cached"], tiny["frozen"], tiny["config"]
    plan = lora_final.final_plan(train, config)
    assert set(plan["inner"]) == {0, 1} and plan.groupby("prompt_id")["inner"].nunique().max() == 1
    experiment = lora.Experiment(train, cached, adapters=True, model_path=tiny["model"], frozen=frozen,
                                 config=config, device="cpu")
    for inner_fold in (0, 1):
        lora_final.run_inner(train, plan, experiment, inner_fold, tmp_path / "select")
    inner = pd.concat([pd.read_parquet(path) for path in sorted((tmp_path / "select").glob("inner*.parquet"))])
    blend = ensemble.BlendConfig(cpu=BaselineConfig(char_max_features=300, word_max_features=100, max_iter=200),
                                 qwen=frozen, bootstrap_samples=10)
    recipe = lora_final.choose(train, cached, plan, inner, config, blend)
    assert recipe["setting"] in {lora.setting_name(*setting) for setting in config.settings()}
    assert recipe["adapted_weight"] in (0.0, 0.25, 0.5, 0.75, 1.0)
    recipe.update(learning_rate=1e-2, duration=1.0)  # exercise adapter saving regardless of the choice
    log = lora_final.fit_all(train, experiment, recipe, tmp_path / "final")
    assert log["rows"] == len(train) and (tmp_path / "final" / "adapter" / "adapter_config.json").is_file()
    _, base, _, _, _ = lora_final.neural.load_backbone(tiny["model"], frozen)
    reloaded = dict(lora_final.load_adapter(base, tmp_path / "final" / "adapter", "cpu").named_parameters())
    trained = {name: value for name, value in experiment.model.named_parameters() if "lora_" in name}
    assert trained and all(tiny["torch"].equal(reloaded[name], value) for name, value in trained.items())

    current_recipe = ensemble.freeze_recipe(train, cached, blend)
    current_recipe["weight"] = 0.25
    ensemble.fit_final(train, tiny["model"], current_recipe, tmp_path / "blend", device="cpu")
    test = train[["id", "text"]].sample(n=9, random_state=2).reset_index(drop=True)
    test["id"] = [f"hidden-{index}" for index in range(len(test))]
    expected = ensemble.predict_final(tmp_path / "blend", test, tiny["model"], tmp_path / "current.parquet", "cpu")
    # With the adapted weight at zero the final system must be the current system exactly.
    stored = json.loads((tmp_path / "final" / "recipe.json").read_text())
    (tmp_path / "final" / "recipe.json").write_text(json.dumps({**stored, "adapted_weight": 0.0}))
    zero = lora_final.predict(tmp_path / "final", tmp_path / "blend", test, tiny["model"],
                              tmp_path / "zero.parquet", "cpu")
    assert zero["model"].tolist() == expected["model"].tolist()
    (tmp_path / "final" / "recipe.json").write_text(json.dumps({**stored, "adapted_weight": 1.0}))
    adapted = lora_final.predict(tmp_path / "final", tmp_path / "blend", test, tiny["model"],
                                 tmp_path / "adapted.parquet", "cpu")
    assert adapted["id"].tolist() == test["id"].tolist() and len(adapted) == 9
    profile = json.loads((tmp_path / "adapted_profile.json").read_text())
    assert profile["adapted_weight"] == 1.0 and np.isfinite(profile["elapsed_seconds"])
