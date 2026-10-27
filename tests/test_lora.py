"""QLoRA adaptation: frozen-recipe starting point, inner-only epoch choice, deterministic resets."""

import json

import numpy as np
import pandas as pd
import pytest

from ntl import lora, neural
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.smoke import synthetic_frames

VOCABULARY = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, ".": 3} | {
    f"ClassMarker{index:02d}": 4 + index for index in range(len(LABELS))}


def save_tokenizer(directory):
    """A word-level tokenizer over the synthetic vocabulary, saved like a snapshot's tokenizer."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    core = Tokenizer(WordLevel(VOCABULARY, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=core, pad_token="[PAD]", eos_token="[EOS]",
                            unk_token="[UNK]").save_pretrained(directory)
    return directory


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(5)
    architecture = Qwen3Config(
        vocab_size=len(VOCABULARY), hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        attention_dropout=0.0, pad_token_id=0, eos_token_id=1,
    )
    Qwen3ForCausalLM(architecture).save_pretrained(tmp_path / "model")
    save_tokenizer(tmp_path / "model")
    monkeypatch.setattr(neural, "validate_snapshot",
                        lambda *_, **__: {"repo_id": "synthetic", "revision": "synthetic"})
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    manifest_path = save_fold_manifest(manifest, metadata, tmp_path / "folds")
    frozen = neural.FrozenConfig(retained_layers=3, pool_layers=(1, 2), quantization="none",
                                 batch_size=2, max_length=16, device="cpu")
    cached, _ = neural.extract_features(train["text"].tolist(), tmp_path / "model", frozen)
    config = lora.AdapterConfig(rank=2, alpha=4, adapted_layers=2, micro_batch=4, accumulation=2,
                                warmup_steps=2, durations=(0.0, 0.5, 1.0), learning_rates=(1e-3, 1e-2))
    plan = lora.split_plan(train, manifest_path, config)
    return {"torch": torch, "train": train, "plan": plan, "cached": cached, "frozen": frozen,
            "config": config, "model": tmp_path / "model"}


def test_plan_inner_folds_partition_each_outer_training_partition_by_prompt(tiny):
    plan = lora.validate_plan(tiny["train"], tiny["plan"], tiny["config"])
    for fold in range(5):
        column = plan[f"inner_{fold}"]
        assert (column[plan["outer_fold"] == fold] == -1).all()
        assert set(column[plan["outer_fold"] != fold]) == {0, 1}
        assert plan.loc[plan["outer_fold"] != fold].groupby("prompt_id")[f"inner_{fold}"].nunique().max() == 1
    broken = tiny["plan"].copy()
    broken.loc[broken.index[0], "inner_0"] = 7
    with pytest.raises(ValueError):
        lora.validate_plan(tiny["train"], broken, tiny["config"])
    # One row of a fold-0 prompt moved to fold 1, with consistent inner columns: only the
    # outer-fold check can catch this leak.
    leaky = tiny["plan"].copy()
    row = leaky.index[leaky["outer_fold"] == 0][0]
    leaky.loc[row, ["outer_fold", "inner_0", "inner_1"]] = [1, 0, -1]
    with pytest.raises(ValueError, match="Outer folds must keep prompts together"):
        lora.validate_plan(tiny["train"], leaky, tiny["config"])
    # Fold 4 merged into fold 3, with consistent inner columns: only the fold-value check
    # catches the missing fold.
    merged = tiny["plan"].copy()
    rows = merged["outer_fold"] == 4
    merged.loc[rows, ["outer_fold", "inner_3", "inner_4"]] = [3, -1, 0]
    with pytest.raises(ValueError, match="exactly the five folds"):
        lora.validate_plan(tiny["train"], merged, tiny["config"])


def test_head_starts_exactly_at_the_frozen_pipeline(tiny):
    torch = tiny["torch"]
    targets = lora.ensemble._targets(tiny["train"])
    probe = lora.fit_probe(tiny["cached"], targets, tiny["frozen"])
    head = lora.head_module(torch, probe, "cpu")
    with torch.no_grad():
        mine = torch.softmax(head(torch.as_tensor(tiny["cached"])), dim=-1).numpy()
    np.testing.assert_allclose(mine, probe["pipeline"].predict_proba(tiny["cached"]), atol=1e-5)


def test_adapters_start_at_frozen_features_train_and_reset_deterministically(tiny):
    torch = tiny["torch"]
    experiment = lora.Experiment(tiny["train"], tiny["cached"], adapters=True, model_path=tiny["model"],
                                 frozen=tiny["frozen"], config=tiny["config"], device="cpu")
    assert experiment.trainable_parameters() > 0
    rows = np.arange(24)
    experiment.model.eval()
    with torch.no_grad():
        initial = experiment._pooled(rows).numpy()
    np.testing.assert_allclose(initial, tiny["cached"][rows], atol=1e-5)
    first = experiment.run(rows, rows, 1e-2, [0.5, 1.0])
    moved = [name for name, value in experiment.model.named_parameters()
             if value.requires_grad and not torch.equal(value, experiment.initial[name])]
    assert moved and all("lora_" in name for name in moved)
    assert all(".layers.2." not in name for name in experiment.initial)  # blocks after pooling are untouched
    second = experiment.run(rows, rows, 1e-2, [0.5, 1.0])
    for duration in (0.0, 0.5, 1.0):
        np.testing.assert_allclose(first["probabilities"][duration], second["probabilities"][duration], atol=1e-6)
    # 24 rows in micro-batches of 4, two per step: 3 steps per epoch, so checkpoints at steps 2 and 3.
    assert [record["optimizer_steps"] for record in first["checkpoints"]] == [2, 3]
    assert [record["durations"] for record in first["checkpoints"]] == [[0.5], [1.0]]
    # A zero-duration run after training (the final fit of a frozen choice) restores the
    # initial adapters rather than keeping the last run's.
    untrained = experiment.run(rows, rows[:0], 1e-2, [])
    assert not untrained["checkpoints"]
    assert all(torch.equal(value, experiment.initial[name])
               for name, value in experiment.model.named_parameters() if name in experiment.initial)


def test_control_trains_on_the_adapted_models_batches(tiny, monkeypatch):
    checks = []
    monkeypatch.setattr(neural, "validate_snapshot",
                        lambda _, **options: checks.append(options) or {"revision": "synthetic"})
    # Synthetic passages all have the same length; vary them so that bucketing matters.
    train = tiny["train"].assign(text=[" ".join(text.split()[:3 + index % 12])
                                       for index, text in enumerate(tiny["train"]["text"])])
    common = {"frozen": tiny["frozen"], "config": tiny["config"], "device": "cpu"}
    adapted = lora.Experiment(train, tiny["cached"], adapters=True, model_path=tiny["model"], **common)
    control = lora.Experiment(train, tiny["cached"], adapters=False, model_path=tiny["model"], **common)
    assert len(set(adapted.lengths)) > 1
    # Batches are a function of the lengths and the seed alone, so equal lengths mean equal batches.
    np.testing.assert_array_equal(control.lengths, adapted.lengths)
    assert checks == [{}, {"weights": False}]  # the control checks the pin of its weightless snapshot
    with pytest.raises(ValueError, match="tokenizer"):
        lora.Experiment(tiny["train"], tiny["cached"], adapters=False, **common)


@pytest.mark.parametrize("adapters", [False, True])
def test_fold_chooses_epochs_on_inner_rows_and_scores_only_outer_rows(tiny, tmp_path, adapters):
    experiment = lora.Experiment(tiny["train"], tiny["cached"], adapters=adapters, model_path=tiny["model"],
                                 frozen=tiny["frozen"], config=tiny["config"], device="cpu")
    log = lora.run_fold(tiny["train"], tiny["plan"], experiment, 1, tmp_path / "fold1")
    plan = tiny["plan"].set_index("id")
    inner = pd.read_parquet(tmp_path / "fold1" / "inner.parquet")
    outer = pd.read_parquet(tmp_path / "fold1" / "outer.parquet")
    assert (plan.loc[inner["id"], "outer_fold"] != 1).all()
    names = {lora.setting_name(*setting) for setting in tiny["config"].settings()}
    assert set(inner["setting"]) == names and len(names) == 5  # frozen + 2 rates x 2 durations
    assert sorted(outer["id"]) == sorted(plan.index[plan["outer_fold"] == 1])
    assert log["selected_setting"] in names and (outer["setting"] == log["selected_setting"]).all()
    np.testing.assert_allclose(outer.filter(like="probability::").sum(axis=1), 1.0, atol=1e-6)
    saved = json.loads((tmp_path / "fold1" / "log.json").read_text())
    assert saved["adapters"] is adapters and set(saved["inner_mean_balanced_accuracy"]) == names
    with pytest.raises(ValueError, match="must not be overwritten"):
        lora.run_fold(tiny["train"], tiny["plan"], experiment, 1, tmp_path / "fold1")


def test_probe_reports_fit_by_epoch_and_speed(tiny, tmp_path):
    experiment = lora.Experiment(tiny["train"], tiny["cached"], adapters=True, model_path=tiny["model"],
                                 frozen=tiny["frozen"], config=tiny["config"], device="cpu")
    experiment.run(np.arange(24), np.arange(0), 1e-1, [2.0])  # leaves trained adapters behind
    report = lora.probe_training(tiny["train"], tiny["plan"], experiment, tmp_path, per_class=1, epochs=2)
    assert report["rows"] == len(LABELS)
    assert report["init_vs_frozen_archive"]["max_abs"] < 1e-4
    assert [item["epoch"] for item in report["fit_by_epoch"]] == [0.0, 1.0, 2.0]
    assert report["train_seconds_per_row_epoch"] > 0
