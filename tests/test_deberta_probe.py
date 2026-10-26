"""Probe subsets stay in training folds; instrumentation observes but never alters updates."""

import numpy as np
import pandas as pd
import pytest

from ntl import deberta_probe as probe
from ntl import gemini


def test_parameter_groups_cover_sequence_classifier_names_and_reject_others():
    assert probe.parameter_group("deberta.embeddings.word_embeddings.weight") == "embeddings"
    assert probe.parameter_group("deberta.encoder.layer.0.attention.self.query_proj.weight") == "encoder"
    assert probe.parameter_group("pooler.dense.bias") == "pooler"
    assert probe.parameter_group("classifier.weight") == "classifier"
    with pytest.raises(ValueError, match="Unexpected"):
        probe.parameter_group("lm_head.weight")


def test_subset_is_balanced_deterministic_and_excludes_outer_validation():
    frame = pd.DataFrame({
        "id": [f"row-{index:02d}" for index in range(40)],
        "model": [gemini.GEMINI_LABELS[index % 4] for index in range(40)],
        "fold": [index // 8 for index in range(40)],
    })
    subset = probe.select_subset(frame, per_class=3, outer_fold=1, seed=4)
    assert subset["model"].value_counts().to_dict() == dict.fromkeys(gemini.GEMINI_LABELS, 3)
    assert not (subset["fold"] == 1).any()
    assert subset.equals(probe.select_subset(frame, per_class=3, outer_fold=1, seed=4))
    everything = probe.select_subset(frame, per_class=None, outer_fold=1, seed=4)
    assert len(everything) == 32 and not (everything["fold"] == 1).any()
    with pytest.raises(ValueError, match="Too few"):
        probe.select_subset(frame, per_class=9, outer_fold=1, seed=4)


def test_instrumented_tiny_deberta_records_updates_without_changing_them(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    architecture = transformers.DebertaV2Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=24, max_position_embeddings=32, num_labels=4,
        pooler_hidden_size=16, relative_attention=False,
    )
    transformers.DebertaV2ForSequenceClassification(architecture).save_pretrained(
        tmp_path / "tiny", safe_serialization=False)

    class TinyTokenizer:
        def __call__(self, texts, **kwargs):
            ids = [[1, 2, 3 + index % 4, 2] for index, _ in enumerate(texts)]
            return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

        def pad(self, rows, **kwargs):
            return {key: torch.tensor([row[key] for row in rows], dtype=torch.long) for key in rows[0]}

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    rows = pd.DataFrame({"text": ["alpha", "beta", "gamma", "delta"], "target": [0, 1, 2, 3],
                         "prompt_id": ["train"] * 4, "selected": [True] * 4})
    config = gemini.GeminiConfig(epochs=3, batch_size=2, gradient_accumulation=1,
                                 learning_rate=1e-3, max_length=16, device="cpu")
    plain, plain_log = gemini._fit_predict_fold(rows, rows, tmp_path / "tiny", config, 0)
    with probe.instrument() as recorder:
        observed, observed_log = gemini._fit_predict_fold(rows, rows, tmp_path / "tiny", config, 0)
        displacement = recorder.displacement()
    np.testing.assert_array_equal(plain, observed)
    assert [epoch["mean_training_loss"] for epoch in plain_log["epochs"]] == \
        [epoch["mean_training_loss"] for epoch in observed_log["epochs"]]
    assert recorder.clip_calls == len(recorder.updates) == observed_log["optimizer_updates"] == 6
    # Linear warmup starts at zero, so the first update is recorded but moves nothing.
    assert recorder.updates[0]["learning_rate"] == 0
    assert all(value == 0 for value in recorder.updates[0]["parameter_change_by_group"].values())
    for update in recorder.updates:
        assert update["gradient_norm"] > 0 and np.isfinite(update["gradient_norm"])
        assert update["parameters_without_gradient"] == []
        assert update["loss_scale"] is None
    for update in recorder.updates[1:]:
        assert all(value > 0 for value in update["parameter_change_by_group"].values())
    assert all(value > 0 for value in displacement["parameter_change_by_group"].values())
    # Patches are removed when the context exits.
    assert torch.optim.AdamW is not None and torch.optim.AdamW.__name__ == "AdamW"
