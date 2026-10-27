"""Neural artifact, fold, and padding contracts; no network or pretrained weights."""

import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from ntl import neural
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest
from ntl.smoke import synthetic_frames


def test_configuration_rejects_nonretained_layers():
    with pytest.raises(ValueError, match="retained decoder"):
        replace(neural.FrozenConfig(), retained_layers=20).validate()


def test_snapshot_checks_pinned_revision_and_file_sizes(tmp_path):
    config = {"model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"test-fixture-not-real-weights")
    manifest = {
        "schema_version": 1, "repo_id": neural.MODEL_ID,
        "revision": neural.MODEL_REVISION, "includes_weights": True,
        "files": [{"path": p.name, "size": p.stat().st_size} for p in tmp_path.iterdir()],
    }
    path = tmp_path / "snapshot_manifest.json"
    path.write_text(json.dumps(manifest))
    assert neural.validate_snapshot(tmp_path)["revision"] == neural.MODEL_REVISION
    manifest["revision"] = "0" * 40
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned"):
        neural.validate_snapshot(tmp_path)
    manifest["revision"] = neural.MODEL_REVISION
    path.write_text(json.dumps(manifest))
    (tmp_path / "model.safetensors").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="size changed"):
        neural.validate_snapshot(tmp_path)


def test_weightless_snapshot_is_accepted_only_for_its_tokenizer(tmp_path):
    config = {"model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text("{}")
    manifest = {
        "schema_version": 1, "repo_id": neural.MODEL_ID,
        "revision": neural.MODEL_REVISION, "includes_weights": False,
        "files": [{"path": p.name, "size": p.stat().st_size} for p in tmp_path.iterdir()],
    }
    path = tmp_path / "snapshot_manifest.json"
    path.write_text(json.dumps(manifest))
    assert neural.validate_snapshot(tmp_path, weights=False)["revision"] == neural.MODEL_REVISION
    with pytest.raises(ValueError, match="weights"):
        neural.validate_snapshot(tmp_path)
    manifest["revision"] = "0" * 40
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned"):
        neural.validate_snapshot(tmp_path, weights=False)


def test_padding_does_not_change_masked_feature_means():
    torch = pytest.importorskip("torch")
    state = torch.tensor([[[1., 3.], [3., 5.], [999., 999.]]])
    mask = torch.tensor([[1, 1, 0]])
    result = neural.masked_layer_mean((state, state, 2 * state), mask, (1, 2))
    assert torch.allclose(result, torch.tensor([[2., 4., 4., 8.]]))
    with pytest.raises(ValueError, match="empty"):
        neural.masked_layer_mean((state, state), torch.zeros_like(mask), (1,))


def test_local_qwen_prefix_loading_and_batch_padding_are_consistent(tmp_path, monkeypatch):
    """Exercise the real Transformers loader with tiny, locally generated random weights."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(17)
    architecture = Qwen3Config(
        vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, attention_dropout=0.0, pad_token_id=0, eos_token_id=1,
    )
    original = Qwen3ForCausalLM(architecture).eval()
    original.save_pretrained(tmp_path)
    tokenizer_core = Tokenizer(WordLevel(
        {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "a": 3, "b": 4, "c": 5},
        unk_token="[UNK]",
    ))
    tokenizer_core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )
    tokenizer.save_pretrained(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(neural, "validate_snapshot", lambda _: {
        "repo_id": "synthetic-random-qwen", "revision": "synthetic-only",
    })
    config = neural.FrozenConfig(
        retained_layers=3, pool_layers=(1, 3), quantization="none",
        batch_size=2, max_length=16, device="cpu",
    )
    texts = ["a b", "a b c a b", " ".join(["a"] * 20)]
    actual, profile = neural.extract_features(texts, tmp_path, config)
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=16, return_tensors="pt")
    with torch.inference_mode():
        hidden = original.model(**encoded, output_hidden_states=True).hidden_states
        # Transformers returns the final retained block after the backbone's final
        # RMSNorm. A middle block in the full model is returned before that norm.
        prefix_hidden = list(hidden)
        prefix_hidden[3] = original.model.norm(hidden[3])
        expected = neural.masked_layer_mean(prefix_hidden, encoded["attention_mask"], (1, 3)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    single, _ = neural.extract_features([texts[0]], tmp_path, replace(config, batch_size=1))
    np.testing.assert_allclose(single[0], actual[0], atol=1e-5, rtol=1e-5)
    assert profile["truncated_rows"] == 1
    assert actual.shape == (3, 32)


def test_frozen_head_uses_identical_prompt_folds_and_canonical_labels(tmp_path, monkeypatch):
    train, test = synthetic_frames()
    lookup = {text: index for index, text in enumerate(train["text"])}
    rng = np.random.default_rng(90210)
    features = rng.normal(size=(len(train), 12)).astype(np.float32)

    def fake_features(texts, model_path, config):
        indices = [lookup.get(text, 0) for text in texts]
        return features[indices], {"rows": len(texts), "synthetic": True}

    monkeypatch.setattr(neural, "extract_features", fake_features)
    config = neural.FrozenConfig(quantization="none")
    output = tmp_path / "cv"
    metrics = neural.run_cv(train, "unused", output, config)
    oof = pd.read_parquet(output / "oof.parquet")
    expected, _ = build_fold_manifest(train)
    pd.testing.assert_series_equal(
        oof.set_index("id")["fold"].sort_index(),
        expected.set_index("id")["fold"].sort_index(),
        check_names=False,
    )
    assert oof["id"].is_unique and len(oof) == len(train)
    assert oof.groupby("prompt_id")["fold"].nunique().max() == 1
    assert 0 <= metrics["balanced_accuracy"] <= 1
    probabilities = oof.filter(like="probability__").to_numpy()
    assert probabilities.shape == (len(train), len(LABELS))
    assert np.allclose(probabilities.sum(axis=1), 1)
    model_dir = tmp_path / "head"
    neural.train_head(train, "unused", model_dir, config)
    submission_path = tmp_path / "submission.parquet"
    neural.predict_head(model_dir, test, submission_path)
    assert pd.read_parquet(submission_path)["id"].tolist() == test["id"].tolist()
    assert (tmp_path / "submission_profile.json").is_file()
    assert not (model_dir / "inference_profile.json").exists()
    artifact = joblib.load(model_dir / "head.joblib")
    artifact["metadata"]["labels"] = list(reversed(LABELS))
    joblib.dump(artifact, model_dir / "head.joblib")
    with pytest.raises(ValueError, match="label order"):
        neural.predict_head(model_dir, test, submission_path)
