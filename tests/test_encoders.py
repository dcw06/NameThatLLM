"""Frozen encoder pooling, immutable inputs, and actual tiny-model extraction."""

import json

import numpy as np
import pytest

from ntl import encoders


def test_masked_mean_includes_special_tokens_and_ignores_padding():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[2., 0.], [4., 2.], [6., 4.], [999., 999.]]])
    mask = torch.tensor([[1, 1, 1, 0]])
    assert torch.allclose(encoders.masked_mean_pool(hidden, mask), torch.tensor([[4., 2.]]))
    with pytest.raises(ValueError, match="empty"):
        encoders.masked_mean_pool(hidden, torch.zeros_like(mask))


def test_last_token_pool_supports_both_padding_sides():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.], [2.], [99.], [99.]], [[99.], [99.], [1.], [2.]]])
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    assert torch.equal(encoders.last_non_padding_pool(hidden, mask), torch.tensor([[2.], [2.]]))
    with pytest.raises(ValueError, match="empty"):
        encoders.last_non_padding_pool(hidden[:1], torch.zeros_like(mask[:1]))


def test_snapshot_rejects_other_revision(tmp_path):
    manifest = {"schema_version": 1, "repo_id": "StyleDistance/styledistance",
                "revision": "0" * 40, "includes_weights": True}
    (tmp_path / "snapshot_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="immutable"):
        encoders.validate_snapshot(tmp_path, "styledistance")


@pytest.mark.parametrize("model_key", ["styledistance", "qwen3-embedding-0.6b"])
def test_actual_tiny_frozen_model_extraction_is_padding_invariant(tmp_path, monkeypatch, model_key):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        AutoModel,
        PreTrainedTokenizerFast,
        Qwen3Config,
        Qwen3Model,
        RobertaConfig,
        RobertaModel,
    )

    torch.manual_seed(9)
    if model_key == "styledistance":
        model = RobertaModel(RobertaConfig(
            vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, max_position_embeddings=64, pad_token_id=0,
            hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        )).eval()
    else:
        model = Qwen3Model(Qwen3Config(
            vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=64, attention_dropout=0, pad_token_id=0, eos_token_id=1,
        )).eval()
    model.save_pretrained(tmp_path)
    core = Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "a": 3, "b": 4, "c": 5}, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
    )
    tokenizer.save_pretrained(tmp_path)
    spec = dict(encoders.ENCODERS[model_key], dimension=16)
    monkeypatch.setitem(encoders.ENCODERS, model_key, spec)
    monkeypatch.setattr(encoders, "validate_snapshot", lambda *args: {})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    original_loader = AutoModel.from_pretrained
    loaded = []

    def checked_loader(*args, **kwargs):
        value = original_loader(*args, **kwargs)
        loaded.append((value, {k: v.detach().clone() for k, v in value.state_dict().items()}))
        return value

    monkeypatch.setattr(AutoModel, "from_pretrained", checked_loader)
    texts = ["a b", "a b c a b"]
    features, metadata = encoders.extract_embeddings(
        texts, tmp_path, model_key, batch_size=2, max_length=32, device="cpu",
    )
    single, _ = encoders.extract_embeddings(
        [texts[0]], tmp_path, model_key, batch_size=1, max_length=32, device="cpu",
    )
    assert features.shape == (2, 16)
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(single[0], features[0], atol=1e-5, rtol=1e-5)
    assert metadata["encoder_frozen"] is True
    assert metadata["feature_options"]["instruction_prefix"] == ""
    for actual, initial in loaded:
        assert not actual.training
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in actual.parameters())
        assert all(torch.equal(actual.state_dict()[key], value) for key, value in initial.items())
