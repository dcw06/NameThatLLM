"""READER fingerprints: the DCT basis, a truncated proxy equal to the full model's early exit,
pinned loading, and READER's probe, on tiny synthetic models."""

import hashlib
import json
import math

import numpy as np
import pytest

from ntl import reader_fingerprint

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
CONFIG = {"vocab_size": 64, "hidden_size": 16, "num_hidden_layers": 3, "num_attention_heads": 2,
          "num_key_value_heads": 1, "head_dim": 8, "intermediate_size": 32, "max_position_embeddings": 64,
          "model_type": "qwen3", "tie_word_embeddings": False}


def test_the_dct_basis_matches_readers_length_normalized_modes():
    lengths = torch.tensor([3, 1, 5])
    basis = reader_fingerprint.dct_basis(lengths, 5)
    assert basis.shape == (3, 2, 5)
    for row, length in enumerate(lengths.tolist()):
        for t in range(5):
            inside = t < length
            assert basis[row, 0, t].item() == pytest.approx(1 / length if inside else 0)
            expected = math.sqrt(2) * math.cos(math.pi * (t + 0.5) / length) / length if inside and length > 1 else 0
            assert basis[row, 1, t].item() == pytest.approx(expected, abs=1e-7)
    hidden = torch.randn(1, 3, 4)
    fingerprint = torch.einsum("bqt,btd->bqd", reader_fingerprint.dct_basis(torch.tensor([3]), 3), hidden)
    assert torch.allclose(fingerprint[0, 0], hidden[0].mean(dim=0))  # DC is the token mean


def _shard(directory, state, monkeypatch):
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(CONFIG))
    save_file({key: value.contiguous() for key, value in state.items()}, str(directory / reader_fingerprint.SHARD))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: type("Tokenizer", (), {"pad_token": "x", "padding_side": "left"})())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
    return digest(directory / reader_fingerprint.SHARD), {"config.json": digest(directory / "config.json")}


def _full_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(**CONFIG)).eval()


def test_the_truncated_proxy_equals_the_full_models_layer_two_state(tmp_path, monkeypatch):
    full = _full_model()
    shard, files = _shard(tmp_path / "ok", full.state_dict(), monkeypatch)
    tokenizer, model = reader_fingerprint.load_proxy(tmp_path / "ok", layer=2, files=files, shard_sha256=shard)
    assert len(model.layers) == 2 and tokenizer.padding_side == "right"
    inputs = {"input_ids": torch.tensor([[5, 9, 12, 3]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    with torch.no_grad():
        expected = full(**inputs, output_hidden_states=True).hidden_states[2]  # block 2's output, before the norm
        captured = reader_fingerprint.last_block_output(model, inputs)
    assert torch.allclose(captured, expected, atol=1e-6)
    with pytest.raises(ValueError, match="published hash"):
        reader_fingerprint.load_proxy(tmp_path / "ok", layer=2, files=files, shard_sha256="0" * 64)
    partial = {key: value for key, value in full.state_dict().items() if not key.startswith("model.layers.1.")}
    shard, files = _shard(tmp_path / "partial", partial, monkeypatch)
    with pytest.raises(ValueError, match="every retained block"):
        reader_fingerprint.load_proxy(tmp_path / "partial", layer=2, files=files, shard_sha256=shard)


def test_extraction_fingerprints_each_passage_over_its_own_tokens(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    full = _full_model()
    shard, files = _shard(tmp_path / "model", full.state_dict(), monkeypatch)
    _, model = reader_fingerprint.load_proxy(tmp_path / "model", layer=2, files=files, shard_sha256=shard)
    vocab = {"[PAD]": 0, "[UNK]": 1, **{word: index + 2 for index, word in enumerate("abcdef")}}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")
    features, lengths = reader_fingerprint.extract(["a b c", "d"], tokenizer, model)
    assert features.shape == (2, 2, 16) and lengths["min"] == 1 and lengths["max"] == 3
    with torch.no_grad():
        hidden = reader_fingerprint.last_block_output(
            model, {"input_ids": torch.tensor([[2, 3, 4]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)})
    np.testing.assert_allclose(features[0, 0], hidden[0].mean(dim=0).numpy(), atol=1e-6)
    assert np.all(features[1, 1] == 0)  # a one-token passage has no AC component


def test_readers_probe_is_deterministic_and_learns_a_separable_signal():
    # READER's probe stops after 40 small Adam steps, so its weights stay near sign-like; give every
    # feature signal (one column per class) for it to find.
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(5), 20)
    features = rng.normal(size=(100, 5)) + 3.0 * np.eye(5)[labels]
    fit, rows = np.arange(0, 100, 2), np.arange(1, 100, 2)
    first = reader_fingerprint.reader_probe(features, labels, 5, fit, rows)
    second = reader_fingerprint.reader_probe(features, labels, 5, fit, rows)
    assert np.array_equal(first, second) and (first == labels[rows]).mean() > 0.8
