"""LUAR-MUD and LUSR readouts: masked mean (+ LUAR's projection), L2-normalized, pinned snapshots only."""

import hashlib
import json

import numpy as np
import pytest

from ntl import style_encoders

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


def _snapshot(directory, key, monkeypatch):
    """A tiny random RoBERTa saved in the layout of the named checkpoint, with its own spec."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

    torch.manual_seed(3)
    vocabulary = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3} | {word: 4 + i for i, word in enumerate(WORDS)}
    backbone = {"model_type": "roberta", "vocab_size": len(vocabulary), "hidden_size": 8, "num_hidden_layers": 2,
                "num_attention_heads": 2, "intermediate_size": 16, "max_position_embeddings": 40,
                "pad_token_id": 1, "bos_token_id": 0, "eos_token_id": 2, "type_vocab_size": 1}
    model = RobertaModel(RobertaConfig(**{k: v for k, v in backbone.items() if k != "model_type"}))
    state = {name: value.contiguous() for name, value in model.state_dict().items()}
    if key == "luar-mud":
        linear = torch.nn.Linear(8, 4)
        state = {f"transformer.{name}": value for name, value in state.items()}
        state |= {"linear.weight": linear.weight.detach().contiguous(), "linear.bias": linear.bias.detach().contiguous()}
        config = {"model_type": "LUAR", "embedding_size": 4, "backbone_config": backbone}
        spec = {"model_type": "LUAR", "layers": 2, "hidden": 8, "dimension": 4}
    else:
        config = backbone
        spec = {"model_type": "roberta", "layers": 2, "hidden": 8, "dimension": 8}
    directory.mkdir()
    save_file(state, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(config))
    core = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    core.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                            unk_token="<unk>").save_pretrained(directory)
    revision = "0" * 40
    files = [{"path": path.name, "size": path.stat().st_size} for path in sorted(directory.iterdir())]
    (directory / "snapshot_manifest.json").write_text(json.dumps(
        {"schema_version": 1, "repo_id": f"test/{key}", "revision": revision, "includes_weights": True,
         "files": files}))
    weights = hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()
    monkeypatch.setitem(style_encoders.ENCODERS, key, {"model_id": f"test/{key}", "model_revision": revision,
                                                       "weights_sha256": weights, **spec})
    return model, state


@pytest.mark.parametrize("key", ["luar-mud", "lusr"])
def test_embedding_is_the_normalized_masked_mean_and_luar_projects_it(tmp_path, monkeypatch, key):
    torch = pytest.importorskip("torch")
    model, state = _snapshot(tmp_path / key, key, monkeypatch)
    texts = ["alpha beta gamma", "delta", "epsilon zeta alpha beta gamma delta", "beta beta"]
    embeddings, metadata = style_encoders.extract(texts, tmp_path / key, key, batch_size=3)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)
    assert metadata["profile"]["truncated_rows"] == 0 and metadata["feature_options"]["remote_code"] is False
    model.eval()
    expected = []
    with torch.inference_mode():
        for text in texts:  # one passage at a time, no padding at all
            ids = torch.tensor([[4 + WORDS.index(word) for word in text.split()]])
            vector = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state.mean(dim=1)
            if key == "luar-mud":
                vector = vector @ state["linear.weight"].T + state["linear.bias"]
            expected.append(torch.nn.functional.normalize(vector, dim=1)[0].numpy())
    np.testing.assert_allclose(embeddings, np.stack(expected), atol=1e-5)


def test_snapshot_must_be_the_pinned_revision_without_remote_code(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _snapshot(tmp_path / "luar", "luar-mud", monkeypatch)
    style_encoders.validate_snapshot(tmp_path / "luar", "luar-mud")
    manifest = json.loads((tmp_path / "luar" / "snapshot_manifest.json").read_text())
    (tmp_path / "luar" / "model.py").write_text("print('remote code')\n")
    manifest["files"].append({"path": "model.py", "size": (tmp_path / "luar" / "model.py").stat().st_size})
    (tmp_path / "luar" / "snapshot_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="never remote code"):
        style_encoders.validate_snapshot(tmp_path / "luar", "luar-mud")
    manifest["revision"] = "1" * 40
    (tmp_path / "luar" / "snapshot_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned encoder revision"):
        style_encoders.validate_snapshot(tmp_path / "luar", "luar-mud")


def test_snapshot_holds_exactly_its_recorded_files_and_the_pinned_weights(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _snapshot(tmp_path / "lusr", "lusr", monkeypatch)
    (tmp_path / "lusr" / "model.py").write_text("print('remote code')\n")  # on disk, not in the manifest
    with pytest.raises(ValueError, match="never remote code"):
        style_encoders.validate_snapshot(tmp_path / "lusr", "lusr")
    (tmp_path / "lusr" / "model.py").unlink()
    (tmp_path / "lusr" / "added_tokens.json").write_text("{}")
    with pytest.raises(ValueError, match="does not record"):
        style_encoders.validate_snapshot(tmp_path / "lusr", "lusr")
    (tmp_path / "lusr" / "added_tokens.json").unlink()
    weights = tmp_path / "lusr" / "model.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 1  # same size, different weights
    weights.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="pinned checkpoint"):
        style_encoders.validate_snapshot(tmp_path / "lusr", "lusr")
