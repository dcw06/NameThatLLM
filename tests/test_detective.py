"""DeTeCtive: pinned, weights-only loading into stock RoBERTa; the authors' pooling; the vote."""

import hashlib

import numpy as np
import pytest

from ntl import detective, ensemble, gpt_specialist, lora, neural
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.smoke import synthetic_frames

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _checkpoint(directory, state):
    from transformers import RobertaConfig

    base = directory / "base"
    base.mkdir(parents=True)
    RobertaConfig(vocab_size=50, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                  max_position_embeddings=40).to_json_file(base / "config.json")
    torch.save(state, directory / detective.CHECKPOINT)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
    return digest(directory / detective.CHECKPOINT), {"config.json": digest(base / "config.json")}


def _state():
    from transformers import RobertaConfig, RobertaModel

    torch.manual_seed(0)
    model = RobertaModel(RobertaConfig(vocab_size=50, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                                       intermediate_size=16, max_position_embeddings=40))
    return {f"model.{key}": value for key, value in model.state_dict().items()}


def test_the_checkpoint_loads_into_stock_roberta_only_when_pinned_and_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: "tokenizer")
    state = _state()
    checkpoint, base = _checkpoint(tmp_path / "ok", state)
    tokenizer, model = detective.load_encoder(tmp_path / "ok", checkpoint_sha256=checkpoint, base_files=base)
    assert tokenizer == "tokenizer" and not model.training
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[f"model.{key}"])
    with pytest.raises(ValueError, match="published hash"):
        detective.load_encoder(tmp_path / "ok", checkpoint_sha256="0" * 64, base_files=base)
    with pytest.raises(ValueError, match="pinned hash"):
        detective.load_encoder(tmp_path / "ok", checkpoint_sha256=checkpoint, base_files={"config.json": "0" * 64})
    extra, base = _checkpoint(tmp_path / "extra", {**state, "classifier.weight": torch.zeros(2)})
    with pytest.raises(ValueError, match="only the encoder"):
        detective.load_encoder(tmp_path / "extra", checkpoint_sha256=extra, base_files=base)
    partial, base = _checkpoint(tmp_path / "partial", {key: value for key, value in list(state.items())[:-1]})
    with pytest.raises(RuntimeError, match="Missing key"):
        detective.load_encoder(tmp_path / "partial", checkpoint_sha256=partial, base_files=base)


def test_pooling_sums_every_padded_position_over_the_real_token_count():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    expected = torch.stack([hidden[0].sum(0) / 2, hidden[1].sum(0) / 4])  # padding included, as the authors had it
    expected = expected / expected.norm(dim=-1, keepdim=True)
    assert torch.allclose(detective.pool(hidden, mask), expected)
    masked = hidden[0, :2].sum(0) / 2
    assert not torch.allclose(detective.pool(hidden, mask)[0], masked / masked.norm())


def test_the_vote_uses_majority_then_summed_similarity_then_label():
    embeddings = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [0.7, 0.7], [1, 0.05]], dtype=float)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.array([0, 0, 1, 1, 2, 0])
    fit = np.arange(5)
    assert detective.knn(embeddings, labels, fit, np.array([5]), k=3).tolist() == [0]  # two of three neighbours
    tie = detective.knn(embeddings, labels, np.array([0, 2]), np.array([4]), k=2)
    assert tie.tolist() == [0]  # one vote each, equal similarity: the smaller label


def test_stage_one_scores_each_predictor_on_the_true_gpt_passages(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    plan = lora.split_plan(train, save_fold_manifest(manifest, metadata, tmp_path / "folds"), lora.AdapterConfig())
    truth = ensemble._targets(train)
    rng = np.random.default_rng(1)
    blocks = {name: rng.normal(size=(len(train), width)) for name, width in
              (("qwen", 8), ("styledistance", 4), (detective.BLOCK, 8))}
    blocks[detective.BLOCK] += 3 * np.eye(8)[truth % 8]  # each GPT class gets its own direction
    blocks[detective.BLOCK] /= np.linalg.norm(blocks[detective.BLOCK], axis=1, keepdims=True)
    folds = plan.set_index("id").loc[train["id"], "outer_fold"].to_numpy()
    locked = rng.dirichlet(np.ones(len(LABELS)), size=len(train))
    result = detective.five_way(train, blocks, folds, locked, neural.FrozenConfig())
    assert set(result) == {"incumbent_forced", "specialist_detective", "specialist_qwen+styledistance",
                           f"specialist_{detective.FEATURE_SET}", "detective_knn5"}
    gpt = np.isin(truth, gpt_specialist.GPT_INDEX)
    from sklearn.metrics import balanced_accuracy_score

    forced = balanced_accuracy_score(np.searchsorted(gpt_specialist.GPT_INDEX, truth[gpt]),
                                     locked[gpt][:, gpt_specialist.GPT_INDEX].argmax(axis=1))
    assert result["incumbent_forced"]["balanced_accuracy"] == pytest.approx(forced)
    assert result["detective_knn5"]["balanced_accuracy"] > 0.5  # the planted signal is found
    assert set(result["detective_knn5"]["recall"]) == set(gpt_specialist.GPT)


def test_the_gpt_specialist_accepts_a_single_custom_feature_set():
    names = [gpt_specialist.option_name(option) for option in gpt_specialist.options((detective.FEATURE_SET,))]
    assert names == ["incumbent"] + [f"{detective.FEATURE_SET}@w={weight:g}" for weight in gpt_specialist.WEIGHTS]
