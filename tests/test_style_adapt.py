"""Adapted StyleDistance: only layers 9-12 train, runs are prefix-stable, caches stay in their partition,
and a = 0 reproduces the locked incumbent."""

import json

import numpy as np
import pandas as pd
import pytest

from ntl import claude_glm, encoders, ensemble, lora, style_adapt
from ntl.constants import LABELS
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.smoke import synthetic_frames

torch = pytest.importorskip("torch")
REVISION = "0" * 40


def _snapshot(directory, monkeypatch, texts, dropout=0.1):
    """A tiny 12-layer RoBERTa saved in StyleDistance's Sentence Transformers layout."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

    torch.manual_seed(1)
    words = sorted({word for text in texts for word in Whitespace().pre_tokenize_str(text) for word in [word[0]]})
    vocabulary = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3} | {word: 4 + i for i, word in enumerate(words)}
    config = {"vocab_size": len(vocabulary), "hidden_size": 8, "num_hidden_layers": 12, "num_attention_heads": 2,
              "intermediate_size": 16, "max_position_embeddings": 128, "pad_token_id": 1, "bos_token_id": 0,
              "eos_token_id": 2, "type_vocab_size": 1, "hidden_dropout_prob": dropout,
              "attention_probs_dropout_prob": dropout}
    model = RobertaModel(RobertaConfig(**config))
    directory.mkdir()
    save_file({name: value.contiguous() for name, value in model.state_dict().items()},
              str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps({"model_type": "roberta", **config}))
    core = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    core.pre_tokenizer = Whitespace()
    core.post_processor = TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                            unk_token="<unk>").save_pretrained(directory)
    (directory / "modules.json").write_text(json.dumps([
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}]))
    (directory / "1_Pooling").mkdir()
    (directory / "1_Pooling" / "config.json").write_text(json.dumps({"pooling_mode_mean_tokens": True}))
    files = [{"path": str(path.relative_to(directory)), "size": path.stat().st_size}
             for path in sorted(directory.rglob("*")) if path.is_file()]
    (directory / "snapshot_manifest.json").write_text(json.dumps(
        {"schema_version": 1, "repo_id": "StyleDistance/styledistance", "revision": REVISION,
         "includes_weights": True, "files": files}))
    monkeypatch.setitem(encoders.ENCODERS, "styledistance",
                        {**encoders.ENCODERS["styledistance"], "model_revision": REVISION, "dimension": 8})
    return directory


CONFIG = style_adapt.AdaptConfig(model_revision=REVISION, max_length=64)


def _texts(per_class=3):
    train, _ = synthetic_frames(n_prompts=per_class)
    seven = train[train["model"].isin(claude_glm.SPECIALIST)]
    return seven["text"].tolist(), claude_glm._POSITION[ensemble._targets(seven)]


def test_readout_matches_the_frozen_extraction(tmp_path, monkeypatch):
    texts, _ = _texts()
    path = _snapshot(tmp_path / "model", monkeypatch, texts)
    frozen, _ = encoders.extract_embeddings(texts, path, "styledistance", device="cpu", max_length=64)
    tokenizer, model = style_adapt.load_encoder(path, CONFIG)
    np.testing.assert_allclose(style_adapt.embed(texts, tokenizer, model, CONFIG), frozen, atol=1e-6)


def test_only_layers_9_to_12_train_and_three_epochs_are_the_prefix_of_eight(tmp_path, monkeypatch):
    texts, targets = _texts()  # 21 passages: one update of 16, one of 5, per epoch
    path = _snapshot(tmp_path / "model", monkeypatch, texts)
    seen = {}

    def keep(run):
        def checkpoint(epoch, tokenizer, model, head, state):
            seen[(run, epoch)] = style_adapt.embed(texts, tokenizer, model, CONFIG)
        return checkpoint

    long = style_adapt.train(texts, targets, path, CONFIG, epochs=8, save_epochs=(3, 8), on_checkpoint=keep(8),
                             log=lambda line: None)
    short = style_adapt.train(texts, targets, path, CONFIG, epochs=3, save_epochs=(3,), on_checkpoint=keep(3),
                              log=lambda line: None)
    assert long["frozen_unchanged"] and long["trainable_changed"]
    assert (long["updates"], short["updates"], long["updates_per_epoch"]) == (16, 6, 2)
    assert [item["weights_sha256"] for item in short["curves"]] == \
        [item["weights_sha256"] for item in long["curves"][:3]]
    np.testing.assert_array_equal(seen[(3, 3)], seen[(8, 3)])
    assert not np.array_equal(seen[(8, 3)], seen[(8, 8)])
    assert all(1.5 < item["train_loss"] < 2.5 for item in long["curves"])  # mean cross-entropy near ln 7
    with pytest.raises(ValueError, match="configured epochs"):
        style_adapt.train(texts, targets, path, CONFIG, epochs=2, save_epochs=(3,), log=lambda line: None)


def test_an_update_is_the_mean_loss_over_its_actual_examples(tmp_path, monkeypatch):
    texts, targets = _texts()
    path = _snapshot(tmp_path / "model", monkeypatch, texts, dropout=0.0)
    tokenizer, model = style_adapt.load_encoder(path, CONFIG)
    layers = style_adapt._trainable(model, CONFIG)
    head = torch.nn.Linear(8, 7)
    rows = [0, 3, 5, 8, 13, 20]  # an incomplete update: microbatches of 4 and 2
    labels = torch.as_tensor(targets, dtype=torch.long)
    loss, _ = style_adapt.accumulate(texts, labels, rows, tokenizer, model, head, CONFIG)
    accumulated = [parameter.grad.clone() for parameter in head.parameters()]
    for parameter in [*head.parameters(), *(p for layer in layers for p in layer.parameters())]:
        parameter.grad = None
    batch = tokenizer([texts[row] for row in rows], padding=True, return_tensors="pt")
    mean = torch.nn.functional.cross_entropy(head(style_adapt._readout(model, batch)), labels[rows])
    mean.backward()
    for got, expected in zip(accumulated, head.parameters()):
        torch.testing.assert_close(got, expected.grad, atol=1e-6, rtol=1e-5)
    assert loss / len(rows) == pytest.approx(float(mean.detach()), rel=1e-5)


def test_frozen_modules_stay_in_evaluation_mode_and_decay_skips_biases_and_layernorm(tmp_path, monkeypatch):
    texts, _ = _texts()
    path = _snapshot(tmp_path / "model", monkeypatch, texts)
    _, model = style_adapt.load_encoder(path, CONFIG)
    layers = style_adapt._trainable(model, CONFIG)
    head = torch.nn.Linear(8, 7)
    style_adapt._set_modes(model, layers, head)
    assert not model.embeddings.training and head.training
    assert [layer.training for layer in model.encoder.layer] == [False] * 8 + [True] * 4
    assert [any(p.requires_grad for p in layer.parameters()) for layer in model.encoder.layer] == [False] * 8 + [True] * 4
    assert not any(p.requires_grad for p in model.embeddings.parameters())
    groups = style_adapt._parameter_groups(layers, head, CONFIG)
    names = {id(p): name for layer in layers for name, p in layer.named_parameters()}
    decayed = [names[id(p)] for p in groups[0]["params"]]
    assert decayed and not any(name.endswith("bias") or "LayerNorm" in name for name in decayed)
    assert all(name.endswith("bias") or "LayerNorm" in name for name in (names[id(p)] for p in groups[1]["params"]))
    assert [(group["lr"], group["weight_decay"]) for group in groups] == [(1e-5, 0.01), (1e-5, 0.0), (1e-3, 0.01),
                                                                          (1e-3, 0.0)]
    assert [style_adapt._warmup(CONFIG)(step) for step in range(12)] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0])


def test_options_mixing_and_the_independent_formula():
    assert [style_adapt.option_name(option) for option in style_adapt.options()] == [
        "incumbent", "epoch3@a=0.25", "epoch8@a=0.25", "epoch3@a=0.5", "epoch8@a=0.5"]
    rng = np.random.default_rng(2)
    current = rng.dirichlet(np.ones(len(LABELS)), size=5)
    frozen, adapted = rng.dirichlet(np.ones(7), size=5), rng.dirichlet(np.ones(7), size=5)
    assert style_adapt.mixed(frozen, adapted, 0.0) is frozen
    for weight in (0.0, 0.25, 0.5):
        q = (1 - weight) * frozen + weight * adapted
        expected = claude_glm.redistribute(current, q, 0.5)
        np.testing.assert_allclose(style_adapt._independent(current, frozen, adapted, weight, 0.5), expected,
                                   atol=1e-15)
        others = np.setdiff1d(np.arange(len(LABELS)), claude_glm.INDEX)
        assert np.array_equal(expected[:, others], current[:, others])


def _probabilities(rng, truth, signal):
    logits = rng.normal(size=(len(truth), len(LABELS)))
    logits[np.arange(len(truth)), truth] += signal
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    plan = lora.split_plan(train, save_fold_manifest(manifest, metadata, tmp_path / "folds"), lora.AdapterConfig())
    path = _snapshot(tmp_path / "model", monkeypatch, train["text"].tolist())
    rng = np.random.default_rng(4)
    truth = ensemble._targets(train)
    blocks = {"qwen": rng.normal(size=(len(train), 8)), "styledistance": rng.normal(size=(len(train), 8))}
    blocks["qwen"][:, 0] += 0.3 * truth
    blocks["styledistance"][:, 0] += 0.3 * np.isin(truth, claude_glm.INDEX) * truth
    folds = plan.set_index("id").loc[train["id"], "outer_fold"].to_numpy()
    current = pd.DataFrame(_probabilities(rng, truth, 1.0), columns=claude_glm.BLEND)
    current.insert(0, "fold", folds)
    current.insert(0, "true_model", train["model"].to_numpy())
    current.insert(0, "prompt_id", train["prompt_id"].to_numpy())
    current.insert(0, "id", train["id"].to_numpy())
    frames = []
    for fold in range(5):
        column = plan[f"inner_{fold}"].to_numpy()
        for inner_fold in (0, 1):
            rows = np.flatnonzero(column == inner_fold)
            frame = pd.DataFrame(_probabilities(rng, truth[rows], 1.0), columns=claude_glm.BLEND)
            frame.insert(0, "groups_sha256", claude_glm.groups_sha256(train))
            frame.insert(0, "weight", 0.25)
            frame.insert(0, "inner_fold", inner_fold)
            frame.insert(0, "outer_fold", fold)
            frame.insert(0, "id", train["id"].to_numpy()[rows])
            frames.append(frame)
    cache = pd.concat(frames, ignore_index=True)
    claude_glm.evaluate(train, blocks, plan, current, cache, tmp_path / "reference", samples=20)
    reference = style_adapt.lock(train, plan, _write(current, tmp_path / "oof.parquet"),
                                 tmp_path / "reference" / "comparison.json", tmp_path / "lock" / "reference.json", [])
    root = tmp_path / "encoders"
    for fold in range(5):
        for inner_fold in (0, 1):
            style_adapt.train_partition(train, plan, fold, inner_fold, path, root, epochs=8, save_epochs=(3, 8),
                                        config=CONFIG)
    return {"train": train, "plan": plan, "blocks": blocks, "current": current, "cache": cache, "root": root,
            "path": path, "reference": reference, "tmp": tmp_path}


def _write(frame, path):
    frame.to_parquet(path, index=False)
    return path


def test_a_zero_reproduces_the_locked_incumbent_and_selected_folds_use_their_own_encoder(pipeline):
    p = pipeline
    selection = style_adapt.select(p["train"], p["blocks"], p["plan"], p["cache"], p["root"], config=CONFIG)
    for item in selection:
        if item["epoch"] is not None:
            style_adapt.train_partition(p["train"], p["plan"], item["outer_fold"], None, p["path"], p["root"],
                                        epochs=item["epoch"], save_epochs=(item["epoch"],), config=CONFIG)
    result = style_adapt.evaluate(p["train"], p["blocks"], p["plan"], p["current"], p["cache"], p["root"],
                                  p["reference"], p["tmp"] / "out", config=CONFIG, samples=20)
    assert result["checks"]["reference_max_abs_difference"] <= 1e-12
    assert result["checks"]["formula_max_abs_difference"] <= 1e-12
    reference = pd.read_parquet(p["tmp"] / "reference" / "predictions.parquet")
    frame = pd.read_parquet(p["tmp"] / "out" / "predictions.parquet")
    assert (frame["prediction::incumbent"] == reference["prediction::current+qwen+styledistance"]).all()
    names = [style_adapt.option_name(option) for option in style_adapt.options()]
    for item in result["selection"]:
        means = item["inner_mean_balanced_accuracy"]
        assert item["selected"] == next(name for name in names if means[name] >= max(means.values()) - 1e-12)
        assert all(means[name] == pytest.approx(np.mean(item["inner_balanced_accuracy"][name])) for name in names)
        rows = (frame["outer_fold"] == item["outer_fold"]).to_numpy()
        if item["epoch"] is None:
            assert (frame.loc[rows, "prediction::candidate"] == frame.loc[rows, "prediction::incumbent"]).all()
    delta = result["systems"]["candidate"]["balanced_accuracy"] - result["systems"]["incumbent"]["balanced_accuracy"]
    assert result["paired"]["balanced_accuracy_delta"] == pytest.approx(delta)
    assert result["promotion"]["passes"] == (delta >= 0.005 and result["paired"]["delta_95_interval"][0] > 0)
    key = json.loads((style_adapt._partition_dir(p["root"], p["train"], "outer0-inner0") / "key.json").read_text())
    assert key["frozen_unchanged"] and key["updates"] == 8 * key["updates_per_epoch"]
    _, fit, valid = style_adapt.partition(p["plan"], 0, 0)
    truth = ensemble._targets(p["train"])
    for value in key["checkpoints"].values():  # the temporary head's training and held-out Claude-GLM rows
        assert value["temporary_head"]["training"]["rows"] == key["training_rows"] == np.isin(
            truth[fit], claude_glm.INDEX).sum()
        assert value["temporary_head"]["held_out"]["rows"] == np.isin(truth[valid], claude_glm.INDEX).sum()
    assert all(value["reload_max_abs_difference"] <= style_adapt.RELOAD_TOLERANCE
               for value in key["checkpoints"].values())


def test_caches_refuse_other_partitions_and_are_nan_elsewhere(pipeline):
    p = pipeline
    adapted = style_adapt.load_partition(p["root"], p["train"], p["plan"], 0, 0, 3, CONFIG)
    name, fit, valid = style_adapt.partition(p["plan"], 0, 0)
    truth = ensemble._targets(p["train"])
    embedded = np.zeros(len(truth), dtype=bool)
    embedded[np.concatenate([fit[np.isin(truth[fit], claude_glm.INDEX)], valid])] = True
    assert np.isfinite(adapted[embedded]).all() and np.isnan(adapted[~embedded]).all()
    source = style_adapt._partition_dir(p["root"], p["train"], "outer0-inner0")
    target = style_adapt._partition_dir(p["root"], p["train"], "outer0-inner1")
    for file in source.iterdir():  # another partition's cache under this partition's name
        (target / file.name).write_bytes(file.read_bytes())
    with pytest.raises(ValueError, match="not trained for this partition"):
        style_adapt.load_partition(p["root"], p["train"], p["plan"], 0, 1, 3, CONFIG)
    other = style_adapt.AdaptConfig(model_revision=REVISION, max_length=64, learning_rate_encoder=2e-5)
    with pytest.raises(ValueError, match="not trained for this partition"):
        style_adapt.load_partition(p["root"], p["train"], p["plan"], 0, 0, 3, other)
    with pytest.raises(ValueError, match="must not be overwritten"):
        style_adapt.train_partition(p["train"], p["plan"], 0, 0, p["path"], p["root"], epochs=3, save_epochs=(3,),
                                    config=CONFIG)


def test_selected_folds_apply_their_own_outer_encoder_through_the_integration(pipeline, monkeypatch):
    """Forced nonzero choices (the synthetic options tie) exercise the outer encoders and the mix."""
    p = pipeline
    # Near-ties between two Claude-GLM classes in the blend, so the adapted specialist flips predictions.
    tied = p["current"].copy()
    values = tied[claude_glm.BLEND].to_numpy()
    truth = ensemble._targets(p["train"])
    for row in np.flatnonzero(np.isin(truth, claude_glm.INDEX)):
        other = claude_glm.INDEX[(list(claude_glm.INDEX).index(truth[row]) + 1) % len(claude_glm.INDEX)]
        values[row, [truth[row], other]] = values[row].max() + 0.05
        values[row] /= values[row].sum()
    tied[claude_glm.BLEND] = values
    claude_glm.evaluate(p["train"], p["blocks"], p["plan"], tied, p["cache"], p["tmp"] / "reference-tied", samples=20)
    reference = style_adapt.lock(p["train"], p["plan"], _write(tied, p["tmp"] / "tied.parquet"),
                                 p["tmp"] / "reference-tied" / "comparison.json", p["tmp"] / "lock-tied" / "ref.json", [])
    forced = [dict(item) for item in style_adapt.select(p["train"], p["blocks"], p["plan"], p["cache"], p["root"],
                                                        config=CONFIG)]
    forced[0].update(selected="epoch3@a=0.5", epoch=3, a=0.5)
    forced[1].update(selected="epoch8@a=0.25", epoch=8, a=0.25)
    monkeypatch.setattr(style_adapt, "select", lambda *args, **kwargs: forced)
    for item in forced[:2]:
        style_adapt.train_partition(p["train"], p["plan"], item["outer_fold"], None, p["path"], p["root"],
                                    epochs=item["epoch"], save_epochs=(item["epoch"],), config=CONFIG)
    result = style_adapt.evaluate(p["train"], p["blocks"], p["plan"], tied, p["cache"], p["root"],
                                  reference, p["tmp"] / "forced", model_path=p["path"], config=CONFIG,
                                  samples=20)
    assert result["checks"]["formula_max_abs_difference"] <= 1e-12
    reload = result["checks"]["reload"]
    assert reload["folds"] == [0, 1] and reload["predictions_identical"]
    assert reload["embeddings"] <= style_adapt.RELOAD_TOLERANCE and reload["probabilities"] <= 1e-6
    saved = np.load(p["tmp"] / "forced" / "probabilities.npz")
    folds = p["plan"]["outer_fold"].to_numpy()
    current = tied.set_index("id").loc[p["train"]["id"], claude_glm.BLEND].to_numpy()
    frame = pd.read_parquet(p["tmp"] / "forced" / "predictions.parquet")
    first, second = (frame[f"prediction::{name}"].to_numpy() for name in ("incumbent", "candidate"))
    assert (first != second).any()  # the reports below compare two different systems
    assert (second == np.asarray(LABELS)[saved["candidate"].argmax(axis=1)]).all()
    assert (first == np.asarray(LABELS)[saved["incumbent"].argmax(axis=1)]).all()
    true = frame["true_model"].to_numpy()
    from sklearn.metrics import balanced_accuracy_score
    for name, values in (("incumbent", first), ("candidate", second)):
        assert result["systems"][name]["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(true, values))
    assert result["paired"]["balanced_accuracy_delta"] == pytest.approx(
        balanced_accuracy_score(true, second) - balanced_accuracy_score(true, first))
    assert result["changes"]["gained"] == ((first != true) & (second == true)).sum()
    assert result["changes"]["lost"] == ((first == true) & (second != true)).sum()
    frozen = style_adapt.neural.FrozenConfig()
    for item in forced:
        rows, fit = np.flatnonzero(folds == item["outer_fold"]), np.flatnonzero(folds != item["outer_fold"])
        q_frozen = claude_glm.specialist_probabilities(p["blocks"], "qwen+styledistance", truth, fit, rows, frozen)
        q = q_frozen
        if item["epoch"] is not None:
            adapted = style_adapt.load_partition(p["root"], p["train"], p["plan"], item["outer_fold"], None,
                                                 item["epoch"], CONFIG)
            q_adapted = claude_glm.specialist_probabilities({"qwen": p["blocks"]["qwen"], "styledistance": adapted},
                                                            "qwen+styledistance", truth, fit, rows, frozen)
            assert not np.allclose(q_adapted, q_frozen)
            q = (1 - item["a"]) * q_frozen + item["a"] * q_adapted
        expected = current[rows].copy()
        mass = expected[:, claude_glm.INDEX].sum(axis=1, keepdims=True)
        expected[:, claude_glm.INDEX] = (1 - item["s"]) * expected[:, claude_glm.INDEX] + item["s"] * mass * q
        np.testing.assert_allclose(saved["candidate"][rows], expected, atol=1e-12, rtol=0)
    curves = result["training"]
    assert {"outer0", "outer1"} <= set(curves) and "outer2" not in curves
    key = json.loads((style_adapt._partition_dir(p["root"], p["train"], "outer0") / "key.json").read_text())
    head = key["checkpoints"]["3"]["temporary_head"]
    assert head["training"]["rows"] == np.isin(truth[folds != 0], claude_glm.INDEX).sum() == 56
    assert head["held_out"]["rows"] == np.isin(truth[folds == 0], claude_glm.INDEX).sum() == 14
    assert curves["outer0"]["updates"] == 3 * curves["outer0"]["updates_per_epoch"]


def test_a_checkpoint_does_not_change_the_rest_of_the_run(tmp_path, monkeypatch):
    """Checkpoints draw random numbers and leave modules in evaluation mode; later epochs must not notice."""
    texts, targets = _texts()
    path = _snapshot(tmp_path / "model", monkeypatch, texts)

    def disruptive(epoch, tokenizer, model, head, state):
        torch.rand(100)
        model.eval()
        head.eval()

    modes = set()
    from transformers import RobertaModel

    original = RobertaModel.forward

    def observed(self, *args, **kwargs):
        if torch.is_grad_enabled():  # a training forward, not an extraction
            modes.add((self.embeddings.training, self.encoder.layer[7].training, self.encoder.layer[8].training,
                       self.encoder.layer[11].training))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RobertaModel, "forward", observed)
    with_checkpoints = style_adapt.train(texts, targets, path, CONFIG, epochs=8, save_epochs=(3, 8),
                                         on_checkpoint=disruptive, log=lambda line: None)
    assert modes == {(False, False, True, True)}  # frozen modules evaluate; layers 9-12 train, every epoch
    without = style_adapt.train(texts, targets, path, CONFIG, epochs=8, save_epochs=(8,), log=lambda line: None)
    assert [item["weights_sha256"] for item in with_checkpoints["curves"]] == \
        [item["weights_sha256"] for item in without["curves"]]


def test_report_and_both_gates_match_independent_counts():
    from sklearn.metrics import balanced_accuracy_score, precision_score

    rng = np.random.default_rng(8)
    truth = np.repeat(np.arange(len(LABELS)), 20)
    folds = np.tile(np.arange(5), len(truth) // 5)
    groups = np.repeat(np.arange(len(truth) // 4), 4)
    incumbent = np.where(rng.random(len(truth)) < 0.5, truth, rng.integers(0, len(LABELS), len(truth)))
    candidate = incumbent.copy()
    fable = LABELS.index("claude-fable-5")
    wrong = np.flatnonzero((truth == fable) & (incumbent != truth))
    candidate[wrong] = fable  # the candidate fixes every Fable error
    glm = LABELS.index("glm-5.3")
    right = np.flatnonzero((truth == glm) & (incumbent == truth))[:2]
    candidate[right] = fable  # and loses two GLM passages to Fable
    result = style_adapt.report(truth, folds, groups, incumbent, candidate, gate="prompt", samples=200, seed=1)
    delta = balanced_accuracy_score(truth, candidate) - balanced_accuracy_score(truth, incumbent)
    assert delta > 0 and result["paired"]["balanced_accuracy_delta"] == pytest.approx(delta)
    assert result["paired"]["delta_95_interval"][1] > 0
    assert result["promotion"]["delta"] == pytest.approx(delta)
    assert result["promotion"]["passes"] == (delta >= 0.005 and result["paired"]["delta_95_interval"][0] > 0)
    assert result["changes"]["gained"] == len(wrong) and result["changes"]["lost"] == len(right)
    assert result["changes"]["by_class"]["glm-5.3"] == {"gained": 0, "lost": len(right)}
    precision = precision_score(truth, candidate, labels=np.arange(len(LABELS)), average=None, zero_division=0)
    assert result["fable"]["candidate"]["precision"] == pytest.approx(precision[fable])
    assert result["fable"]["candidate"]["recall"] == 1.0
    assert result["targeted_confusions"]["candidate"]["glm-5.3->claude-fable-5"] == int(
        ((truth == glm) & (candidate == fable)).sum())
    confusion = result["claude_glm_confusion"]["incumbent"]["glm-5.3"]
    assert confusion["other"] == int(((truth == glm) & ~np.isin(incumbent, claude_glm.INDEX)).sum())
    assert sum(confusion.values()) == 20
    changes = [balanced_accuracy_score(truth[folds == fold], candidate[folds == fold])
               - balanced_accuracy_score(truth[folds == fold], incumbent[folds == fold]) for fold in range(5)]
    assert result["promotion"]["fold_changes"] == pytest.approx(changes)
    topic = style_adapt.report(truth, folds, groups, incumbent, candidate, gate="topic", samples=200, seed=1)
    assert topic["promotion"]["passes"] == (delta > 0 and sum(change >= 0 for change in changes) >= 3)
    worse = style_adapt.report(truth, folds, groups, candidate, incumbent, gate="topic", samples=200, seed=1)
    assert worse["promotion"]["delta"] < 0 and not worse["promotion"]["passes"]
    # A large gain in two folds and small losses in three: positive overall, yet the topic gate fails.
    split = incumbent.copy()
    for fold in (0, 1):
        split[(folds == fold) & (split != truth)] = truth[(folds == fold) & (split != truth)]
    for fold in (2, 3, 4):
        row = np.flatnonzero((folds == fold) & (split == truth))[0]
        split[row] = (truth[row] + 1) % len(LABELS)
    two = style_adapt.report(truth, folds, groups, incumbent, split, gate="topic", samples=200, seed=1)
    assert two["promotion"]["delta"] > 0 and two["promotion"]["nonnegative_folds"] == 2
    assert not two["promotion"]["passes"]
    with pytest.raises(ValueError, match="Unknown gate"):
        style_adapt.report(truth, folds, groups, incumbent, candidate, gate="other", samples=10, seed=1)


def test_the_topic_gate_treats_float_rounding_of_exact_ties_as_ties():
    from sklearn.metrics import balanced_accuracy_score

    # Three classes of ten passages in each of five folds; `correct` counts the right passages
    # per fold and class, the rest being called the next class.
    truth, folds = np.tile(np.repeat([0, 1, 2], 10), 5), np.repeat(np.arange(5), 30)
    groups = np.arange(len(truth)) // 5

    def predictions(correct):
        result = truth.copy()
        for fold, counts in enumerate(correct):
            for label, count in enumerate(counts):
                rows = np.flatnonzero((folds == fold) & (truth == label))
                result[rows[count:]] = (label + 1) % 3
        return result

    def raw(first, second, rows=slice(None)):
        return balanced_accuracy_score(truth[rows], second[rows]) - balanced_accuracy_score(truth[rows], first[rows])

    # An exact overall tie: fold 0 trades 1 + 2 + 0 more right passages for 3 + 0 + 0.
    incumbent = predictions([[3, 4, 2]] + [[2, 2, 2]] * 4)
    tie = predictions([[5, 2, 2]] + [[2, 2, 2]] * 4)
    assert raw(incumbent, tie) > 0  # the float residue of an exact tie
    result = style_adapt.report(truth, folds, groups, incumbent, tie, gate="topic", samples=20, seed=1)
    assert result["promotion"]["delta"] == 0.0 and not result["promotion"]["passes"]
    # A real gain in fold 4, fold 3 unchanged, and the same exact trade in folds 0-2.
    incumbent = predictions([[5, 4, 2]] * 3 + [[2, 4, 2]] * 2)
    gain = predictions([[3, 6, 2]] * 3 + [[2, 4, 2], [3, 4, 2]])
    assert all(raw(incumbent, gain, folds == fold) < 0 for fold in range(3))  # float residues
    result = style_adapt.report(truth, folds, groups, incumbent, gain, gate="topic", samples=20, seed=1)
    assert result["promotion"]["fold_changes"][:4] == [0.0] * 4
    assert result["promotion"]["delta"] == pytest.approx(1 / 150)
    assert result["promotion"]["nonnegative_folds"] == 5 and result["promotion"]["passes"]


@pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()),
                    reason="needs a CUDA or MPS device")
def test_a_partition_trains_and_checkpoints_on_an_accelerator(tmp_path, monkeypatch):
    device = "cuda" if torch.cuda.is_available() else "mps"
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    plan = lora.split_plan(train, save_fold_manifest(manifest, metadata, tmp_path / "folds"), lora.AdapterConfig())
    path = _snapshot(tmp_path / "model", monkeypatch, train["text"].tolist())
    key = style_adapt.train_partition(train, plan, 0, 0, path, tmp_path / "encoders", epochs=3, save_epochs=(3,),
                                      config=CONFIG, device=device)
    head = key["checkpoints"]["3"]["temporary_head"]
    assert head["training"]["rows"] == key["training_rows"] and np.isfinite(head["held_out"]["loss"])
    assert key["checkpoints"]["3"]["reload_max_abs_difference"] <= style_adapt.RELOAD_TOLERANCE
    assert key["environment"]["device"] == device and key["frozen_unchanged"]
    adapted = style_adapt.load_partition(tmp_path / "encoders", train, plan, 0, 0, 3, CONFIG)
    assert np.isfinite(adapted).any(axis=1).sum() == key["training_rows"] + key["validation_rows"]


def test_partitions_follow_the_plan_and_never_overlap(tmp_path):
    train, _ = synthetic_frames(n_prompts=10)
    manifest, metadata = build_fold_manifest(train)
    plan = lora.split_plan(train, save_fold_manifest(manifest, metadata, tmp_path / "folds"), lora.AdapterConfig())
    outer = plan["outer_fold"].to_numpy()
    for fold in range(5):
        name, fit, valid = style_adapt.partition(plan, fold, None)
        assert name == f"outer{fold}"
        assert set(fit) == set(np.flatnonzero(outer != fold)) and set(valid) == set(np.flatnonzero(outer == fold))
        column = plan[f"inner_{fold}"].to_numpy()
        for inner in (0, 1):
            name, fit, valid = style_adapt.partition(plan, fold, inner)
            assert name == f"outer{fold}-inner{inner}" and not set(fit) & set(valid)
            assert set(valid) == set(np.flatnonzero(column == inner))
            assert set(fit) == {row for row in range(len(plan)) if outer[row] != fold and column[row] not in (inner,)}
            assert not set(fit) & set(np.flatnonzero(outer == fold))
