"""Adapted StyleDistance for the Claude-GLM specialist (docs/styledistance-adapt.md).

StyleDistance's transformer layers 9-12 are trained, with a temporary 768 -> 7 linear head,
on the Claude and GLM passages of one training partition; everything below them stays
frozen and in evaluation mode. The head is then discarded. The adapted embeddings replace
the frozen StyleDistance block in the deployed specialist (Qwen + StyleDistance, block
balanced, C = 0.005), and the two specialists are mixed before the existing integration:
q_mix = (1 - a) q_frozen + a q_adapted, p'_S = (1 - s) p_B,S + s M q_mix.

Commands, in order:
- `lock` records the exact reference (probabilities, predictions, folds, recipes).
- `probe` runs the readout check, the training-only fit check and one-epoch profiling.
- `train` trains one partition's encoder and caches its embeddings, keyed to that partition.
- `select` chooses s, then the checkpoint and a, on the inner folds of every outer fold.
- `evaluate` scores the selected options on the outer folds against the locked reference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from . import claude_glm, ensemble, neural
from .constants import LABELS
from .data import fingerprint_frame, read_train
from .encoders import ENCODERS, masked_mean_pool, validate_snapshot
from .lora import AdapterConfig, validate_plan
from .selection import first_best
from .specialist_mix import CONFUSIONS

INCUMBENT = "qwen+styledistance"
EPOCHS = (3, 8)
MIX_WEIGHTS = (0.25, 0.5)
REFERENCE = "incumbent"
# Probabilities that must match the locked reference; predictions must match exactly.
REFERENCE_TOLERANCE = 1e-12
# Reloaded weights must reproduce cached embeddings this closely (FP32, same CPU kernels).
RELOAD_TOLERANCE = 1e-6
# The untrained encoder's readout against the cached frozen features (eager vs default attention).
READOUT_TOLERANCE = 1e-5


@dataclass(frozen=True)
class AdaptConfig:
    model_revision: str = ENCODERS["styledistance"]["model_revision"]
    trainable_layers: tuple[int, ...] = (8, 9, 10, 11)  # layers 9-12, counted from zero
    learning_rate_encoder: float = 1e-5
    learning_rate_head: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    microbatch: int = 4
    batch: int = 16
    clip_norm: float = 1.0
    warmup_updates: int = 10
    checkpoints: tuple[int, ...] = EPOCHS
    seed: int = 2026
    max_length: int = 512
    extract_batch: int = 16
    dtype: str = "float32"
    attention: str = "eager"

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def options() -> list[tuple[int, float] | None]:
    """In tie order: the incumbent, then the smaller a, then the shorter training."""
    return [None] + [(epoch, weight) for weight in MIX_WEIGHTS for epoch in EPOCHS]


def option_name(option: tuple[int, float] | None) -> str:
    return REFERENCE if option is None else f"epoch{option[0]}@a={option[1]:g}"


def mixed(frozen: np.ndarray, adapted: np.ndarray | None, weight: float) -> np.ndarray:
    return frozen if weight == 0 else (1 - weight) * frozen + weight * adapted


def _ids_sha256(ids) -> str:
    return hashlib.sha256(json.dumps([str(value) for value in ids], separators=(",", ":")).encode()).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# Encoder training


def load_encoder(model_path: str | Path, config: AdaptConfig, device: str = "cpu"):
    """The pinned checkpoint in FP32 with eager attention, every parameter frozen, in evaluation mode."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    manifest = validate_snapshot(model_path, "styledistance")
    if manifest["revision"] != config.model_revision:
        raise ValueError("The snapshot is not the configured StyleDistance revision")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False,
                                              padding_side="right", use_fast=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False,
                                      use_safetensors=True, dtype=getattr(torch, config.dtype),
                                      attn_implementation=config.attention).to(device)
    model.requires_grad_(False)
    model.eval()
    return tokenizer, model


def _readout(model, batch):
    """Masked mean over every unmasked token, special tokens included, then L2 normalization."""
    import torch

    vectors = masked_mean_pool(model(**batch).last_hidden_state, batch["attention_mask"])
    norms = vectors.norm(p=2, dim=1, keepdim=True)
    if not torch.isfinite(vectors).all() or torch.any(norms <= 0):
        raise ValueError("Encoder produced non-finite or zero embeddings")
    return vectors / norms


def embed(texts: list[str], tokenizer, model, config: AdaptConfig, device: str = "cpu") -> np.ndarray:
    import torch

    model.eval()
    chunks = []
    with torch.inference_mode():
        for offset in range(0, len(texts), config.extract_batch):
            batch = tokenizer(texts[offset:offset + config.extract_batch], padding=True, truncation=True,
                              max_length=config.max_length, return_tensors="pt").to(device)
            chunks.append(_readout(model, batch).cpu().numpy())
    embeddings = np.concatenate(chunks).astype(np.float32, copy=False)
    if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5):
        raise ValueError("Adapted embeddings must be finite unit vectors")
    return embeddings


def _trainable(model, config: AdaptConfig) -> list:
    layers = [model.encoder.layer[index] for index in config.trainable_layers]
    for layer in layers:
        layer.requires_grad_(True)
    return layers


def _set_modes(model, layers: list, head) -> None:
    """Frozen modules stay in evaluation mode; only the trained layers and the head train."""
    model.eval()
    for layer in layers:
        layer.train()
    head.train()


def _parameter_groups(layers: list, head, config: AdaptConfig) -> list[dict]:
    """Weight decay on weight matrices only: no biases, no LayerNorm parameters."""
    groups = {"encoder_decay": [], "encoder_no_decay": [], "head_decay": [], "head_no_decay": []}
    for layer in layers:
        for name, parameter in layer.named_parameters():
            no_decay = name.endswith("bias") or "LayerNorm" in name
            groups["encoder_no_decay" if no_decay else "encoder_decay"].append(parameter)
    groups["head_decay"].append(head.weight)
    groups["head_no_decay"].append(head.bias)
    return [{"params": groups["encoder_decay"], "lr": config.learning_rate_encoder,
             "weight_decay": config.weight_decay},
            {"params": groups["encoder_no_decay"], "lr": config.learning_rate_encoder, "weight_decay": 0.0},
            {"params": groups["head_decay"], "lr": config.learning_rate_head, "weight_decay": config.weight_decay},
            {"params": groups["head_no_decay"], "lr": config.learning_rate_head, "weight_decay": 0.0}]


def _warmup(config: AdaptConfig):
    # LambdaLR calls this with the number of updates already taken: 1/10, 2/10, ..., then 1.
    return lambda step: min(1.0, (step + 1) / config.warmup_updates)


def trained_state(model, layers: list, head, config: AdaptConfig) -> dict:
    state = {f"encoder.layer.{index}.{name}": tensor
             for index, layer in zip(config.trainable_layers, layers) for name, tensor in layer.state_dict().items()}
    state.update({f"head.{name}": tensor for name, tensor in head.state_dict().items()})
    return state


def _state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def accumulate(texts: list[str], targets, rows: list[int], tokenizer, model, head, config: AdaptConfig,
               device: str = "cpu") -> tuple[float, int]:
    """Gradients of one update: microbatches of `config.microbatch`, each loss summed and divided
    by the update's actual number of examples. Returns the summed loss and the correct count."""
    import torch

    loss_sum, correct = 0.0, 0
    for offset in range(0, len(rows), config.microbatch):
        micro = rows[offset:offset + config.microbatch]
        batch = tokenizer([texts[row] for row in micro], padding=True, truncation=True,
                          max_length=config.max_length, return_tensors="pt").to(device)
        logits = head(_readout(model, batch))
        labels = targets[micro].to(device)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
        (loss / len(rows)).backward()
        loss_sum += float(loss.detach())
        correct += int((logits.detach().argmax(dim=1) == labels).sum())
    return loss_sum, correct


def train(texts: list[str], targets: np.ndarray, model_path: str | Path, config: AdaptConfig, *, epochs: int,
          save_epochs: tuple[int, ...] = (), on_checkpoint=None, device: str = "cpu", log=print) -> dict:
    """One continuous run from the original checkpoint; `on_checkpoint(epoch, tokenizer, model, head, state)`
    is called at the end of each epoch in `save_epochs`. Neither data order nor schedule depends on `epochs`."""
    import torch

    if not set(save_epochs) <= set(config.checkpoints) or max(save_epochs, default=0) > epochs:
        raise ValueError("Checkpoints must be configured epochs within the run")
    if set(np.unique(targets)) != set(range(len(claude_glm.SPECIALIST))):
        raise ValueError("Every specialist class needs training rows")
    torch.use_deterministic_algorithms(True)
    tokenizer, model = load_encoder(model_path, config, device)
    torch.manual_seed(config.seed)  # after loading, so the head and dropout draws start from the seed
    original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    layers = _trainable(model, config)
    head = torch.nn.Linear(model.config.hidden_size, len(claude_glm.SPECIALIST)).to(device)
    torch.nn.init.xavier_uniform_(head.weight)
    torch.nn.init.zeros_(head.bias)
    parameters = [parameter for layer in layers for parameter in layer.parameters()] + list(head.parameters())
    optimizer = torch.optim.AdamW(_parameter_groups(layers, head, config), betas=config.betas, eps=config.epsilon)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup(config))
    order = torch.Generator().manual_seed(config.seed)
    labels = torch.as_tensor(np.asarray(targets), dtype=torch.long)
    updates, curves, started = 0, [], time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        _set_modes(model, layers, head)
        permutation = torch.randperm(len(texts), generator=order).tolist()
        loss_sum, correct, norms = 0.0, 0, []
        for start in range(0, len(texts), config.batch):
            optimizer.zero_grad(set_to_none=True)
            loss, right = accumulate(texts, labels, permutation[start:start + config.batch], tokenizer, model, head,
                                     config, device)
            loss_sum, correct = loss_sum + loss, correct + right
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in parameters):
                raise ValueError(f"Missing or non-finite gradient at update {updates + 1}")
            norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, config.clip_norm)))
            optimizer.step()
            schedule.step()
            updates += 1
        state = trained_state(model, layers, head, config)
        curves.append({"epoch": epoch, "updates": updates, "train_loss": loss_sum / len(texts),
                       "train_accuracy": correct / len(texts), "gradient_norm_mean": float(np.mean(norms)),
                       "gradient_norm_max": float(np.max(norms)), "seconds": time.perf_counter() - epoch_started,
                       "weights_sha256": _state_sha256(state)})
        log(json.dumps(curves[-1]))
        if epoch in save_epochs and on_checkpoint is not None:
            # A checkpoint (including its reload check) must not move later dropout draws.
            generator_state = torch.get_rng_state()
            on_checkpoint(epoch, tokenizer, model, head, state)
            torch.set_rng_state(generator_state)
    trained = {f"encoder.layer.{index}." for index in config.trainable_layers}
    after = model.state_dict()
    frozen_unchanged = all(torch.equal(after[name], tensor) for name, tensor in original.items()
                           if not any(name.startswith(prefix) for prefix in trained))
    trainable_changed = all(not torch.equal(after[name], tensor) for name, tensor in original.items()
                            if any(name.startswith(prefix) for prefix in trained) and tensor.is_floating_point())
    if not frozen_unchanged or not trainable_changed:
        raise ValueError("Only layers 9-12 may change, and every one of their tensors must be updated")
    return {"rows": len(texts), "epochs": epochs, "updates": updates,
            "updates_per_epoch": -(-len(texts) // config.batch), "curves": curves,
            "frozen_unchanged": frozen_unchanged, "trainable_changed": trainable_changed,
            "seconds": time.perf_counter() - started}


def load_trained(model_path: str | Path, weights: str | Path, config: AdaptConfig, device: str = "cpu"):
    """The original checkpoint with saved layers 9-12, and the saved temporary head."""
    import torch
    from safetensors.torch import load_file

    tokenizer, model = load_encoder(model_path, config, device)
    state = load_file(str(weights))
    head = torch.nn.Linear(model.config.hidden_size, len(claude_glm.SPECIALIST)).to(device)
    head.load_state_dict({name.removeprefix("head."): tensor for name, tensor in state.items()
                          if name.startswith("head.")})
    encoder = {name: tensor for name, tensor in state.items() if name.startswith("encoder.layer.")}
    expected = {f"encoder.layer.{index}.{name}" for index in config.trainable_layers
                for name in model.encoder.layer[index].state_dict()}
    if set(encoder) != expected:
        raise ValueError("Saved weights must hold exactly the trained layers")
    result = model.load_state_dict(encoder, strict=False)
    if result.unexpected_keys:
        raise ValueError("Saved weights do not match the encoder")
    model.eval()
    return tokenizer, model, head.eval()


# ---------------------------------------------------------------------------------------------
# Partitions and their caches


def partition(plan: pd.DataFrame, outer_fold: int, inner_fold: int | None) -> tuple[str, np.ndarray, np.ndarray]:
    """(name, training-partition rows, validation rows): an inner split, or an outer fold."""
    folds = plan["outer_fold"].to_numpy()
    if inner_fold is None:
        return f"outer{outer_fold}", np.flatnonzero(folds != outer_fold), np.flatnonzero(folds == outer_fold)
    column = plan[f"inner_{outer_fold}"].to_numpy()
    return (f"outer{outer_fold}-inner{inner_fold}", np.flatnonzero((folds != outer_fold) & (column != inner_fold)),
            np.flatnonzero(column == inner_fold))


def _partition_dir(root: Path, train: pd.DataFrame, name: str) -> Path:
    return Path(root) / claude_glm.groups_sha256(train)[:16] / name


def _environment(device: str) -> dict:
    import torch

    return {"software": {package: importlib.metadata.version(package)
                         for package in ("torch", "transformers", "numpy", "scikit-learn", "tokenizers")},
            "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(), "device": device,
            "torch_threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled()}


def train_partition(train_frame: pd.DataFrame, plan: pd.DataFrame, outer_fold: int, inner_fold: int | None,
                    model_path: str | Path, root: str | Path, *, epochs: int, save_epochs: tuple[int, ...],
                    config: AdaptConfig | None = None, device: str = "cpu") -> dict:
    """Train one partition's encoder on its Claude and GLM rows; cache, per saved checkpoint, the
    embeddings of those training rows and of every validation row, with the weights."""
    from safetensors.torch import save_file

    config = config or AdaptConfig()
    plan = validate_plan(train_frame, plan, AdapterConfig())
    name, fit, valid = partition(plan, outer_fold, inner_fold)
    directory = _partition_dir(root, train_frame, name)
    if (directory / "key.json").exists():
        raise ValueError(f"{directory} already holds a trained partition; it must not be overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    truth = ensemble._targets(train_frame)
    training = fit[np.isin(truth[fit], claude_glm.INDEX)]
    texts = train_frame["text"].tolist()
    ids, prompts = train_frame["id"].to_numpy(), train_frame["prompt_id"].to_numpy()
    rows = np.concatenate([training, valid])
    checkpoints = {}

    def on_checkpoint(epoch, tokenizer, model, head, state):
        import torch

        embeddings = embed([texts[row] for row in rows], tokenizer, model, config, device)
        weights = directory / f"epoch{epoch}-weights.safetensors"
        save_file({key: tensor.detach().cpu().contiguous() for key, tensor in state.items()}, str(weights))
        np.savez(directory / f"epoch{epoch}.npz", ids=ids[rows].astype(str), embeddings=embeddings,
                 training=np.arange(len(rows)) < len(training))  # training rows first, then validation rows
        with torch.inference_mode():
            logits = head.eval()(torch.as_tensor(embeddings)).numpy()
        is_training = np.arange(len(rows)) < len(training)
        diagnostics = {}
        for role, mask in (("training", is_training),
                           ("held_out", ~is_training & np.isin(truth[rows], claude_glm.INDEX))):
            target = claude_glm._POSITION[truth[rows][mask]]
            shifted = logits[mask] - logits[mask].max(axis=1, keepdims=True)
            log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            diagnostics[role] = {"rows": int(mask.sum()),
                                 "loss": float(-log_probabilities[np.arange(mask.sum()), target].mean()),
                                 "accuracy": float((logits[mask].argmax(axis=1) == target).mean())}
        # Reload the saved weights into a fresh encoder and re-embed a few rows.
        reload_tokenizer, reloaded, _ = load_trained(model_path, weights, config, device)
        sample = list(range(min(4, len(training)))) + list(range(len(training), min(len(training) + 4, len(rows))))
        again = embed([texts[rows[index]] for index in sample], reload_tokenizer, reloaded, config, device)
        reload_difference = float(np.abs(again - embeddings[sample]).max())
        if reload_difference > RELOAD_TOLERANCE:
            raise ValueError("Reloaded weights do not reproduce the cached embeddings")
        checkpoints[str(epoch)] = {"weights_sha256": _state_sha256(state),
                                   "embeddings_sha256": _array_sha256(embeddings),
                                   "weights_file_sha256": claude_glm._sha256(weights),
                                   "temporary_head": diagnostics, "reload_max_abs_difference": reload_difference}

    result = train([texts[row] for row in training], claude_glm._POSITION[truth[training]], model_path, config,
                   epochs=epochs, save_epochs=save_epochs, on_checkpoint=on_checkpoint, device=device)
    key = {"schema_version": 1, "kind": "adapted-styledistance-partition", "partition": name,
           "outer_fold": outer_fold, "inner_fold": inner_fold, "groups_sha256": claude_glm.groups_sha256(train_frame),
           "train_fingerprint": fingerprint_frame(train_frame),
           "training_ids_sha256": _ids_sha256(ids[training]),
           "training_prompts_sha256": _ids_sha256(sorted(set(prompts[training]))),
           "validation_ids_sha256": _ids_sha256(ids[valid]),
           "training_rows": len(training), "training_prompts": int(len(set(prompts[training]))),
           "validation_rows": len(valid),
           "training_class_counts": {label: int((truth[training] == index).sum())
                                     for label, index in zip(claude_glm.SPECIALIST, claude_glm.INDEX)},
           "configuration": asdict(config), "configuration_sha256": config.sha256(),
           "checkpoints": checkpoints, **result, "environment": _environment(device)}
    claude_glm._json(directory / "key.json", key)
    return key


def load_partition(root: str | Path, train_frame: pd.DataFrame, plan: pd.DataFrame, outer_fold: int,
                   inner_fold: int | None, epoch: int, config: AdaptConfig) -> np.ndarray:
    """One partition's adapted embeddings, checked against that partition, NaN on every other row."""
    name, fit, valid = partition(plan, outer_fold, inner_fold)
    directory = _partition_dir(root, train_frame, name)
    key = json.loads((directory / "key.json").read_text())
    truth = ensemble._targets(train_frame)
    training = fit[np.isin(truth[fit], claude_glm.INDEX)]
    ids = train_frame["id"].to_numpy()
    expected = {"partition": name, "groups_sha256": claude_glm.groups_sha256(train_frame),
                "train_fingerprint": fingerprint_frame(train_frame), "training_ids_sha256": _ids_sha256(ids[training]),
                "validation_ids_sha256": _ids_sha256(ids[valid]), "configuration_sha256": config.sha256()}
    if any(key.get(field) != value for field, value in expected.items()) or str(epoch) not in key["checkpoints"]:
        raise ValueError(f"{directory} was not trained for this partition, grouping, configuration and epoch")
    data = np.load(directory / f"epoch{epoch}.npz", allow_pickle=False)
    rows = np.concatenate([training, valid])
    if (not np.array_equal(data["ids"], ids[rows].astype(str))
            or _array_sha256(data["embeddings"]) != key["checkpoints"][str(epoch)]["embeddings_sha256"]):
        raise ValueError(f"{directory} embeddings do not match their key")
    adapted = np.full((len(train_frame), data["embeddings"].shape[1]), np.nan, dtype=np.float32)
    adapted[rows] = data["embeddings"]
    return adapted


# ---------------------------------------------------------------------------------------------
# Reference, selection and evaluation


def incumbent_probabilities(current: np.ndarray, specialist: np.ndarray, weights: list[float],
                            folds: np.ndarray) -> np.ndarray:
    result = np.empty_like(current)
    for fold, weight in enumerate(weights):
        rows = folds == fold
        result[rows] = claude_glm.redistribute(current[rows], specialist[rows], weight)
    return result


def lock(train_frame: pd.DataFrame, plan: pd.DataFrame, current_oof: Path, comparison: Path, output: Path,
         recipes: list[Path], groups: Path | None = None) -> dict:
    """Rebuild the reference's OOF probabilities and record hashes of everything that defines it."""
    plan = validate_plan(train_frame, plan, AdapterConfig())
    directory = comparison.parent
    folds = plan["outer_fold"].to_numpy()
    current = claude_glm._aligned_current(pd.read_parquet(current_oof), train_frame, folds)
    recorded = json.loads(comparison.read_text())
    if recorded["groups_sha256"] != claude_glm.groups_sha256(train_frame):
        raise ValueError("The reference comparison was made under a different row grouping")
    weights = [item["weight"] for item in recorded["selection"][INCUMBENT]]
    specialist = np.load(directory / f"specialist-{INCUMBENT}.npy")
    probabilities = incumbent_probabilities(current, specialist, weights, folds)
    frame = pd.read_parquet(directory / "predictions.parquet").set_index("id").loc[train_frame["id"]]
    predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    if (predicted != frame[f"prediction::current+{INCUMBENT}"].to_numpy()).any():
        raise ValueError("The rebuilt reference does not reproduce its recorded predictions")
    truth = ensemble._targets(train_frame)
    score = float(balanced_accuracy_score(truth, probabilities.argmax(axis=1)))
    if score != recorded["systems"][f"current+{INCUMBENT}"]["balanced_accuracy"]:
        raise ValueError("The rebuilt reference does not reproduce its recorded balanced accuracy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output.with_suffix(".npz"), ids=train_frame["id"].to_numpy().astype(str), probabilities=probabilities)
    folds_frame = plan[["id", "outer_fold"] + [f"inner_{fold}" for fold in range(5)]]
    result = {"schema_version": 1, "kind": "adapted-styledistance-reference",
              "balanced_accuracy": score, "s_by_fold": weights,
              "probabilities_sha256": _array_sha256(probabilities.astype(np.float64)),
              "predictions_sha256": _ids_sha256(predicted), "folds_sha256": _ids_sha256(
                  folds_frame.astype(str).to_numpy().ravel()),
              "groups_sha256": claude_glm.groups_sha256(train_frame), "train_fingerprint": fingerprint_frame(train_frame),
              "file_sha256": {str(path): claude_glm._sha256(path) for path in
                              [current_oof, comparison, directory / "predictions.parquet",
                               directory / f"specialist-{INCUMBENT}.npy", *recipes, *([groups] if groups else [])]},
              "probabilities_file": str(output.with_suffix(".npz"))}
    claude_glm._json(output, result)
    return result


def _ba(truth: np.ndarray, probabilities: np.ndarray) -> float:
    return float(balanced_accuracy_score(truth, probabilities.argmax(axis=1)))


def _independent(current: np.ndarray, frozen: np.ndarray, adapted: np.ndarray | None, weight: float,
                 s: float) -> np.ndarray:
    """p'_S = (1 - s) p_B,S + s M q_mix, written out class by class."""
    result = current.copy()
    mass = sum(current[:, index] for index in claude_glm.INDEX)
    for position, index in enumerate(claude_glm.INDEX):
        q = frozen[:, position] if weight == 0 else (1 - weight) * frozen[:, position] + weight * adapted[:, position]
        result[:, index] = (1 - s) * current[:, index] + s * mass * q
    return result


def select(train_frame: pd.DataFrame, blocks: dict[str, np.ndarray], plan: pd.DataFrame, current_cache: pd.DataFrame,
           root: str | Path, *, config: AdaptConfig | None = None, frozen: neural.FrozenConfig | None = None) -> list[dict]:
    """For each outer fold: s as the incumbent chooses it, then one of the five options, on the inner folds."""
    config, frozen = config or AdaptConfig(), frozen or neural.FrozenConfig()
    plan = validate_plan(train_frame, plan, AdapterConfig())
    if set(current_cache["groups_sha256"]) != {claude_glm.groups_sha256(train_frame)}:
        raise ValueError("Cached current-system inner predictions were built under a different row grouping")
    truth, folds = ensemble._targets(train_frame), plan["outer_fold"].to_numpy()
    grid, selection = options(), []
    for fold in range(5):
        column = plan[f"inner_{fold}"].to_numpy()
        inner = []
        for inner_fold, (valid, base) in claude_glm._inner_current(current_cache, train_frame, plan, fold).items():
            fit = np.flatnonzero((folds != fold) & (column != inner_fold))
            frozen_q = claude_glm.specialist_probabilities(blocks, INCUMBENT, truth, fit, valid, frozen)
            adapted_q = {epoch: claude_glm.specialist_probabilities(
                {"qwen": blocks["qwen"], "styledistance": load_partition(root, train_frame, plan, fold, inner_fold,
                                                                         epoch, config)},
                INCUMBENT, truth, fit, valid, frozen) for epoch in EPOCHS}
            inner.append((valid, base, frozen_q, adapted_q))
        # The incumbent's own choice of s, as claude_glm.evaluate makes it.
        s_scores = {s: float(np.mean([_ba(truth[valid], claude_glm.redistribute(base, frozen_q, s))
                                      for valid, base, frozen_q, _ in inner]))
                    for s in claude_glm.WEIGHTS}
        s = first_best(claude_glm.WEIGHTS, s_scores.__getitem__)

        def system(base, frozen_q, adapted_q, option):
            if option is None:
                return claude_glm.redistribute(base, frozen_q, s)
            return claude_glm.redistribute(base, mixed(frozen_q, adapted_q[option[0]], option[1]), s)

        scores = {option_name(option): [_ba(truth[valid], system(base, frozen_q, adapted_q, option))
                                        for valid, base, frozen_q, adapted_q in inner]
                  for option in grid}
        means = {name: float(np.mean(values)) for name, values in scores.items()}
        chosen = first_best(grid, lambda option: means[option_name(option)])
        selection.append({"outer_fold": fold, "s": s, "selected": option_name(chosen),
                          "epoch": None if chosen is None else chosen[0], "a": 0.0 if chosen is None else chosen[1],
                          "inner_mean_balanced_accuracy": means, "inner_balanced_accuracy": scores,
                          "s_inner_mean_balanced_accuracy": {str(key): value for key, value in s_scores.items()}})
    return selection


def _metrics(truth: np.ndarray, predictions: np.ndarray, folds: np.ndarray) -> dict:
    result = ensemble._system_metrics(truth, predictions, folds)
    precision, _, _, _ = precision_recall_fscore_support(truth, predictions, labels=np.arange(len(LABELS)),
                                                         zero_division=0)
    result["precision"] = {label: float(precision[index]) for index, label in enumerate(LABELS)}
    return result


def _confusion(truth: np.ndarray, predictions: np.ndarray) -> dict:
    """True Claude and GLM rows by predicted label (outside the seven as "other")."""
    result = {}
    for true in claude_glm.SPECIALIST:
        predicted = predictions[truth == LABELS.index(true)]
        counts = {label: int((predicted == LABELS.index(label)).sum()) for label in claude_glm.SPECIALIST}
        counts["other"] = int(len(predicted) - sum(counts.values()))
        result[true] = counts
    return result


def report(truth: np.ndarray, folds: np.ndarray, groups: np.ndarray, incumbent: np.ndarray, candidate: np.ndarray,
           *, gate: str, samples: int, seed: int) -> dict:
    """Metrics, the paired bootstrap (candidate minus incumbent, whole groups resampled) and the gate.
    Prompt gate: an unrounded gain of at least 0.005 with the 95% interval above zero.
    Topic gate: a positive gain overall and a change of zero or more in at least three of the five folds."""
    predictions = {"incumbent": incumbent, "candidate": candidate}
    systems = {name: _metrics(truth, values, folds) for name, values in predictions.items()}
    paired = ensemble.prompt_bootstrap(truth, incumbent, candidate, groups, samples, seed)
    delta = systems["candidate"]["balanced_accuracy"] - systems["incumbent"]["balanced_accuracy"]
    fold_changes = [after - before for after, before in zip(systems["candidate"]["fold_balanced_accuracy"],
                                                            systems["incumbent"]["fold_balanced_accuracy"])]
    if gate == "prompt":
        passes = delta >= 0.005 and paired["delta_95_interval"][0] > 0
    elif gate == "topic":
        passes = delta > 0 and sum(change >= 0 for change in fold_changes) >= 3
    else:
        raise ValueError(f"Unknown gate {gate!r}")
    gained, lost = (incumbent != truth) & (candidate == truth), (incumbent == truth) & (candidate != truth)
    return {
        "systems": systems, "paired": paired,
        "promotion": {"gate": gate, "delta": delta, "fold_changes": fold_changes,
                      "nonnegative_folds": int(sum(change >= 0 for change in fold_changes)), "passes": bool(passes)},
        "changes": {"gained": int(gained.sum()), "lost": int(lost.sum()),
                    "by_class": {label: {"gained": int(gained[truth == index].sum()),
                                         "lost": int(lost[truth == index].sum())}
                                 for index, label in enumerate(LABELS)}},
        "targeted_confusions": {name: {f"{a}->{b}": int(((truth == LABELS.index(a))
                                                          & (values == LABELS.index(b))).sum())
                                       for a, b in CONFUSIONS} for name, values in predictions.items()},
        "claude_glm_confusion": {name: _confusion(truth, values) for name, values in predictions.items()},
        "fable": {name: {"precision": systems[name]["precision"]["claude-fable-5"],
                         "recall": systems[name]["recall"]["claude-fable-5"]} for name in systems},
    }


def evaluate(train_frame: pd.DataFrame, blocks: dict[str, np.ndarray], plan: pd.DataFrame, current_oof: pd.DataFrame,
             current_cache: pd.DataFrame, root: str | Path, reference: dict, output_dir: str | Path, *,
             gate: str = "prompt", model_path: str | Path | None = None, config: AdaptConfig | None = None,
             frozen: neural.FrozenConfig | None = None, samples: int = 10000, seed: int = 2026,
             device: str = "cpu", provenance: dict | None = None) -> dict:
    """With `model_path`, every outer encoder's saved weights are reloaded and must reproduce its
    embeddings, the candidate's probabilities and its predictions."""
    config, frozen = config or AdaptConfig(), frozen or neural.FrozenConfig()
    plan = validate_plan(train_frame, plan, AdapterConfig())
    output = Path(output_dir)
    if (output / "comparison.json").exists():
        raise ValueError("Use a new output directory; an existing comparison must not be overwritten")
    if (reference["groups_sha256"] != claude_glm.groups_sha256(train_frame)
            or reference["train_fingerprint"] != fingerprint_frame(train_frame)):
        raise ValueError("The locked reference was made for other rows or another grouping")
    locked = np.load(reference["probabilities_file"], allow_pickle=False)
    if (not np.array_equal(locked["ids"], train_frame["id"].to_numpy().astype(str))
            or _array_sha256(locked["probabilities"]) != reference["probabilities_sha256"]):
        raise ValueError("The locked reference probabilities changed")
    selection = select(train_frame, blocks, plan, current_cache, root, config=config, frozen=frozen)
    output.mkdir(parents=True, exist_ok=True)
    truth, prompts = ensemble._targets(train_frame), train_frame["prompt_id"].to_numpy()
    folds = plan["outer_fold"].to_numpy()
    current = claude_glm._aligned_current(current_oof, train_frame, folds)
    incumbent, candidate = np.empty_like(current), np.empty_like(current)
    texts = train_frame["text"].tolist()
    formula_difference = 0.0
    reload = {"folds": [], "embeddings": 0.0, "probabilities": 0.0, "predictions_identical": True}
    for item in selection:
        fold, s = item["outer_fold"], item["s"]
        rows, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        frozen_q = claude_glm.specialist_probabilities(blocks, INCUMBENT, truth, fit, rows, frozen)
        incumbent[rows] = claude_glm.redistribute(current[rows], frozen_q, s)
        adapted_q = None
        if item["epoch"] is not None:  # the incumbent's folds train no outer encoder
            adapted = load_partition(root, train_frame, plan, fold, None, item["epoch"], config)
            adapted_q = claude_glm.specialist_probabilities({"qwen": blocks["qwen"], "styledistance": adapted},
                                                            INCUMBENT, truth, fit, rows, frozen)
        candidate[rows] = claude_glm.redistribute(current[rows], mixed(frozen_q, adapted_q, item["a"]), s)
        if adapted_q is not None and model_path is not None:
            name, outer_fit, _ = partition(plan, fold, None)
            embedded = np.flatnonzero(np.isfinite(adapted).all(axis=1))
            tokenizer, model, _ = load_trained(model_path, _partition_dir(root, train_frame, name)
                                               / f"epoch{item['epoch']}-weights.safetensors", config, device)
            again = np.full_like(adapted, np.nan)
            again[embedded] = embed([texts[row] for row in embedded], tokenizer, model, config, device)
            del model
            again_q = claude_glm.specialist_probabilities({"qwen": blocks["qwen"], "styledistance": again},
                                                          INCUMBENT, truth, fit, rows, frozen)
            again_candidate = claude_glm.redistribute(current[rows], mixed(frozen_q, again_q, item["a"]), s)
            reload["embeddings"] = max(reload["embeddings"], float(np.abs(again[embedded] - adapted[embedded]).max()))
            reload["probabilities"] = max(reload["probabilities"],
                                          float(np.abs(again_candidate - candidate[rows]).max()))
            reload["predictions_identical"] &= bool(
                (again_candidate.argmax(axis=1) == candidate[rows].argmax(axis=1)).all())
            reload["folds"].append(fold)
        for weight, q in ((0.0, None), (item["a"], adapted_q)):
            if weight == 0 or q is not None:
                checked = claude_glm.redistribute(current[rows], mixed(frozen_q, q, weight), s)
                formula_difference = max(formula_difference, float(np.abs(
                    checked - _independent(current[rows], frozen_q, q, weight, s)).max()))
    reference_difference = float(np.abs(incumbent - locked["probabilities"]).max())
    if (incumbent.argmax(axis=1) != locked["probabilities"].argmax(axis=1)).any() or \
            reference_difference > REFERENCE_TOLERANCE:
        raise ValueError("a = 0 does not reproduce the locked incumbent")
    if formula_difference > REFERENCE_TOLERANCE:
        raise ValueError("The integration disagrees with the independently written formula")
    if (reload["embeddings"] > RELOAD_TOLERANCE or reload["probabilities"] > RELOAD_TOLERANCE
            or not reload["predictions_identical"]):
        raise ValueError("Reloaded outer encoders do not reproduce the candidate")
    predictions = {"incumbent": incumbent.argmax(axis=1), "candidate": candidate.argmax(axis=1)}
    reported = report(truth, folds, prompts, predictions["incumbent"], predictions["candidate"], gate=gate,
                      samples=samples, seed=seed)
    frame = train_frame[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"}).assign(outer_fold=folds)
    for name, values in predictions.items():
        frame[f"prediction::{name}"] = np.asarray(LABELS)[values]
    frame.to_parquet(output / "predictions.parquet", index=False)
    np.savez(output / "probabilities.npz", ids=train_frame["id"].to_numpy().astype(str), incumbent=incumbent,
             candidate=candidate)
    curves = {}
    for item in selection:
        names = [partition(plan, item["outer_fold"], inner)[0] for inner in (0, 1)]
        names += [partition(plan, item["outer_fold"], None)[0]] if item["epoch"] is not None else []
        for name in names:
            key = json.loads((_partition_dir(root, train_frame, name) / "key.json").read_text())
            curves[name] = {field: key[field] for field in ("training_rows", "training_prompts", "validation_rows",
                                                            "updates", "updates_per_epoch", "curves", "seconds")}
            curves[name]["checkpoints"] = {epoch: value["temporary_head"] for epoch, value in key["checkpoints"].items()}
    result = {
        "reference": reference, "options": [option_name(option) for option in options()], "selection": selection,
        **reported,
        "checks": {"reference_max_abs_difference": reference_difference,
                   "reference_predictions_identical": True, "formula_max_abs_difference": formula_difference,
                   "reload": reload if model_path is not None else None},
        "training": curves, "configuration": asdict(config), "configuration_sha256": config.sha256(),
        "groups": int(train_frame["prompt_id"].nunique()), "groups_sha256": claude_glm.groups_sha256(train_frame),
        "train_fingerprint": fingerprint_frame(train_frame), "provenance": provenance or {},
    }
    claude_glm._json(output / "comparison.json", result)
    return result


# ---------------------------------------------------------------------------------------------
# Probe: readout check, training-only fit check, one-epoch profiling


def probe(train_frame: pd.DataFrame, plan: pd.DataFrame, model_path: str | Path, frozen_features: np.ndarray,
          output: Path, *, config: AdaptConfig | None = None, device: str = "cpu", fit_epochs: int = 30,
          per_class: int = 8) -> dict:
    config = config or AdaptConfig()
    plan = validate_plan(train_frame, plan, AdapterConfig())
    truth = ensemble._targets(train_frame)
    texts = train_frame["text"].tolist()
    name, fit, _ = partition(plan, 0, 0)
    training = fit[np.isin(truth[fit], claude_glm.INDEX)]
    # The untrained encoder's readout against the cached frozen features.
    tokenizer, model = load_encoder(model_path, config, device)
    sample = training[:32]
    started = time.perf_counter()
    untrained = embed([texts[row] for row in sample], tokenizer, model, config, device)
    embed_seconds = (time.perf_counter() - started) / len(sample)
    readout_difference = float(np.abs(untrained - frozen_features[sample]).max())
    if readout_difference > READOUT_TOLERANCE:
        raise ValueError("The adapted readout does not reproduce the cached frozen StyleDistance features")
    del model
    # Training-only fit check: a small balanced subset, the same loop, more epochs.
    subset = np.concatenate([training[truth[training] == index][:per_class] for index in claude_glm.INDEX])
    fit_check = train([texts[row] for row in subset], claude_glm._POSITION[truth[subset]], model_path, config,
                      epochs=fit_epochs, device=device)
    # One-epoch profiling on a real inner partition.
    profile = train([texts[row] for row in training], claude_glm._POSITION[truth[training]], model_path, config,
                    epochs=1, device=device)
    result = {"readout_max_abs_difference": readout_difference, "readout_rows": len(sample),
              "embedding_seconds_per_passage": embed_seconds,
              "fit_check": {"rows": len(subset), "epochs": fit_epochs, "curves": fit_check["curves"]},
              "profile": {"partition": name, "rows": len(training), "updates": profile["updates"],
                          "epoch_seconds": profile["curves"][0]["seconds"],
                          "epoch1_weights_sha256": profile["curves"][0]["weights_sha256"]},
              "configuration_sha256": config.sha256(), "environment": _environment(device)}
    claude_glm._json(output, result)
    return result


def _plan(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"id": str, "prompt_id": str}, keep_default_na=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("lock", "probe", "train", "select", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--train", type=Path, required=True)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--groups", type=Path, help="CSV of id, prompt_id regrouping rows (e.g. by topic)")
        if name in ("probe", "train", "evaluate"):
            command.add_argument("--model-path", type=Path, required=True)
            command.add_argument("--device", default="cpu")
            command.add_argument("--threads", type=int, required=True)
        if name in ("probe", "select", "evaluate"):
            command.add_argument("--qwen-features", type=Path, required=True)
            command.add_argument("--styledistance-features", type=Path, required=True)
        if name in ("train", "select", "evaluate"):
            command.add_argument("--root", type=Path, required=True, help="directory of partition caches")
        if name in ("select", "evaluate"):
            command.add_argument("--current-inner", type=Path, nargs="+", required=True)
        if name == "lock":
            command.add_argument("--current-oof", type=Path, required=True)
            command.add_argument("--comparison", type=Path, required=True)
            command.add_argument("--recipe", type=Path, nargs="*", default=[])
        if name == "evaluate":
            command.add_argument("--current-oof", type=Path, required=True)
            command.add_argument("--reference", type=Path, required=True)
        if name == "train":
            command.add_argument("--outer-fold", type=int, required=True)
            command.add_argument("--inner-fold", type=int)
            command.add_argument("--epochs", type=int, required=True)
            command.add_argument("--save-epochs", type=int, nargs="+", required=True)
        command.add_argument("--output", type=Path, required=name in ("lock", "probe", "select", "evaluate"))
    args = parser.parse_args(argv)
    if getattr(args, "threads", None):
        import torch

        torch.set_num_threads(args.threads)
    train_frame = read_train(args.train)
    plan = _plan(args.plan)
    blocks = None
    if args.command in ("probe", "select", "evaluate"):
        blocks = claude_glm._load_blocks(train_frame, args.qwen_features, args.styledistance_features)
    train_frame, plan = claude_glm._grouped(train_frame, plan, args.groups)
    if args.command == "lock":
        if args.output.exists():
            raise ValueError("Use a new output path; a locked reference must not be overwritten")
        result = lock(train_frame, plan, args.current_oof, args.comparison, args.output, args.recipe, args.groups)
        print(json.dumps({key: result[key] for key in ("balanced_accuracy", "s_by_fold", "probabilities_sha256",
                                                       "predictions_sha256", "folds_sha256")}, indent=2))
    elif args.command == "probe":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = probe(train_frame, plan, args.model_path, blocks["styledistance"], args.output, device=args.device)
        print(json.dumps({key: value for key, value in result.items() if key != "fit_check"}, indent=2))
        print(json.dumps([{key: round(item[key], 4) for key in ("train_loss", "train_accuracy")}
                          for item in result["fit_check"]["curves"]]))
    elif args.command == "train":
        key = train_partition(train_frame, plan, args.outer_fold, args.inner_fold, args.model_path, args.root,
                              epochs=args.epochs, save_epochs=tuple(args.save_epochs), device=args.device)
        print(json.dumps({field: key[field] for field in ("partition", "training_rows", "training_prompts",
                                                          "validation_rows", "updates", "seconds")}, indent=2))
    else:
        cache = pd.concat([pd.read_parquet(path) for path in args.current_inner], ignore_index=True)
        if args.command == "select":
            selection = select(train_frame, blocks, plan, cache, args.root)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            claude_glm._json(args.output, {"selection": selection, "outer_training": [
                {"outer_fold": item["outer_fold"], "epoch": item["epoch"]} for item in selection
                if item["epoch"] is not None]})
            print(json.dumps([{key: item[key] for key in ("outer_fold", "s", "selected")} for item in selection],
                             indent=2))
        else:
            inputs = [args.plan, args.qwen_features, args.styledistance_features, args.current_oof, args.reference,
                      *args.current_inner, *([args.groups] if args.groups else [])]
            result = evaluate(train_frame, blocks, plan, pd.read_parquet(args.current_oof), cache, args.root,
                              json.loads(args.reference.read_text()), args.output,
                              gate="topic" if args.groups else "prompt", model_path=args.model_path,
                              device=args.device,
                              provenance={"input_sha256": {str(path): claude_glm._sha256(path) for path in inputs}})
            paired = result["paired"]
            print(json.dumps({
                "balanced_accuracy": {name: values["balanced_accuracy"] for name, values in result["systems"].items()},
                "delta": paired["balanced_accuracy_delta"], "delta_95_interval": paired["delta_95_interval"],
                "promotion": result["promotion"], "selected": [item["selected"] for item in result["selection"]],
                "changes": {key: value for key, value in result["changes"].items() if key != "by_class"},
                "fable": result["fable"], "checks": result["checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.style_adapt import main as imported_main

    raise SystemExit(imported_main())
