"""Frozen Qwen3-8B-Base features, their 18-class OOF head, and a nested CPU blend.

Frozen extraction is pointwise and label-free, so every official passage is embedded
once and cached with exact ID, text and array hashes. Every TF-IDF vocabulary, scaler
and classifier is fit only on its inner or outer training partition. The blend
``(1 - w) * CPU + w * Qwen`` takes ``w`` from inner prompt folds; each outer
partition is predicted once, and both models score every passage.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from threadpoolctl import threadpool_limits

from . import neural
from .constants import LABELS
from .data import fingerprint_frame, read_train, write_submission
from .folds import inner_prompt_splits, load_fold_manifest
from .frozen_gemini import load_feature_archive, save_feature_archive
from .modeling import BaselineConfig, aligned_probabilities, build_pipeline

RECORDED_REFERENCE_BALANCED_ACCURACY = 0.359667
# neural.validate_snapshot pins the backbone's hidden width.
HIDDEN_SIZE = 4096


@dataclass(frozen=True)
class BlendConfig:
    cpu: BaselineConfig = field(default_factory=BaselineConfig)
    qwen: neural.FrozenConfig = field(default_factory=neural.FrozenConfig)
    weight_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    inner_folds: int = 3
    seed: int = 2026
    bootstrap_samples: int = 10000
    # Fixed before the first blend result: the user's practical target and fold support.
    minimum_gain_over_recorded_reference: float = 0.01
    minimum_folds_beating_cpu: int = 4
    maximum_zero_weight_folds: int = 1

    def validate(self) -> None:
        self.qwen.validate()
        if not self.weight_grid or len(set(self.weight_grid)) != len(self.weight_grid):
            raise ValueError("Blend weight grid must be nonempty and contain no duplicates")
        if any(not 0 <= weight <= 1 for weight in self.weight_grid):
            raise ValueError("Blend weights must lie in [0, 1]")
        if self.inner_folds < 2 or self.bootstrap_samples < 1:
            raise ValueError("Use at least two inner folds and one bootstrap sample")


def _json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def feature_options(config: neural.FrozenConfig) -> dict:
    return {
        "pooling": "masked token mean of each listed decoder block output, concatenated",
        "pool_layers": list(config.pool_layers), "retained_layers": config.retained_layers,
        "quantization": config.quantization, "max_length": config.max_length,
        "batch_size": config.batch_size, "normalization": "none", "padding_side": "right",
        "language_model_head": False, "use_cache": False,
    }


def extract_archive(frame: pd.DataFrame, model_path: str | Path, output: str | Path,
                    config: neural.FrozenConfig) -> dict:
    """Embed every row once with the frozen recipe; the archive pins ids, text and bytes."""
    output = Path(output)
    if output.exists():
        raise ValueError("Feature output already exists; preserve it or choose another output path")
    features, profile = neural.extract_features(frame["text"].tolist(), model_path, config)
    if features.shape != (len(frame), HIDDEN_SIZE * len(config.pool_layers)):
        raise ValueError("Unexpected frozen Qwen feature shape")
    metadata = {"model_id": neural.MODEL_ID, "model_revision": neural.MODEL_REVISION,
                "encoder_frozen": True, "feature_options": feature_options(config), "profile": profile}
    if {"prompt_id", "model"}.issubset(frame.columns):
        metadata["train_fingerprint"] = fingerprint_frame(frame)
    saved = save_feature_archive(output, frame["id"].tolist(), features, metadata, texts=frame["text"].tolist())
    output.with_suffix(".json").write_text(json.dumps(saved, indent=2) + "\n")
    return saved


def load_archive(path: str | Path, frame: pd.DataFrame, config: neural.FrozenConfig) -> tuple[np.ndarray, dict]:
    """Align a verified archive to the frame and require the configured frozen recipe."""
    features, metadata = load_feature_archive(path, frame)
    if metadata.get("model_id") != neural.MODEL_ID or metadata.get("model_revision") != neural.MODEL_REVISION:
        raise ValueError("Features were not extracted from the pinned Qwen3-8B-Base revision")
    expected = feature_options(config)
    stored = metadata["feature_options"]
    # Batch size and padding change fp16 numerics slightly, so they are part of the recipe too.
    for key in ("pool_layers", "retained_layers", "quantization", "max_length", "normalization",
                "batch_size", "padding_side"):
        if stored.get(key) != expected[key]:
            raise ValueError(f"Feature archive {key} differs from the configured frozen recipe")
    return features, metadata


def _targets(frame: pd.DataFrame) -> np.ndarray:
    return frame["model"].map({label: index for index, label in enumerate(LABELS)}).to_numpy()


def _cpu_probabilities(fit_texts: np.ndarray, fit_labels: np.ndarray, texts: np.ndarray,
                       config: BaselineConfig) -> np.ndarray:
    pipeline = build_pipeline(config)
    with threadpool_limits(limits=2):
        pipeline.fit(fit_texts, fit_labels)
        return aligned_probabilities(pipeline, texts)


def _qwen_probabilities(fit_features: np.ndarray, fit_targets: np.ndarray, features: np.ndarray,
                        config: neural.FrozenConfig) -> np.ndarray:
    # The existing recipe's fresh StandardScaler and C=0.005 logistic head.
    head = neural._head(config)
    with threadpool_limits(limits=2):
        head.fit(fit_features, fit_targets)
        probabilities = neural._probabilities(head, features)
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Qwen head probabilities must sum to one")
    return probabilities


def _blend(cpu: np.ndarray, qwen: np.ndarray, weight: float) -> np.ndarray:
    return (1 - weight) * cpu + weight * qwen


def _outer_assignments(train: pd.DataFrame, manifest_path: str | Path) -> tuple[np.ndarray, dict]:
    manifest, metadata = load_fold_manifest(train, manifest_path)
    if metadata["n_splits"] != 5:
        raise ValueError("This comparison requires the original five prompt-grouped folds")
    return train["id"].map(manifest.set_index("id")["fold"]).to_numpy(), metadata


def _system_metrics(truth: np.ndarray, predictions: np.ndarray, folds: np.ndarray) -> dict:
    recall = recall_score(truth, predictions, labels=np.arange(len(LABELS)), average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "fold_balanced_accuracy": [float(balanced_accuracy_score(truth[folds == fold], predictions[folds == fold]))
                                   for fold in range(5)],
        "recall": {label: float(recall[index]) for index, label in enumerate(LABELS)},
    }


def prompt_bootstrap(truth: np.ndarray, first: np.ndarray, second: np.ndarray, prompts: np.ndarray,
                     samples: int, seed: int) -> dict:
    """Balanced-accuracy change (second - first) when whole prompts are resampled."""
    _, prompt_index = np.unique(prompts, return_inverse=True)
    shape = (prompt_index.max() + 1, len(LABELS))
    totals, first_correct, second_correct = (np.zeros(shape) for _ in range(3))
    np.add.at(totals, (prompt_index, truth), 1)
    np.add.at(first_correct, (prompt_index, truth), first == truth)
    np.add.at(second_correct, (prompt_index, truth), second == truth)
    draws = np.random.default_rng(seed).integers(0, shape[0], size=(samples, shape[0]))
    weights = np.zeros((samples, shape[0]))
    np.add.at(weights, (np.repeat(np.arange(samples), shape[0]), draws.ravel()), 1)
    denominator = weights @ totals
    with np.errstate(invalid="ignore", divide="ignore"):
        deltas = np.nanmean((weights @ second_correct - weights @ first_correct) / denominator, axis=1)
    observed = balanced_accuracy_score(truth, second) - balanced_accuracy_score(truth, first)
    return {"balanced_accuracy_delta": float(observed), "samples": samples, "seed": seed,
            "prompts": int(shape[0]),
            "delta_95_interval": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "fraction_delta_at_most_zero": float((deltas <= 0).mean()),
            "second_only_correct": int(((second == truth) & (first != truth)).sum()),
            "first_only_correct": int(((first == truth) & (second != truth)).sum())}


def qwen_oof(train: pd.DataFrame, features: np.ndarray, manifest_path: str | Path, output_dir: str | Path,
             config: neural.FrozenConfig) -> dict:
    """Step 1: one held-out 18-class Qwen score vector for every official row."""
    output = Path(output_dir)
    if (output / "oof.parquet").exists():
        raise ValueError("Use a new output directory; an existing OOF must not be overwritten")
    folds, fold_metadata = _outer_assignments(train, manifest_path)
    truth = _targets(train)
    probabilities = np.full((len(train), len(LABELS)), np.nan)
    for fold in range(5):
        fit, valid = folds != fold, folds == fold
        if set(train.loc[fit, "prompt_id"]) & set(train.loc[valid, "prompt_id"]):
            raise ValueError("Prompt leakage detected before the Qwen head fit")
        probabilities[valid] = _qwen_probabilities(features[fit], truth[fit], features[valid], config)
    if not np.isfinite(probabilities).all():
        raise ValueError("Every official row must receive exactly one finite OOF score vector")
    predictions = probabilities.argmax(axis=1)
    output.mkdir(parents=True, exist_ok=True)
    result = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    result["fold"] = folds
    result["predicted_model"] = np.asarray(LABELS)[predictions]
    for index, label in enumerate(LABELS):
        result[f"probability::{label}"] = probabilities[:, index]
    result.to_parquet(output / "oof.parquet", index=False)
    metrics = {**_system_metrics(truth, predictions, folds), "n_rows": len(train),
               "configuration": asdict(config), "validation": fold_metadata,
               "head_policy": "Fresh StandardScaler and logistic head fit on each outer training partition."}
    _json(output / "metrics.json", metrics)
    return metrics


def _select_weight(texts, labels, features, targets, prompts, config: BlendConfig, outer_fold: int):
    """Both classifiers are refit inside every inner split; no outer row is passed in."""
    splits, split_metadata = inner_prompt_splits(targets, prompts, config.inner_folds, config.seed + outer_fold)
    scores = {weight: [] for weight in config.weight_grid}
    for fit, valid in splits:
        cpu = _cpu_probabilities(texts[fit], labels[fit], texts[valid], config.cpu)
        qwen = _qwen_probabilities(features[fit], targets[fit], features[valid], config.qwen)
        for weight in config.weight_grid:
            scores[weight].append(float(balanced_accuracy_score(targets[valid], _blend(cpu, qwen, weight).argmax(axis=1))))
    candidates = [{"weight": weight, "inner_balanced_accuracy": values,
                   "mean_inner_balanced_accuracy": float(np.mean(values))}
                  for weight, values in sorted(scores.items())]
    # Ties keep the smaller Qwen weight, i.e. stay closer to the CPU-only system.
    best = min(candidates, key=lambda item: (-item["mean_inner_balanced_accuracy"], item["weight"]))
    return best["weight"], {"outer_fold": outer_fold, "selected_weight": best["weight"],
                            "candidates": candidates, "inner_splits": split_metadata,
                            "selection_metric": "Mean inner prompt-fold balanced accuracy; ties choose smaller weight."}


def evaluate_blend(
    train: pd.DataFrame, features: np.ndarray, manifest_path: str | Path, output_dir: str | Path,
    config: BlendConfig | None = None, reference_oof: pd.DataFrame | None = None,
    feature_metadata: dict | None = None,
) -> dict:
    """Steps 2-3: nested CPU-Qwen blend on all rows, compared with both CPU references."""
    config = config or BlendConfig()
    config.validate()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory; an existing experiment must not be overwritten")
    if len(features) != len(train) or not np.isfinite(features).all():
        raise ValueError("Features must be finite and aligned to every training row")
    folds, fold_metadata = _outer_assignments(train, manifest_path)
    texts = train["text"].to_numpy(dtype=object)
    labels = train["model"].to_numpy(dtype=object)
    truth = _targets(train)
    prompts = train["prompt_id"].to_numpy()
    cpu, qwen, blended = (np.full((len(train), len(LABELS)), np.nan) for _ in range(3))
    weights, nested = np.full(len(train), np.nan), []
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for fold in range(5):
        fit, valid = folds != fold, folds == fold
        if set(prompts[fit]) & set(prompts[valid]):
            raise ValueError("Prompt leakage detected before the outer fits")
        weight, record = _select_weight(texts[fit], labels[fit], features[fit], truth[fit], prompts[fit],
                                        config, fold)
        cpu[valid] = _cpu_probabilities(texts[fit], labels[fit], texts[valid], config.cpu)
        qwen[valid] = _qwen_probabilities(features[fit], truth[fit], features[valid], config.qwen)
        blended[valid] = _blend(cpu[valid], qwen[valid], weight)
        weights[valid] = weight
        record["outer_balanced_accuracy"] = {
            name: float(balanced_accuracy_score(truth[valid], values[valid].argmax(axis=1)))
            for name, values in (("cpu", cpu), ("qwen", qwen), ("blend", blended))}
        nested.append(record)
        _json(output / "inner_cv.json", nested)
        print(json.dumps({"outer_fold": fold, "selected_weight": weight, **record["outer_balanced_accuracy"]}),
              flush=True)
    if not all(np.isfinite(values).all() for values in (cpu, qwen, blended)):
        raise ValueError("Every official row must receive one finite outer score vector per system")
    predictions = {"cpu": cpu.argmax(axis=1), "qwen": qwen.argmax(axis=1), "blend": blended.argmax(axis=1)}
    if reference_oof is not None:
        reference = train[["id"]].merge(reference_oof, on="id", how="left", validate="one_to_one")
        if (reference["true_model"].to_numpy() != labels).any() or (reference["fold"].to_numpy() != folds).any():
            raise ValueError("Reference OOF labels or folds differ from the official rows and manifest")
        predictions["recorded_reference"] = reference["predicted_model"].map(LABELS.index).to_numpy()
    result = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    result["fold"] = folds
    result["blend_weight"] = weights
    for name, values in predictions.items():
        result[f"prediction::{name}"] = np.asarray(LABELS)[values]
    for name, values in (("cpu", cpu), ("qwen", qwen), ("blend", blended)):
        for index, label in enumerate(LABELS):
            result[f"{name}::{label}"] = values[:, index]
    result.to_parquet(output / "oof.parquet", index=False)
    systems = {name: _system_metrics(truth, values, folds) for name, values in predictions.items()}
    pairs = [("cpu", "blend"), ("cpu", "qwen")]
    if "recorded_reference" in predictions:
        pairs += [("recorded_reference", "blend"), ("recorded_reference", "cpu")]
    paired = {f"{second}_vs_{first}": prompt_bootstrap(truth, predictions[first], predictions[second], prompts,
                                                       config.bootstrap_samples, config.seed)
              for first, second in pairs}
    selected = [record["selected_weight"] for record in nested]
    blend_folds, cpu_folds = systems["blend"]["fold_balanced_accuracy"], systems["cpu"]["fold_balanced_accuracy"]
    gain = systems["blend"]["balanced_accuracy"] - RECORDED_REFERENCE_BALANCED_ACCURACY
    gate = {
        "gain_over_recorded_reference": gain,
        "gain_over_same_environment_cpu": systems["blend"]["balanced_accuracy"] - systems["cpu"]["balanced_accuracy"],
        "folds_blend_beats_same_environment_cpu": int(sum(b > c for b, c in zip(blend_folds, cpu_folds, strict=True))),
        "zero_weight_folds": int(sum(weight == 0 for weight in selected)),
    }
    gate["passed"] = bool(gain >= config.minimum_gain_over_recorded_reference
                          and gate["folds_blend_beats_same_environment_cpu"] >= config.minimum_folds_beating_cpu
                          and gate["zero_weight_folds"] <= config.maximum_zero_weight_folds)
    versions = {package: importlib.metadata.version(package) for package in ("numpy", "scikit-learn", "scipy")}
    comparison = {"systems": systems, "paired": paired, "selected_weight_per_outer_fold": selected,
                  "gate": gate, "recorded_reference_balanced_accuracy": RECORDED_REFERENCE_BALANCED_ACCURACY,
                  "elapsed_seconds": time.perf_counter() - started}
    _json(output / "comparison.json", comparison)
    _json(output / "metadata.json", {
        "schema_version": 1, "kind": "cpu-qwen-nested-18-class-blend", "labels": list(LABELS),
        "configuration": asdict(config), "versions": versions, "validation": fold_metadata,
        "train_fingerprint": fingerprint_frame(train), "feature_provenance": feature_metadata,
        "blend": "(1 - w) * CPU probabilities + w * Qwen probabilities, aligned by id and canonical label order.",
        "selection_policy": "w from inner prompt folds of each outer training partition; both models refit per split.",
        "same_environment_cpu": "The CPU system refit here; the recorded reference is the submitted Kaggle OOF.",
    })
    return comparison


# Outer folds used seed offsets 0-4; the one full-data selection gets its own.
FINAL_SELECTION_SEED_OFFSET = 5


def freeze_recipe(train: pd.DataFrame, features: np.ndarray, config: BlendConfig | None = None,
                  validation: dict | None = None) -> dict:
    """Apply the nested-validated weight selection once, to all official rows."""
    config = config or BlendConfig()
    config.validate()
    weight, record = _select_weight(
        train["text"].to_numpy(dtype=object), train["model"].to_numpy(dtype=object), features,
        _targets(train), train["prompt_id"].to_numpy(), config, FINAL_SELECTION_SEED_OFFSET)
    summary = None if validation is None else {
        "balanced_accuracy": {name: values["balanced_accuracy"] for name, values in validation["systems"].items()},
        "selected_weight_per_outer_fold": validation["selected_weight_per_outer_fold"], "gate": validation["gate"]}
    return {"schema_version": 1, "kind": "cpu-qwen-blend-recipe", "labels": list(LABELS), "weight": weight,
            "cpu": asdict(config.cpu), "qwen": asdict(config.qwen), "model_id": neural.MODEL_ID,
            "model_revision": neural.MODEL_REVISION, "feature_options": feature_options(config.qwen),
            "train_fingerprint": fingerprint_frame(train), "weight_selection": record,
            "nested_validation": summary}


def load_recipe(recipe: dict) -> tuple[float, BaselineConfig, neural.FrozenConfig]:
    if (recipe.get("schema_version") != 1 or recipe.get("kind") != "cpu-qwen-blend-recipe"
            or recipe.get("labels") != list(LABELS) or recipe.get("model_id") != neural.MODEL_ID
            or recipe.get("model_revision") != neural.MODEL_REVISION):
        raise ValueError("Unsupported blend recipe, label order or backbone revision")
    weight = recipe.get("weight")
    if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
        raise ValueError("Blend recipe weight must lie in [0, 1]")
    qwen = dict(recipe["qwen"], pool_layers=tuple(recipe["qwen"]["pool_layers"]))
    qwen_config = neural.FrozenConfig(**qwen)
    if recipe.get("feature_options") != feature_options(qwen_config):
        raise ValueError("Blend recipe feature options differ from its frozen configuration")
    return float(weight), BaselineConfig(**recipe["cpu"]), qwen_config


def fit_final(train: pd.DataFrame, model_path: str | Path, recipe: dict, output_dir: str | Path,
              device: str = "auto") -> dict:
    """Refit both classifiers on every official row for the frozen recipe."""
    weight, cpu_config, qwen_config = load_recipe(recipe)
    if recipe["train_fingerprint"] != fingerprint_frame(train):
        raise ValueError("Training data differ from the rows used to freeze the recipe")
    output = Path(output_dir)
    if (output / "blend.joblib").exists():
        raise ValueError("Use a new output directory; an existing fitted blend must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    qwen_config = neural.FrozenConfig(**{**asdict(qwen_config), "device": device})
    started = time.perf_counter()
    saved = extract_archive(train, model_path, output / "train-features.npz", qwen_config)
    features, _ = load_archive(output / "train-features.npz", train, qwen_config)
    cpu = build_pipeline(cpu_config)
    head = neural._head(qwen_config)
    with threadpool_limits(limits=2):
        cpu.fit(train["text"], train["model"])
        head.fit(features, _targets(train))
    # Fail early if either classifier lost a label or its canonical alignment.
    aligned_probabilities(cpu, train["text"].iloc[:1])
    neural._probabilities(head, features[:1])
    metadata = {"weight": weight, "train_fingerprint": fingerprint_frame(train), "n_train": len(train),
                "feature_profile": saved["profile"], "fit_seconds": time.perf_counter() - started,
                "versions": {package: importlib.metadata.version(package)
                             for package in ("numpy", "scikit-learn", "scipy")}}
    # joblib is trusted project output; never load a third-party pickle with predict_final.
    joblib.dump({"cpu": cpu, "qwen_head": head, "recipe": recipe, "metadata": metadata}, output / "blend.joblib")
    _json(output / "fit.json", metadata)
    return metadata


def predict_final(model_dir: str | Path, test: pd.DataFrame, model_path: str | Path, output: str | Path,
                  device: str = "auto") -> pd.DataFrame:
    """Score every current test row with both models and the frozen blend weight."""
    artifact = joblib.load(Path(model_dir) / "blend.joblib")
    weight, _, qwen_config = load_recipe(artifact["recipe"])
    if artifact["metadata"]["weight"] != weight:
        raise ValueError("Fitted blend weight differs from its recipe")
    started = time.perf_counter()
    qwen_config = neural.FrozenConfig(**{**asdict(qwen_config), "device": device})
    features, profile = neural.extract_features(test["text"].tolist(), model_path, qwen_config)
    with threadpool_limits(limits=2):
        cpu = aligned_probabilities(artifact["cpu"], test["text"])
        qwen = neural._probabilities(artifact["qwen_head"], features)
    blended = _blend(cpu, qwen, weight)
    if not np.isfinite(blended).all() or not np.allclose(blended.sum(axis=1), 1.0):
        raise ValueError("Blended scores must be finite probability vectors")
    submission = write_submission(test["id"], np.asarray(LABELS)[blended.argmax(axis=1)].tolist(), output)
    output = Path(output)
    _json(output.with_name(f"{output.stem}_profile.json"),
          {"rows": len(test), "weight": weight, "elapsed_seconds": time.perf_counter() - started,
           "feature_profile": profile})
    return submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="Embed every row of a CSV with the frozen recipe")
    extract.add_argument("--input", type=Path, required=True, help="train.csv or test.csv")
    extract.add_argument("--model-path", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--device", default="auto")
    for name in ("qwen-oof", "blend-cv"):
        sub = commands.add_parser(name)
        sub.add_argument("--train", type=Path, required=True)
        sub.add_argument("--fold-manifest", type=Path, required=True)
        sub.add_argument("--features", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
    commands.choices["blend-cv"].add_argument("--reference-oof", type=Path)
    freeze = commands.add_parser("freeze", help="Select the final weight on all rows and write the recipe")
    freeze.add_argument("--train", type=Path, required=True)
    freeze.add_argument("--features", type=Path, required=True)
    freeze.add_argument("--validation", type=Path, help="comparison.json from blend-cv")
    freeze.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("fit", help="Refit both classifiers on all rows for a frozen recipe")
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--model-path", type=Path, required=True)
    fit.add_argument("--recipe", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--device", default="auto")
    predict = commands.add_parser("predict", help="Write submission.parquet from a fitted blend")
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--test", type=Path, required=True)
    predict.add_argument("--model-path", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.command == "fit":
        metadata = fit_final(read_train(args.train), args.model_path, json.loads(args.recipe.read_text()),
                             args.output_dir, args.device)
        print(json.dumps({key: metadata[key] for key in ("weight", "n_train", "fit_seconds")}, indent=2))
        return 0
    if args.command == "predict":
        from .data import read_test
        submission = predict_final(args.model_dir, read_test(args.test), args.model_path, args.output, args.device)
        print(json.dumps({"submission": str(args.output), "rows": len(submission)}, indent=2))
        return 0
    if args.command == "extract":
        from .data import read_test
        frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        frame = read_train(args.input) if "model" in frame.columns else read_test(args.input)
        saved = extract_archive(frame, args.model_path, args.output, neural.FrozenConfig(device=args.device))
        print(json.dumps({"feature_archive": str(args.output), **saved["profile"]}, indent=2))
        return 0
    train = read_train(args.train)
    config = BlendConfig()
    features, metadata = load_archive(args.features, train, config.qwen)
    if args.command == "freeze":
        if args.output.exists():
            raise ValueError("Recipe output already exists; preserve it or choose another path")
        validation = None if args.validation is None else json.loads(args.validation.read_text())
        recipe = freeze_recipe(train, features, config, validation)
        _json(args.output, recipe)
        print(json.dumps({"weight": recipe["weight"], "candidates": [
            (item["weight"], item["mean_inner_balanced_accuracy"]) for item in recipe["weight_selection"]["candidates"]]},
            indent=2))
        return 0
    if args.command == "qwen-oof":
        metrics = qwen_oof(train, features, args.fold_manifest, args.output_dir, config.qwen)
        print(json.dumps({key: metrics[key] for key in ("balanced_accuracy", "fold_balanced_accuracy")}, indent=2))
        return 0
    reference = None if args.reference_oof is None else pd.read_parquet(args.reference_oof)
    comparison = evaluate_blend(train, features, args.fold_manifest, args.output_dir, config, reference, metadata)
    print(json.dumps({"balanced_accuracy": {name: values["balanced_accuracy"]
                                            for name, values in comparison["systems"].items()},
                      "selected_weight_per_outer_fold": comparison["selected_weight_per_outer_fold"],
                      "gate": comparison["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
