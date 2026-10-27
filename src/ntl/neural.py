"""Offline frozen Qwen features and a prompt-grouped linear classification head.

This is a diagnostic baseline. It does not train LoRA adapters or claim a T4 benchmark.
Run ``python -m ntl.neural --help``. Neural dependencies are imported only when needed.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .constants import LABELS
from .data import fingerprint_frame, read_test, read_train, write_submission
from .folds import build_fold_manifest, load_fold_manifest, save_fold_manifest
from .gemini import _initialize_cuda_memory_stats

MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"


@dataclass(frozen=True)
class FrozenConfig:
    retained_layers: int = 24
    pool_layers: tuple[int, ...] = (7, 22)
    quantization: str = "nf4"
    max_length: int = 512
    batch_size: int = 2
    C: float = 0.005
    seed: int = 2026
    device: str = "auto"

    def validate(self) -> None:
        if not 1 <= self.retained_layers <= 36:
            raise ValueError("retained_layers must be between 1 and 36")
        if not self.pool_layers or len(set(self.pool_layers)) != len(self.pool_layers):
            raise ValueError("pool_layers must be nonempty and unique")
        if any(layer < 1 or layer > self.retained_layers for layer in self.pool_layers):
            raise ValueError("Each pooled layer must be a retained decoder block (1-based)")
        if self.quantization not in {"none", "nf4"}:
            raise ValueError("quantization must be none or nf4")
        if self.max_length < 16 or self.batch_size < 1 or self.C <= 0:
            raise ValueError("Invalid token limit, batch size, or regularization")


def validate_snapshot(directory: str | Path, *, weights: bool = True) -> dict:
    """Reject mutable/wrong model inputs before loading any serialized weights.

    With weights=False, a tokenizer-only snapshot of the pinned revision is accepted.
    """
    directory = Path(directory)
    manifest_path = directory / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Missing snapshot_manifest.json; run scripts/prepare_model.py first")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("repo_id") != MODEL_ID
        or manifest.get("revision") != MODEL_REVISION
        or (weights and manifest.get("includes_weights") is not True)
    ):
        raise ValueError(f"Snapshot must contain the pinned Qwen3-8B-Base {'weights' if weights else 'tokenizer'}")
    entries = manifest.get("files", [])
    if not entries:
        raise ValueError("Snapshot has no recorded files")
    for item in entries:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid path in snapshot manifest")
        target = directory / relative
        if not target.is_file() or target.stat().st_size != item["size"]:
            raise ValueError(f"Snapshot file missing or size changed: {relative}")
    if weights and not any(str(item["path"]).endswith(".safetensors") for item in entries):
        raise ValueError("Expected safetensors model weights")
    architecture = json.loads((directory / "config.json").read_text())
    if (
        architecture.get("model_type") != "qwen3"
        or architecture.get("num_hidden_layers") != 36
        or architecture.get("hidden_size") != 4096
    ):
        raise ValueError("Snapshot architecture is not the selected full Qwen3-8B-Base")
    return manifest


def masked_layer_mean(hidden_states, attention_mask, layers: tuple[int, ...]):
    """Concatenate float32 means; padding must contribute neither mass nor count."""
    import torch

    mask = attention_mask.unsqueeze(-1).to(dtype=torch.float32)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Cannot pool an empty token sequence")
    pooled = [(hidden_states[layer].float() * mask).sum(dim=1) / counts for layer in layers]
    return torch.cat(pooled, dim=-1)


def _versions() -> dict:
    versions = {}
    for package in ("torch", "transformers", "bitsandbytes", "numpy", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def load_tokenizer(model_path: str | Path):
    """The snapshot's tokenizer as the frozen recipe configures it; reads no model weights."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_backbone(model_path: str | Path, config: FrozenConfig) -> tuple:
    """Load the pinned tokenizer and truncated backbone exactly as the frozen recipe does.

    Returns (tokenizer, model, device, dtype, manifest); shared by extraction and adaptation.
    """
    config.validate()
    manifest = validate_snapshot(model_path)
    try:
        import torch
        from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install the neural extra; NF4 also needs the Linux gpu extra") from exc
    device = config.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if config.quantization == "nf4" and not device.startswith("cuda"):
        raise ValueError("NF4 requires a CUDA GPU; use --quantization none for CPU diagnostics")
    torch.manual_seed(config.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
        _initialize_cuda_memory_stats(torch, device)
    tokenizer = load_tokenizer(model_path)
    architecture = AutoConfig.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    architecture.num_hidden_layers = config.retained_layers
    architecture.use_cache = False
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    load_args = {
        "config": architecture,
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": dtype,
        "attn_implementation": "sdpa",
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "device_map": {"": device},
    }
    if config.quantization == "nf4":
        load_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    # AutoModel loads Qwen3Model, avoiding the unused language-model vocabulary head.
    # num_hidden_layers controls instantiation, so discarded upper blocks never reside on GPU.
    model = AutoModel.from_pretrained(str(model_path), **load_args).eval()
    return tokenizer, model, device, dtype, manifest


def extract_features(
    texts: list[str], model_path: str | Path, config: FrozenConfig
) -> tuple[np.ndarray, dict]:
    """Read only a local pinned snapshot, with no network fallback or remote code."""
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Expected nonempty passage strings")
    import torch

    started = time.perf_counter()
    tokenizer, model, device, dtype, manifest = load_backbone(model_path, config)
    chunks = []
    token_lengths: list[int] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), config.batch_size):
                batch_text = texts[start : start + config.batch_size]
                lengths = tokenizer(batch_text, truncation=False, return_length=True)["length"]
                token_lengths.extend(lengths)
                encoded = tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    max_length=config.max_length,
                    return_tensors="pt",
                ).to(device)
                output = model(**encoded, output_hidden_states=True, use_cache=False)
                pooled = masked_layer_mean(
                    output.hidden_states, encoded["attention_mask"], config.pool_layers
                )
                chunks.append(pooled.cpu().numpy())
                del output, pooled, encoded
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        profile = {
            "elapsed_seconds": time.perf_counter() - started,
            "device": device,
            "device_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU",
            "compute_dtype": str(dtype),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None
            ),
            "rows": len(texts),
            "truncated_rows": sum(length > config.max_length for length in token_lengths),
            "token_length_quantiles": {
                str(q): float(np.quantile(token_lengths, q)) for q in (0.5, 0.9, 0.99, 1.0)
            },
            "model_id": manifest["repo_id"],
            "model_revision": manifest["revision"],
            "versions": _versions(),
        }
        features = np.concatenate(chunks).astype(np.float32, copy=False)
        if not np.isfinite(features).all():
            raise ValueError("Backbone produced non-finite features; inspect precision and inputs")
        return features, profile
    finally:
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def _head(config: FrozenConfig):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=config.C, max_iter=2000, random_state=config.seed),
    )


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _labels(train: pd.DataFrame) -> np.ndarray:
    return np.array([LABELS.index(label) for label in train["model"]], dtype=np.int64)


def _probabilities(head, features: np.ndarray) -> np.ndarray:
    if not np.array_equal(head.classes_, np.arange(len(LABELS))):
        raise ValueError("Classifier does not contain the complete canonical label order")
    probabilities = head.predict_proba(features)
    if not np.isfinite(probabilities).all():
        raise ValueError("Classifier produced non-finite probabilities")
    return probabilities


def run_cv(
    train: pd.DataFrame,
    model_path: str | Path,
    output_dir: str | Path,
    config: FrozenConfig,
    n_splits: int = 5,
    fold_manifest: str | Path | None = None,
) -> dict:
    manifest, fold_metadata = (
        load_fold_manifest(train, fold_manifest)
        if fold_manifest is not None
        else build_fold_manifest(train, n_splits=n_splits, seed=config.seed)
    )
    features, profile = extract_features(train["text"].tolist(), model_path, config)
    targets = _labels(train)
    assignments = train["id"].map(manifest.set_index("id")["fold"]).to_numpy()
    oof = np.full((len(train), len(LABELS)), np.nan, dtype=np.float64)
    scores = []
    for fold in range(fold_metadata["n_splits"]):
        fit, valid = assignments != fold, assignments == fold
        head = _head(config)
        with threadpool_limits(limits=2):
            head.fit(features[fit], targets[fit])
            oof[valid] = _probabilities(head, features[valid])
        scores.append(float(balanced_accuracy_score(targets[valid], oof[valid].argmax(axis=1))))
    if not np.isfinite(oof).all():
        raise ValueError("OOF predictions are incomplete or non-finite")
    predictions = oof.argmax(axis=1)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "fold_scores": scores,
        "labels": list(LABELS),
        "config": asdict(config),
        "train_fingerprint": fingerprint_frame(train),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "validation": "prompt-grouped; frozen unsupervised feature extraction, fold-fitted heads",
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    save_fold_manifest(manifest, fold_metadata, directory)
    frame = train[["id", "prompt_id", "model"]].copy()
    frame["fold"] = assignments
    frame["prediction"] = [LABELS[index] for index in predictions]
    for index, label in enumerate(LABELS):
        frame[f"probability__{label}"] = oof[:, index]
    frame.to_parquet(directory / "oof.parquet", index=False)
    pd.DataFrame({
        "model": LABELS,
        "recall": recall_score(targets, predictions, labels=np.arange(len(LABELS)), average=None),
    }).to_csv(directory / "class_metrics.csv", index=False)
    pd.DataFrame(
        confusion_matrix(targets, predictions, labels=np.arange(len(LABELS))),
        index=pd.Index(LABELS, name="true_model"), columns=LABELS,
    ).to_csv(directory / "confusion_matrix.csv")
    _json(directory / "metrics.json", metrics)
    _json(directory / "feature_profile.json", profile)
    return metrics


def train_head(
    train: pd.DataFrame, model_path: str | Path, output_dir: str | Path, config: FrozenConfig
) -> dict:
    features, profile = extract_features(train["text"].tolist(), model_path, config)
    head = _head(config)
    with threadpool_limits(limits=2):
        head.fit(features, _labels(train))
    metadata = {
        "schema_version": 1,
        "kind": "frozen-qwen-linear-head",
        "labels": list(LABELS),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "snapshot_path": str(Path(model_path).resolve()),
        "config": asdict(config),
        "train_fingerprint": fingerprint_frame(train),
        "n_train": len(train),
        "versions": _versions(),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump({"head": head, "metadata": metadata}, directory / "head.joblib")
    _json(directory / "metadata.json", metadata)
    _json(directory / "feature_profile.json", profile)
    return metadata


def predict_head(
    model_dir: str | Path,
    test: pd.DataFrame,
    output: str | Path,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    # joblib is trusted project output; it must never load an untrusted third-party pickle.
    artifact = joblib.load(Path(model_dir) / "head.joblib")
    metadata = artifact["metadata"]
    if (
        metadata.get("kind") != "frozen-qwen-linear-head"
        or metadata.get("labels") != list(LABELS)
        or metadata.get("model_id") != MODEL_ID
        or metadata.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("Incompatible neural artifact metadata or label order")
    config_values = dict(metadata["config"])
    config_values["pool_layers"] = tuple(config_values["pool_layers"])
    config = FrozenConfig(**config_values)
    features, profile = extract_features(
        test["text"].tolist(), model_path or metadata["snapshot_path"], config
    )
    with threadpool_limits(limits=2):
        probabilities = _probabilities(artifact["head"], features)
    result = write_submission(test["id"], [LABELS[i] for i in probabilities.argmax(axis=1)], output)
    output_path = Path(output)
    _json(output_path.with_name(f"{output_path.stem}_profile.json"), profile)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("cv", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--train", required=True, type=Path)
        sub.add_argument("--model-path", required=True, type=Path)
        sub.add_argument("--output-dir", required=True, type=Path)
        sub.add_argument("--quantization", choices=("none", "nf4"), default="nf4")
        sub.add_argument("--batch-size", type=int, default=2)
        sub.add_argument("--max-length", type=int, default=512)
        sub.add_argument("--retained-layers", type=int, default=24)
        sub.add_argument("--pool-layers", type=int, nargs="+", default=[7, 22])
        sub.add_argument("--C", type=float, default=0.005)
        sub.add_argument("--seed", type=int, default=2026)
        sub.add_argument("--device", default="auto")
        if name == "cv":
            sub.add_argument("--folds", type=int, default=5)
            sub.add_argument("--fold-manifest", type=Path)
    predict = commands.add_parser("predict")
    predict.add_argument("--model-dir", required=True, type=Path)
    predict.add_argument("--model-path", type=Path)
    predict.add_argument("--test", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "predict":
        result = predict_head(args.model_dir, read_test(args.test), args.output, args.model_path)
        print(f"Wrote {len(result)} predictions to {args.output}")
        return 0
    config = FrozenConfig(
        retained_layers=args.retained_layers, pool_layers=tuple(args.pool_layers),
        quantization=args.quantization, max_length=args.max_length, batch_size=args.batch_size,
        C=args.C, seed=args.seed, device=args.device,
    )
    config.validate()
    train = read_train(args.train)
    if args.command == "cv":
        result = run_cv(
            train, args.model_path, args.output_dir, config, args.folds, args.fold_manifest
        )
        print(f"Prompt-grouped balanced accuracy: {result['balanced_accuracy']:.5f}")
    else:
        train_head(train, args.model_path, args.output_dir, config)
        print(f"Saved frozen feature classifier to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
