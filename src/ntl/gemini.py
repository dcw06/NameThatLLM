"""A fixed, prompt-grouped DeBERTa experiment for exact Gemini version attribution.

Train each fresh four-class head/backbone on every true-Gemini training-fold row.
Evaluate all 600 held-out Gemini rows and the preselected 570 correctly routed by
the saved CPU out-of-fold predictions. Outer validation never selects an epoch.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .constants import LABELS
from .data import fingerprint_frame, read_train, validate_train
from .folds import load_fold_manifest, save_fold_manifest

MODEL_ID = "microsoft/deberta-v3-base"
MODEL_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
GEMINI_LABELS = tuple(label for label in LABELS if label.startswith("gemini-"))
OFFICIAL_BASELINE_COUNTS = {
    "train_rows": 2699,
    "gemini_rows": 600,
    "selected_rows": 570,
    "baseline_selected_correct": 291,
}


@dataclass(frozen=True)
class GeminiConfig:
    epochs: int = 3
    learning_rate: float = 2e-5
    max_length: int = 512
    batch_size: int = 8
    gradient_accumulation: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    amp_initial_scale: float = 1024.0
    max_consecutive_amp_skips: int = 20
    seed: int = 2026
    device: str = "cuda:0"

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("Epochs, microbatch size and accumulation must be positive")
        if not 16 <= self.max_length <= 512:
            raise ValueError("The selected DeBERTa protocol requires a token limit from 16 to 512")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("Warmup ratio must be in [0, 1)")
        if self.amp_initial_scale <= 0 or self.max_consecutive_amp_skips < 1:
            raise ValueError("Invalid AMP loss-scale recovery settings")


def _json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _frame_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values("id").loc[:, columns].itertuples(index=False, name=None):
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_gemini_evaluation(
    train: pd.DataFrame,
    baseline_oof: pd.DataFrame,
    fold_manifest_path: str | Path,
    expected_counts: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Join exact baseline OOF truth/IDs/folds before selecting the fixed cohort.

    Selection changes the reporting cohort only, never the training population.
    The fingerprinted fold manifest always belongs to the complete 18-class train.
    """
    train = validate_train(train)
    manifest, fold_metadata = load_fold_manifest(train, fold_manifest_path)
    if fold_metadata["n_splits"] != 5:
        raise ValueError("This fixed comparison requires the original five prompt-grouped folds")
    required = {"id", "prompt_id", "true_model", "fold", "predicted_model"}
    if not required.issubset(baseline_oof.columns):
        raise ValueError(f"Baseline OOF is missing columns: {sorted(required - set(baseline_oof.columns))}")
    baseline = baseline_oof.loc[:, sorted(required)].copy()
    if baseline.isna().any().any() or baseline["id"].duplicated().any():
        raise ValueError("Baseline OOF contains null values or duplicate ids")
    for column in ("id", "prompt_id", "true_model", "predicted_model"):
        if not baseline[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Baseline OOF {column} must contain nonempty strings")
    if not pd.api.types.is_integer_dtype(baseline["fold"]):
        raise ValueError("Baseline OOF fold assignments must be integers")
    if len(baseline) != len(train) or set(baseline["id"]) != set(train["id"]):
        raise ValueError("Baseline OOF must contain every official training id exactly once")
    if not set(baseline["predicted_model"]).issubset(LABELS):
        raise ValueError("Baseline OOF contains predictions outside the 18 valid labels")
    merged = train.merge(baseline, on="id", suffixes=("", "_baseline"), validate="one_to_one")
    expected_folds = merged["id"].map(manifest.set_index("id")["fold"])
    if not (merged["prompt_id"] == merged["prompt_id_baseline"]).all():
        raise ValueError("Baseline OOF prompt ids differ from official training rows")
    if not (merged["model"] == merged["true_model"]).all():
        raise ValueError("Baseline OOF true model labels differ from official training rows")
    if not (merged["fold"] == expected_folds).all():
        raise ValueError("Baseline OOF folds differ from the saved prompt-grouped manifest")
    gemini = merged.loc[merged["model"].isin(GEMINI_LABELS)].copy()
    gemini = gemini.rename(columns={"predicted_model": "baseline_model"})
    gemini["selected"] = gemini["baseline_model"].isin(GEMINI_LABELS)
    gemini["baseline_correct"] = gemini["baseline_model"] == gemini["model"]
    gemini["target"] = gemini["model"].map({label: index for index, label in enumerate(GEMINI_LABELS)})
    gemini = gemini.sort_values("id", kind="stable").reset_index(drop=True)
    for fold in range(5):
        fitting = gemini.loc[gemini["fold"] != fold]
        validation = gemini.loc[gemini["fold"] == fold]
        if set(fitting["prompt_id"]) & set(validation["prompt_id"]):
            raise ValueError("Prompt leakage in Gemini fold assignments")
        if set(fitting["model"]) != set(GEMINI_LABELS) or set(validation["model"]) != set(GEMINI_LABELS):
            raise ValueError("Every Gemini class must occur in fitting and validation for every fold")
    counts = {
        "train_rows": len(train), "gemini_rows": len(gemini),
        "selected_rows": int(gemini["selected"].sum()),
        "baseline_selected_correct": int((gemini["selected"] & gemini["baseline_correct"]).sum()),
    }
    if expected_counts is not None:
        for key, expected in expected_counts.items():
            if counts.get(key) != expected:
                raise ValueError(f"Saved baseline cohort mismatch for {key}: expected {expected}, found {counts.get(key)}")
    protocol = {
        "counts": counts,
        "train_fingerprint": fingerprint_frame(train),
        "baseline_oof_fingerprint": _frame_digest(baseline, ["id", "prompt_id", "true_model", "fold", "predicted_model"]),
        "fold_metadata": fold_metadata,
        "labels": list(GEMINI_LABELS),
        "cohort_definition": "True model is Gemini AND saved CPU OOF prediction is Gemini; selection is for reporting only.",
        "training_population": "All true-Gemini rows in the fitting prompt folds, including CPU family false negatives.",
        "outer_validation_policy": "Fixed epochs; no checkpoint, epoch, hyperparameter or threshold selection on outer validation.",
    }
    return gemini, protocol


def validate_model_snapshot(directory: str | Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "snapshot_manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("repo_id") != MODEL_ID
            or manifest.get("revision") != MODEL_REVISION or manifest.get("includes_weights") is not True):
        raise ValueError("Expected a local snapshot of the pinned microsoft/deberta-v3-base revision")
    entries = manifest.get("files", [])
    declared = set()
    if not entries:
        raise ValueError("Snapshot manifest has no file inventory")
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in declared:
            raise ValueError("Snapshot manifest contains an unsafe or repeated file path")
        declared.add(str(relative))
        target = directory / relative
        if target.is_symlink() or not target.is_file() or target.stat().st_size != entry["size"]:
            raise ValueError(f"Missing or changed pinned snapshot file: {relative}")
    if not {"config.json", "spm.model", "pytorch_model.bin"}.issubset(declared):
        raise ValueError("Pinned DeBERTa snapshot needs config.json, spm.model and pytorch_model.bin")
    architecture = json.loads((directory / "config.json").read_text())
    if (architecture.get("model_type"), architecture.get("hidden_size"),
            architecture.get("num_hidden_layers")) != ("deberta-v2", 768, 12):
        raise ValueError("Snapshot architecture does not match DeBERTa-v3-base")
    return manifest


def validate_experiment_inputs(
    train_path: str | Path, baseline_oof_path: str | Path, fold_manifest_path: str | Path,
) -> dict:
    """CPU-only preflight before downloading a model or allocating GPU memory."""
    _, protocol = prepare_gemini_evaluation(
        read_train(train_path), pd.read_parquet(baseline_oof_path), fold_manifest_path,
        OFFICIAL_BASELINE_COUNTS,
    )
    return protocol


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False


def _initialize_cuda_memory_stats(torch, device) -> None:
    """Initialize this subprocess's caching allocator before resetting counters."""
    torch.cuda.init()
    torch.cuda.set_device(device)
    transient = torch.empty(1, device=device)
    del transient
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _fit_predict_fold(
    fitting: pd.DataFrame, validation: pd.DataFrame, model_path: str | Path,
    config: GeminiConfig, fold: int,
) -> tuple[np.ndarray, dict]:
    """Full fine-tuning; a new pretrained backbone and random head per outer fold."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    config.validate()
    device = torch.device(config.device)
    cuda = device.type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise ValueError("This experiment requires a CUDA GPU; select a free Kaggle T4")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Use CUDA for the experiment or CPU for a tiny diagnostic")
    fold_seed = config.seed + fold
    _seed_everything(fold_seed, torch)
    started = time.perf_counter()
    if cuda:
        _initialize_cuda_memory_stats(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False, use_fast=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        use_safetensors=False, dtype=torch.float32, num_labels=len(GEMINI_LABELS),
        id2label=dict(enumerate(GEMINI_LABELS)),
        label2id={label: index for index, label in enumerate(GEMINI_LABELS)},
    ).to(device)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The experiment requires a fully trainable pretrained backbone and classifier")

    def rows(frame):
        encoded = tokenizer(frame["text"].tolist(), truncation=True, max_length=config.max_length)
        return [{**{name: values[index] for name, values in encoded.items()},
                 "labels": int(frame.iloc[index]["target"])} for index in range(len(frame))]

    def collate(items):
        labels = torch.tensor([item["labels"] for item in items], dtype=torch.long)
        batch = tokenizer.pad(
            [{key: value for key, value in item.items() if key != "labels"} for item in items],
            padding=True, return_tensors="pt",
        )
        batch["labels"] = labels
        return batch

    fitting_rows, validation_rows = rows(fitting), rows(validation)
    generator = torch.Generator().manual_seed(fold_seed)
    train_loader = DataLoader(fitting_rows, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate, generator=generator, num_workers=0)
    valid_loader = DataLoader(validation_rows, batch_size=config.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=0)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config.learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation)
    total_updates = updates_per_epoch * config.epochs
    warmup_updates = math.ceil(config.warmup_ratio * total_updates)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_updates, total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda, init_scale=config.amp_initial_scale)
    epoch_logs = []
    optimizer_updates = 0
    skipped_amp_updates = 0
    consecutive_amp_skips = 0
    try:
        for epoch in range(config.epochs):
            epoch_started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for step, batch in enumerate(train_loader):
                labels = batch.pop("labels").to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                group_start = (step // config.gradient_accumulation) * config.gradient_accumulation * config.batch_size
                group_rows = min(config.gradient_accumulation * config.batch_size, len(fitting_rows) - group_start)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cuda):
                    logits = model(**inputs).logits
                    summed_loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
                    loss = summed_loss / group_rows
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite fitting loss in fold {fold}, epoch {epoch + 1}")
                total_loss += float(summed_loss.detach().cpu())
                scaler.scale(loss).backward()
                if (step + 1) % config.gradient_accumulation == 0 or step + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    # Dynamic fp16 loss scaling may overflow scaled gradients even
                    # with a finite loss. GradScaler records that at unscale_,
                    # skips the unsafe update, and lowers its scale for retry.
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm, error_if_nonfinite=not cuda,
                    )
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < previous_scale:
                        skipped_amp_updates += 1
                        consecutive_amp_skips += 1
                        if consecutive_amp_skips >= config.max_consecutive_amp_skips:
                            raise ValueError("Persistent non-finite CUDA gradients despite AMP scale backoff")
                    else:
                        scheduler.step()
                        optimizer_updates += 1
                        consecutive_amp_skips = 0
                    optimizer.zero_grad(set_to_none=True)
            epoch_logs.append({"epoch": epoch + 1, "mean_training_loss": total_loss / len(fitting_rows),
                               "elapsed_seconds": time.perf_counter() - epoch_started,
                               "learning_rate_after_epoch": scheduler.get_last_lr()[0]})
            print(json.dumps({"fold": fold, **epoch_logs[-1]}), flush=True)
        if optimizer_updates == 0:
            raise ValueError("Fine-tuning completed without a successful optimizer update")
        # One validation pass after exactly the predeclared number of epochs.
        validation_started = time.perf_counter()
        model.eval()
        chunks = []
        with torch.inference_mode():
            for batch in valid_loader:
                batch.pop("labels")
                inputs = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cuda):
                    logits = model(**inputs).logits
                chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        probabilities = np.concatenate(chunks)
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Non-finite or invalid DeBERTa validation probabilities")
        if cuda:
            torch.cuda.synchronize(device)
        log = {
            "fold": fold, "seed": fold_seed, "train_rows": len(fitting), "validation_rows": len(validation),
            "train_prompts": int(fitting["prompt_id"].nunique()),
            "validation_prompts": int(validation["prompt_id"].nunique()),
            "training_selected_rows": int(fitting["selected"].sum()),
            "validation_selected_rows": int(validation["selected"].sum()),
            "optimizer_updates": optimizer_updates, "skipped_amp_updates": skipped_amp_updates,
            "amp_initial_scale": config.amp_initial_scale if cuda else None,
            "amp_final_scale": scaler.get_scale() if cuda else None,
            "epochs": epoch_logs,
            "validation_elapsed_seconds": time.perf_counter() - validation_started,
            "elapsed_seconds": time.perf_counter() - started,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if cuda else None,
            "precision": "float32 parameters with CUDA float16 autocast/GradScaler" if cuda else "float32 CPU diagnostic",
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
        }
        return probabilities, log
    finally:
        del model, optimizer, scheduler, scaler
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()


def _cohort_metrics(frame: pd.DataFrame) -> dict:
    truth, prediction = frame["model"], frame["prediction"]
    baseline_correct = frame["baseline_correct"].to_numpy(dtype=bool)
    specialist_correct = (truth == prediction).to_numpy(dtype=bool)
    return {
        "rows": len(frame), "baseline_correct": int(baseline_correct.sum()),
        "specialist_correct": int(specialist_correct.sum()),
        "baseline_accuracy": float(baseline_correct.mean()),
        "specialist_accuracy": float(accuracy_score(truth, prediction)),
        "specialist_balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "accuracy_delta": float(specialist_correct.mean() - baseline_correct.mean()),
        "paired_transitions": {
            "both_correct": int((baseline_correct & specialist_correct).sum()),
            "baseline_wrong_specialist_correct": int((~baseline_correct & specialist_correct).sum()),
            "baseline_correct_specialist_wrong": int((baseline_correct & ~specialist_correct).sum()),
            "both_wrong": int((~baseline_correct & ~specialist_correct).sum()),
        },
    }


def _save_class_reports(frame: pd.DataFrame, directory: Path, suffix: str) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        frame["model"], frame["prediction"], labels=list(GEMINI_LABELS), zero_division=0,
    )
    baseline_recall = frame.groupby("model")["baseline_correct"].mean().reindex(GEMINI_LABELS)
    pd.DataFrame({"model": GEMINI_LABELS, "precision": precision, "recall": recall,
                  "f1": f1, "support": support,
                  "baseline_recall": baseline_recall.to_numpy(),
                  "recall_delta": recall - baseline_recall.to_numpy()}).to_csv(
                      directory / f"class_metrics{suffix}.csv", index=False)
    pd.DataFrame(confusion_matrix(frame["model"], frame["prediction"], labels=list(GEMINI_LABELS)),
                 index=pd.Index(GEMINI_LABELS, name="true_model"), columns=GEMINI_LABELS).to_csv(
                     directory / f"confusion_matrix{suffix}.csv")


def run_experiment(
    train: pd.DataFrame, baseline_oof: pd.DataFrame, fold_manifest_path: str | Path,
    model_path: str | Path, output_dir: str | Path, config: GeminiConfig | None = None,
    expected_counts: dict[str, int] | None = OFFICIAL_BASELINE_COUNTS,
) -> dict:
    config = config or GeminiConfig()
    config.validate()
    gemini, protocol = prepare_gemini_evaluation(train, baseline_oof, fold_manifest_path, expected_counts)
    manifest = validate_model_snapshot(model_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists() or (output / "oof.parquet").exists():
        raise ValueError("Use a new output directory; an existing experiment must not be overwritten")
    versions = {}
    for package in ("torch", "transformers", "numpy", "pandas", "scikit-learn", "sentencepiece"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {"schema_version": 1, "kind": "gemini-four-class-full-finetuning-oof",
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "configuration": asdict(config), "protocol": protocol, "versions": versions,
                "snapshot_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                "checkpoint_policy": "No full checkpoints saved; diagnostic OOF results only.",
                "seed_policy": "Fresh pretrained backbone/head each fold, seed + fold; bitwise CUDA determinism is not promised."}
    _json(output / "metadata.json", metadata)
    full_manifest, fold_metadata = load_fold_manifest(train, fold_manifest_path)
    save_fold_manifest(full_manifest, fold_metadata, output)
    probabilities = np.full((len(gemini), len(GEMINI_LABELS)), np.nan)
    predicted_count = np.zeros(len(gemini), dtype=np.int64)
    training_logs = []
    started = time.perf_counter()
    for fold in range(5):
        fitting = gemini.loc[gemini["fold"] != fold].copy()
        valid_mask = gemini["fold"] == fold
        validation = gemini.loc[valid_mask].copy()
        # Fitting rows are deliberately not filtered by `selected`.
        if set(fitting["prompt_id"]) & set(validation["prompt_id"]):
            raise ValueError("Prompt leakage detected before fine-tuning")
        values, log = _fit_predict_fold(fitting, validation, model_path, config, fold)
        if values.shape != (len(validation), len(GEMINI_LABELS)) or not np.isfinite(values).all():
            raise ValueError("Fold prediction shape or finite-value validation failed")
        if (values < 0).any() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Fold probabilities must be nonnegative and sum to one")
        probabilities[valid_mask] = values
        predicted_count[valid_mask] += 1
        training_logs.append(log)
        _json(output / "training_log.json", training_logs)
    if not np.all(predicted_count == 1) or not np.isfinite(probabilities).all():
        raise ValueError("Every true-Gemini row must receive exactly one OOF prediction")
    result = gemini[["id", "prompt_id", "model", "fold", "baseline_model", "selected", "baseline_correct"]].copy()
    result["prediction"] = np.asarray(GEMINI_LABELS)[probabilities.argmax(axis=1)]
    for index, label in enumerate(GEMINI_LABELS):
        result[f"probability::{label}"] = probabilities[:, index]
    selected = result.loc[result["selected"]].copy()
    metrics = {"all_true_gemini": _cohort_metrics(result),
               "correctly_routed_gemini": _cohort_metrics(selected),
               "elapsed_seconds": time.perf_counter() - started,
               "interpretation": "Conditional exact-version attribution on saved CPU-routed true Gemini; not an end-to-end family router or hidden-test score."}
    result.to_parquet(output / "oof.parquet", index=False)
    selected.to_parquet(output / "selected_oof.parquet", index=False)
    _save_class_reports(result, output, "")
    _save_class_reports(selected, output, "_selected")
    _json(output / "metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--baseline-oof", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    config = GeminiConfig(epochs=args.epochs, learning_rate=args.learning_rate,
                          max_length=args.max_length, batch_size=args.batch_size,
                          gradient_accumulation=args.gradient_accumulation,
                          warmup_ratio=args.warmup_ratio, seed=args.seed, device=args.device)
    metrics = run_experiment(read_train(args.train), pd.read_parquet(args.baseline_oof),
                             args.fold_manifest, args.model_path, args.output_dir, config)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
