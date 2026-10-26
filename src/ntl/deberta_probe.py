"""Instrumented small-subset fitting probe of the unchanged DeBERTa training loop.

It calls ``gemini._fit_predict_fold`` with one training-only subset as both the fitting
and the scored rows. Wrappers record pre-clipping gradient norms, per-update parameter
movement, learning rates and AMP loss scale without changing the update. Failing to
fit warrants investigation; fitting a subset does not validate the full-data recipe.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from . import gemini
from .data import read_train

GROUPS = ("embeddings", "encoder", "pooler", "classifier")


def parameter_group(name: str) -> str:
    for prefix, group in (("deberta.embeddings.", "embeddings"), ("deberta.encoder.", "encoder"),
                          ("pooler.", "pooler"), ("classifier.", "classifier")):
        if name.startswith(prefix):
            return group
    raise ValueError(f"Unexpected DeBERTa sequence-classification parameter: {name}")


def select_subset(frame: pd.DataFrame, per_class: int | None, outer_fold: int, seed: int) -> pd.DataFrame:
    """Sample equally per version from one outer fold's fitting rows; None keeps them all."""
    fitting = frame.loc[frame["fold"] != outer_fold]
    if per_class is None:
        return fitting.sort_values("id", kind="stable").reset_index(drop=True)
    if fitting.groupby("model").size().reindex(gemini.GEMINI_LABELS).min() < per_class:
        raise ValueError("Too few fitting rows for the requested per-class subset")
    subset = fitting.groupby("model", group_keys=False).sample(n=per_class, random_state=seed)
    return subset.sort_values("id", kind="stable").reset_index(drop=True)


class Recorder:
    """Observes the training loop through the objects it creates; never alters updates."""

    def __init__(self, torch):
        self.torch = torch
        self.model = None
        self.scaler = None
        self.initial = None
        self.previous = None
        self.pending = None
        self.clip_calls = 0
        self.updates = []

    def _norms(self, tensors) -> dict[str, float]:
        squares = dict.fromkeys(GROUPS, 0.0)
        for name, tensor in tensors:
            squares[parameter_group(name)] += float(tensor.detach().float().pow(2).sum())
        return {group: math.sqrt(value) for group, value in squares.items()}

    def before_clip(self) -> None:
        # Called after scaler.unscale_, so gradients are in true (unscaled) units.
        if self.initial is None:
            with self.torch.no_grad():
                self.initial = {name: p.detach().clone() for name, p in self.model.named_parameters()}
                self.previous = {name: value.clone() for name, value in self.initial.items()}
        named = list(self.model.named_parameters())
        self.clip_calls += 1
        self.pending = {
            "gradient_norm_by_group": self._norms((name, p.grad) for name, p in named if p.grad is not None),
            "parameters_without_gradient": [name for name, p in named if p.grad is None],
            "nonfinite_gradient_parameters": sum(
                int(not self.torch.isfinite(p.grad).all()) for _, p in named if p.grad is not None),
        }

    def after_step(self, optimizer) -> None:
        with self.torch.no_grad():
            named = list(self.model.named_parameters())
            change = self._norms((name, p - self.previous[name]) for name, p in named)
            size = self._norms(named)
            for name, p in named:
                self.previous[name].copy_(p)
        record = {"update": len(self.updates) + 1, "learning_rate": optimizer.param_groups[0]["lr"],
                  **(self.pending or {}),
                  "parameter_change_by_group": change,
                  "relative_change_by_group": {group: change[group] / size[group] if size[group] else 0.0
                                               for group in GROUPS},
                  "loss_scale": self.scaler.get_scale() if self.scaler is not None and self.scaler.is_enabled() else None}
        self.pending = None
        self.updates.append(record)

    def displacement(self) -> dict:
        with self.torch.no_grad():
            named = list(self.model.named_parameters())
            change = self._norms((name, p - self.initial[name]) for name, p in named)
            size = self._norms((name, self.initial[name]) for name, _ in named)
        return {"parameter_change_by_group": change,
                "relative_change_by_group": {group: change[group] / size[group] for group in GROUPS}}


@contextlib.contextmanager
def instrument():
    import torch
    import transformers

    recorder = Recorder(torch)
    load = transformers.AutoModelForSequenceClassification.from_pretrained
    clip = torch.nn.utils.clip_grad_norm_
    AdamW, GradScaler = torch.optim.AdamW, torch.amp.GradScaler

    def from_pretrained(*args, **kwargs):
        recorder.model = load(*args, **kwargs)
        return recorder.model

    def clip_grad_norm_(parameters, max_norm, *args, **kwargs):
        recorder.before_clip()
        total = clip(parameters, max_norm, *args, **kwargs)
        recorder.pending["gradient_norm"] = float(total)
        recorder.pending["clipped"] = bool(float(total) > max_norm)
        return total

    class ObservedAdamW(AdamW):
        def step(self, closure=None):
            result = super().step(closure)
            recorder.after_step(self)
            return result

    class ObservedGradScaler(GradScaler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            recorder.scaler = self

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(transformers.AutoModelForSequenceClassification,
                                              "from_pretrained", from_pretrained))
        stack.enter_context(mock.patch.object(torch.nn.utils, "clip_grad_norm_", clip_grad_norm_))
        stack.enter_context(mock.patch.object(torch.optim, "AdamW", ObservedAdamW))
        stack.enter_context(mock.patch.object(torch.amp, "GradScaler", ObservedGradScaler))
        yield recorder


def run_probe(
    train: pd.DataFrame, baseline_oof: pd.DataFrame, fold_manifest_path: str | Path, model_path: str | Path,
    output_dir: str | Path, config: gemini.GeminiConfig, per_class: int | None = 8, outer_fold: int = 0,
) -> dict:
    frame, protocol = gemini.prepare_gemini_evaluation(train, baseline_oof, fold_manifest_path,
                                                       gemini.OFFICIAL_BASELINE_COUNTS)
    gemini.validate_model_snapshot(model_path)
    output = Path(output_dir)
    if (output / "probe.json").exists():
        raise ValueError("Use a new output directory; an existing probe must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    subset = select_subset(frame, per_class, outer_fold, config.seed)
    with instrument() as recorder:
        probabilities, log = gemini._fit_predict_fold(subset, subset, model_path, config, outer_fold)
        displacement = recorder.displacement()
    targets = subset["target"].to_numpy()
    predictions = probabilities.argmax(axis=1)
    gradient_norms = np.asarray([update["gradient_norm"] for update in recorder.updates])
    result = {
        "schema_version": 1, "kind": "deberta-small-subset-fitting-probe",
        "configuration": asdict(config), "per_class": per_class, "outer_fold": outer_fold,
        "subset_policy": ("Equal per-version sample" if per_class else "Every row")
                         + " of the outer fitting partition; fitted and scored on the same rows.",
        "subset_ids": subset["id"].tolist(), "subset_prompts": int(subset["prompt_id"].nunique()),
        "baseline_oof_fingerprint": protocol["baseline_oof_fingerprint"],
        "subset_fit_after_training": {
            "accuracy": float((predictions == targets).mean()),
            "cross_entropy": float(-np.log(probabilities[np.arange(len(targets)), targets]).mean()),
            "uniform_cross_entropy": math.log(len(gemini.GEMINI_LABELS)),
            "prediction_counts": {label: int((predictions == index).sum())
                                  for index, label in enumerate(gemini.GEMINI_LABELS)},
            "mean_max_probability": float(probabilities.max(axis=1).mean()),
            "scoring_mode": "model.eval() after the final epoch, as in the outer validation pass",
        },
        "epoch_training_loss": [epoch["mean_training_loss"] for epoch in log["epochs"]],
        "clip_calls": recorder.clip_calls, "optimizer_updates": len(recorder.updates),
        "gradient_norm_summary": {"first": float(gradient_norms[0]), "last": float(gradient_norms[-1]),
                                  "median": float(np.median(gradient_norms)),
                                  "fraction_clipped": float(np.mean([u["clipped"] for u in recorder.updates]))},
        "parameters_without_gradient": sorted({name for update in recorder.updates
                                               for name in update["parameters_without_gradient"]}),
        "total_displacement": displacement,
        "training_log": log,
        "updates": recorder.updates,
    }
    (output / "probe.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--baseline-oof", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=8, help="Rows per version; 0 uses the whole fitting fold")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    config = gemini.GeminiConfig(epochs=args.epochs, learning_rate=args.learning_rate, device=args.device)
    result = run_probe(read_train(args.train), pd.read_parquet(args.baseline_oof), args.fold_manifest,
                       args.model_path, args.output_dir, config, args.per_class or None, args.outer_fold)
    summary = {key: result[key] for key in ("subset_fit_after_training", "epoch_training_loss",
                                            "optimizer_updates", "gradient_norm_summary",
                                            "parameters_without_gradient", "total_displacement")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
