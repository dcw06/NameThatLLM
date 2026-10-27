"""QLoRA adaptation of the frozen Qwen3-8B-Base recipe for 18-class attribution.

The pinned NF4 backbone, tokenizer, 512-token limit, 24 retained blocks and masked mean
pooling of blocks 7 and 22 are unchanged. The head is the frozen pipeline itself: the
training partition's standardization and a linear layer initialised from the C=0.005
logistic fit, trained against that same L2-penalised objective. Rank-16 adapters then train
jointly with it (linear probe, then fine-tune); with adapters disabled the identical loop is
the control. The number of epochs, including zero, is chosen by inner prompt folds of each
outer training partition, so outer validation rows never select anything.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from . import ensemble, neural
from .constants import LABELS
from .data import fingerprint_frame, read_train
from .folds import inner_prompt_splits, load_fold_manifest


@dataclass(frozen=True)
class AdapterConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    # Blocks after the last pooled block cannot change the pooled features.
    adapted_layers: int = 22
    # Gentler than the first run (1e-4, whole epochs), which overfit within its first epoch.
    learning_rates: tuple[float, ...] = (1e-5, 3e-5)
    # Training durations in epochs; 0 is the frozen head itself, for every learning rate.
    durations: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
    warmup_steps: int = 10
    micro_batch: int = 4
    accumulation: int = 4
    max_grad_norm: float = 1.0
    eval_batch: int = 8
    inner_folds: int = 2
    seed: int = 2026

    def validate(self) -> None:
        if min(self.rank, self.alpha, self.micro_batch, self.accumulation, self.eval_batch, self.warmup_steps) < 1:
            raise ValueError("Adapter sizes, batches and warmup must be positive")
        if 0 not in self.durations or min(self.durations) < 0 or len(set(self.durations)) != len(self.durations):
            raise ValueError("Durations must be distinct, nonnegative and include 0 (the frozen head)")
        if not self.learning_rates or min(self.learning_rates) <= 0 or not 0 <= self.dropout < 1 or self.inner_folds < 2:
            raise ValueError("Invalid learning rates, dropout or inner fold count")

    def settings(self) -> list[tuple[float, float]]:
        """(learning rate, duration) candidates; the frozen head is one candidate, not one per rate."""
        return [(0.0, 0.0)] + [(rate, duration) for rate in sorted(self.learning_rates)
                               for duration in sorted(self.durations) if duration > 0]

    @property
    def trained_durations(self) -> list[float]:
        return sorted(duration for duration in self.durations if duration > 0)


def setting_name(rate: float, duration: float) -> str:
    return "frozen" if duration == 0 else f"lr{rate:g}-d{duration:g}"


def _json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def split_plan(train: pd.DataFrame, manifest_path: str | Path, config: AdapterConfig) -> pd.DataFrame:
    """Original outer prompt folds plus the inner prompt folds of every outer training partition."""
    manifest, metadata = load_fold_manifest(train, manifest_path)
    if metadata["n_splits"] != 5:
        raise ValueError("The adaptation experiment uses the original five prompt folds")
    plan = train[["id", "prompt_id"]].merge(manifest[["id", "fold"]], on="id", validate="one_to_one")
    plan = plan.rename(columns={"fold": "outer_fold"})
    targets, prompts = ensemble._targets(train), plan["prompt_id"].to_numpy()
    for fold in range(5):
        fit = np.flatnonzero(plan["outer_fold"].to_numpy() != fold)
        splits, _ = inner_prompt_splits(targets[fit], prompts[fit], config.inner_folds, config.seed + fold)
        inner = np.full(len(plan), -1)
        for inner_fold, (_, valid) in enumerate(splits):
            inner[fit[valid]] = inner_fold
        plan[f"inner_{fold}"] = inner
    return plan


def validate_plan(train: pd.DataFrame, plan: pd.DataFrame, config: AdapterConfig) -> pd.DataFrame:
    plan = train[["id"]].merge(plan, on="id", how="left", validate="one_to_one")
    if plan.isna().any().any() or (plan["prompt_id"] != train["prompt_id"]).any():
        raise ValueError("Split plan must cover every training row with its own prompt")
    if set(plan["outer_fold"]) != set(range(5)):
        raise ValueError("Outer folds must be exactly the five folds 0-4")
    if plan.groupby("prompt_id")["outer_fold"].nunique().max() != 1:
        raise ValueError("Outer folds must keep prompts together")
    for fold in range(5):
        column = plan[f"inner_{fold}"].to_numpy()
        outer = plan["outer_fold"].to_numpy() == fold
        if (column[outer] != -1).any() or set(column[~outer]) != set(range(config.inner_folds)):
            raise ValueError("Inner folds must partition exactly the outer training rows")
        if plan.loc[~outer].groupby("prompt_id")[f"inner_{fold}"].nunique().max() != 1:
            raise ValueError("Inner folds must keep prompts together")
    return plan


def fit_probe(features: np.ndarray, targets: np.ndarray, frozen: neural.FrozenConfig) -> dict:
    """The frozen recipe's head, fit on a training partition: the head's starting point."""
    head = neural._head(frozen)
    with threadpool_limits(limits=2):
        head.fit(features, targets)
    scaler, classifier = head.steps[0][1], head.steps[-1][1]
    if not np.array_equal(classifier.classes_, np.arange(len(LABELS))):
        raise ValueError("Probe head must contain every canonical label")
    return {"mean": scaler.mean_, "scale": scaler.scale_, "weight": classifier.coef_,
            "bias": classifier.intercept_, "pipeline": head}


def head_module(torch, probe: dict, device):
    class StandardizedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.as_tensor(probe["mean"], dtype=torch.float32))
            self.register_buffer("scale", torch.as_tensor(probe["scale"], dtype=torch.float32))
            self.linear = torch.nn.Linear(probe["weight"].shape[1], probe["weight"].shape[0])
            with torch.no_grad():
                self.linear.weight.copy_(torch.as_tensor(probe["weight"], dtype=torch.float32))
                self.linear.bias.copy_(torch.as_tensor(probe["bias"], dtype=torch.float32))

        def forward(self, pooled):
            return self.linear((pooled.float() - self.mean) / self.scale)

    return StandardizedLinear().to(device)


def _batches(lengths: np.ndarray, size: int, seed: int) -> list[np.ndarray]:
    """Shuffle, sort within chunks of 16 batches by length (less padding), shuffle batches."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lengths))
    batches = []
    for start in range(0, len(order), size * 16):
        chunk = order[start:start + size * 16]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        batches += [chunk[index:index + size] for index in range(0, len(chunk), size)]
    return [batches[index] for index in rng.permutation(len(batches))]


class Experiment:
    """One process: the backbone (if adapters are on) is loaded once and reset between runs."""

    def __init__(self, train: pd.DataFrame, cached: np.ndarray, *, adapters: bool, model_path=None,
                 frozen: neural.FrozenConfig | None = None, config: AdapterConfig | None = None,
                 device: str = "cpu"):
        import torch

        self.torch, self.adapters = torch, adapters
        self.frozen = replace(frozen or neural.FrozenConfig(), device=device)
        self.config = config or AdapterConfig()
        self.config.validate()
        self.texts = train["text"].tolist()
        self.targets = ensemble._targets(train)
        self.cached = np.asarray(cached, dtype=np.float32)
        if self.cached.shape[0] != len(train):
            raise ValueError("Cached frozen features must align with the training rows")
        self.device = device if not adapters else None
        if model_path is None:
            raise ValueError("model_path is required: the control also needs the backbone's tokenizer")
        if adapters:
            from peft import LoraConfig, get_peft_model

            tokenizer, model, device, _, self.manifest = neural.load_backbone(model_path, self.frozen)
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            # The bare Qwen3Model names blocks "layers.N" with no prefix, which peft's
            # layers_to_transform cannot match; a full-name pattern selects the same modules.
            layers = "|".join(str(index) for index in range(self.config.adapted_layers))
            modules = "|".join(self.config.target_modules)
            self.model = get_peft_model(model, LoraConfig(
                r=self.config.rank, lora_alpha=self.config.alpha, lora_dropout=self.config.dropout, bias="none",
                target_modules=rf"(?:.*\.)?layers\.(?:{layers})\.(?:self_attn|mlp)\.(?:{modules})"))
            self.tokenizer, self.device = tokenizer, device
            # Every run restarts from the same adapter initialisation (B = 0: the frozen backbone).
            self.initial = {name: value.detach().clone() for name, value in self.model.named_parameters()
                            if value.requires_grad}
        else:
            self.manifest = neural.validate_snapshot(model_path, weights=False)
            tokenizer = neural.load_tokenizer(model_path)
        # Batches are bucketed by token length, so the control needs the same lengths to train
        # on exactly the adapted model's batches.
        self.lengths = np.asarray(tokenizer(self.texts, truncation=True, max_length=self.frozen.max_length,
                                            return_length=True)["length"])
        self.cuda = str(self.device).startswith("cuda")

    def reset(self) -> None:
        """Restore the adapter initialisation (B = 0: the frozen backbone); no-op for the control."""
        if self.adapters:
            with self.torch.no_grad():
                for name, value in self.model.named_parameters():
                    if name in self.initial:
                        value.copy_(self.initial[name])

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.adapters else 0

    def _pooled(self, rows: np.ndarray):
        if not self.adapters:
            return self.torch.as_tensor(self.cached[rows], device=self.device)
        encoded = self.tokenizer([self.texts[index] for index in rows], padding=True, truncation=True,
                                 max_length=self.frozen.max_length, return_tensors="pt").to(self.device)
        output = self.model(**encoded, output_hidden_states=True, use_cache=False)
        return neural.masked_layer_mean(output.hidden_states, encoded["attention_mask"], self.frozen.pool_layers)

    def predict(self, head, rows: np.ndarray) -> np.ndarray:
        torch = self.torch
        probabilities = np.empty((len(rows), len(LABELS)))
        order = np.argsort(self.lengths[rows], kind="stable")
        if self.adapters:
            self.model.eval()
        head.eval()
        with torch.inference_mode():
            for start in range(0, len(rows), self.config.eval_batch):
                positions = order[start:start + self.config.eval_batch]
                logits = head(self._pooled(rows[positions]))
                probabilities[positions] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite adapted probabilities")
        return probabilities

    def run(self, fit_rows: np.ndarray, eval_rows: np.ndarray, learning_rate: float,
            checkpoints: list[float]) -> dict:
        """Probe head on the fit rows, then train adapters and head at one learning rate.

        Returns eval probabilities at duration 0 (the frozen head) and at every checkpoint, in
        epochs. The learning rate is constant after warmup, so the checkpoint at d epochs is the
        model trained for exactly d epochs.
        """
        torch = self.torch
        probe = fit_probe(self.cached[fit_rows], self.targets[fit_rows], self.frozen)
        head = head_module(torch, probe, self.device)
        result = {"probabilities": {}, "checkpoints": [], "head": head, "probe": probe}
        if len(eval_rows):
            result["probabilities"][0.0] = probe["pipeline"].predict_proba(self.cached[eval_rows])
        # Reset even when nothing trains, so a zero-duration run never keeps an earlier run's adapters.
        self.reset()
        if not checkpoints:
            return result
        torch.manual_seed(self.config.seed)
        if self.adapters:
            self.model.train()
        head.train()
        parameters = list(head.parameters())
        if self.adapters:
            parameters += [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(1.0, (step + 1) / self.config.warmup_steps))
        scaler = torch.amp.GradScaler("cuda", enabled=self.cuda)
        # Mean cross-entropy + ||W||^2 / (2 C n): the frozen logistic objective per row.
        l2 = 1.0 / (self.frozen.C * len(fit_rows))
        groups_per_epoch = -(-len(_batches(self.lengths[fit_rows], self.config.micro_batch, 0))
                             // self.config.accumulation)
        targets = {}
        for duration in sorted(checkpoints):
            targets.setdefault(max(1, round(duration * groups_per_epoch)), []).append(duration)
        last = max(targets)
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        step, epoch, started, objective, since, skipped = 0, 0, time.perf_counter(), 0.0, 0, 0
        while step < last:
            epoch += 1
            batches = _batches(self.lengths[fit_rows], self.config.micro_batch, self.config.seed + epoch)
            for first in range(0, len(batches), self.config.accumulation):
                members = batches[first:first + self.config.accumulation]
                rows_in_group = sum(len(batch) for batch in members)
                for position, batch in enumerate(members):
                    rows = fit_rows[batch]
                    logits = head(self._pooled(rows))
                    labels = torch.as_tensor(self.targets[rows], device=self.device)
                    loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="sum") / rows_in_group
                    if position == 0:
                        loss = loss + 0.5 * l2 * head.linear.weight.pow(2).sum()
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite adaptation loss at step {step + 1}")
                    objective += float(loss.detach())
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < scale)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step, since = step + 1, since + 1
                if step in targets:
                    record = {"durations": targets[step], "optimizer_steps": step, "learning_rate": learning_rate,
                              "mean_objective": objective / since, "skipped_amp_steps": skipped,
                              "train_seconds": time.perf_counter() - started, "train_rows": len(fit_rows)}
                    if len(eval_rows):
                        evaluated = time.perf_counter()
                        probabilities = self.predict(head, eval_rows)
                        for duration in targets[step]:
                            result["probabilities"][duration] = probabilities
                        record["eval_seconds"] = time.perf_counter() - evaluated
                        if self.adapters:
                            self.model.train()
                        head.train()
                    if self.cuda:
                        record["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
                    result["checkpoints"].append(record)
                    print(json.dumps(record), flush=True)
                    started, objective, since = time.perf_counter(), 0.0, 0
                if step >= last:
                    break
        return result


def run_fold(train: pd.DataFrame, plan: pd.DataFrame, experiment: Experiment, outer_fold: int,
             output_dir: str | Path) -> dict:
    """Inner folds choose the (learning rate, duration); then train once on the outer partition."""
    config = experiment.config
    plan = validate_plan(train, plan, config)
    output = Path(output_dir)
    if (output / "outer.parquet").exists():
        raise ValueError("Use a new output directory; an existing fold must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    outer = plan["outer_fold"].to_numpy()
    inner = plan[f"inner_{outer_fold}"].to_numpy()
    settings = config.settings()
    started = time.perf_counter()
    inner_rows, inner_logs, scores = [], [], {setting: [] for setting in settings}
    for inner_fold in range(config.inner_folds):
        fit_rows = np.flatnonzero((outer != outer_fold) & (inner != inner_fold))
        eval_rows = np.flatnonzero(inner == inner_fold)
        probabilities = {}
        for rate in sorted(config.learning_rates):
            result = experiment.run(fit_rows, eval_rows, rate, config.trained_durations)
            probabilities[(0.0, 0.0)] = result["probabilities"][0.0]
            probabilities.update({(rate, duration): result["probabilities"][duration]
                                  for duration in config.trained_durations})
            inner_logs.append({"inner_fold": inner_fold, "learning_rate": rate, "fit_rows": len(fit_rows),
                               "eval_rows": len(eval_rows), "checkpoints": result["checkpoints"]})
        for setting in settings:
            values = probabilities[setting]
            scores[setting].append(float(balanced_accuracy_score(experiment.targets[eval_rows],
                                                                 values.argmax(axis=1))))
            frame = pd.DataFrame(values, columns=[f"probability::{label}" for label in LABELS])
            frame.insert(0, "duration", setting[1])
            frame.insert(0, "learning_rate", setting[0])
            frame.insert(0, "setting", setting_name(*setting))
            frame.insert(0, "inner_fold", inner_fold)
            frame.insert(0, "id", train["id"].to_numpy()[eval_rows])
            inner_rows.append(frame)
    means = {setting: float(np.mean(values)) for setting, values in scores.items()}
    # Ties keep the shorter duration (zero is the frozen head), then the smaller learning rate.
    rate, duration = min(settings, key=lambda setting: (-means[setting], setting[1], setting[0]))
    fit_rows, eval_rows = np.flatnonzero(outer != outer_fold), np.flatnonzero(outer == outer_fold)
    result = experiment.run(fit_rows, eval_rows, rate, [duration] if duration > 0 else [])
    probabilities = result["probabilities"][duration]
    outer_frame = pd.DataFrame(probabilities, columns=[f"probability::{label}" for label in LABELS])
    outer_frame.insert(0, "duration", duration)
    outer_frame.insert(0, "learning_rate", rate)
    outer_frame.insert(0, "setting", setting_name(rate, duration))
    outer_frame.insert(0, "id", train["id"].to_numpy()[eval_rows])
    pd.concat(inner_rows, ignore_index=True).to_parquet(output / "inner.parquet", index=False)
    outer_frame.to_parquet(output / "outer.parquet", index=False)
    log = {"outer_fold": outer_fold, "adapters": experiment.adapters,
           "model_revision": experiment.manifest.get("revision"),
           "selected_setting": setting_name(rate, duration), "selected_learning_rate": rate,
           "selected_duration": duration,
           "inner_balanced_accuracy": {setting_name(*key): values for key, values in scores.items()},
           "inner_mean_balanced_accuracy": {setting_name(*key): value for key, value in means.items()},
           "outer_balanced_accuracy": float(balanced_accuracy_score(experiment.targets[eval_rows],
                                                                    probabilities.argmax(axis=1))),
           "inner_runs": inner_logs, "outer_run": result["checkpoints"],
           "elapsed_seconds": time.perf_counter() - started,
           "trainable_parameters": experiment.trainable_parameters(),
           "configuration": asdict(config), "frozen": asdict(experiment.frozen)}
    _json(output / "log.json", log)
    print(json.dumps({key: log[key] for key in ("outer_fold", "selected_setting", "inner_mean_balanced_accuracy",
                                                "outer_balanced_accuracy", "elapsed_seconds")}), flush=True)
    return log


def probe_training(train: pd.DataFrame, plan: pd.DataFrame, experiment: Experiment, output_dir: str | Path,
                   per_class: int = 4, epochs: int = 6) -> dict:
    """Small check: init features match the frozen archive; memory, speed, and fitting a subset."""
    torch = experiment.torch
    plan = validate_plan(train, plan, experiment.config)
    subset = (plan.loc[plan["outer_fold"] != 0].assign(model=train["model"].to_numpy()[plan["outer_fold"] != 0])
              .groupby("model", group_keys=False).sample(n=per_class, random_state=experiment.config.seed))
    rows = np.sort(train.index.get_indexer(subset.index))
    report = {"rows": len(rows), "trainable_parameters": experiment.trainable_parameters()}
    if experiment.adapters:
        experiment.reset()  # the initial-feature check must not see an earlier run's adapters
        experiment.model.eval()
        with torch.inference_mode():
            start = time.perf_counter()
            pooled = torch.cat([experiment._pooled(rows[index:index + 4]) for index in range(0, len(rows), 4)])
            report["init_feature_seconds_per_row"] = (time.perf_counter() - start) / len(rows)
        difference = np.abs(pooled.cpu().numpy() - experiment.cached[rows])
        report["init_vs_frozen_archive"] = {"max_abs": float(difference.max()),
                                            "mean_abs": float(difference.mean()),
                                            "relative": float(difference.mean() / np.abs(experiment.cached[rows]).mean())}
    rate = max(experiment.config.learning_rates)
    result = experiment.run(rows, rows, rate, [float(epoch) for epoch in range(1, epochs + 1)])
    truth = experiment.targets[rows]
    report["learning_rate"] = rate
    report["fit_by_epoch"] = [
        {"epoch": epoch, "accuracy": float((probabilities.argmax(axis=1) == truth).mean()),
         "cross_entropy": float(-np.log(probabilities[np.arange(len(rows)), truth] + 1e-12).mean())}
        for epoch, probabilities in sorted(result["probabilities"].items())]
    report["epochs"] = result["checkpoints"]
    train_seconds = sum(record["train_seconds"] for record in result["checkpoints"])
    report["train_seconds_per_row_epoch"] = train_seconds / (len(rows) * epochs)
    report["eval_seconds_per_row"] = float(np.mean([record["eval_seconds"] for record in result["checkpoints"]])) / len(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _json(output / ("probe.json" if experiment.adapters else "probe_control.json"), report)
    print(json.dumps({key: value for key, value in report.items() if key != "epochs"}, indent=2), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_command = commands.add_parser("plan", help="Write the outer/inner prompt-fold plan")
    plan_command.add_argument("--train", type=Path, required=True)
    plan_command.add_argument("--fold-manifest", type=Path, required=True)
    plan_command.add_argument("--output", type=Path, required=True)
    for name in ("probe", "fold"):
        sub = commands.add_parser(name)
        sub.add_argument("--train", type=Path, required=True)
        sub.add_argument("--plan", type=Path, required=True)
        sub.add_argument("--features", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--model-path", type=Path, required=True,
                         help="Pinned snapshot; the control reads only its tokenizer")
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--control", action="store_true", help="Adapters disabled; head only, on cached features")
    commands.choices["fold"].add_argument("--outer-fold", type=int, nargs="+", required=True)
    args = parser.parse_args(argv)
    train = read_train(args.train)
    if args.command == "plan":
        plan = split_plan(train, args.fold_manifest, AdapterConfig())
        plan.to_csv(args.output, index=False)
        print(json.dumps({"rows": len(plan), "train_fingerprint": fingerprint_frame(train)}))
        return 0
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    cached, _ = ensemble.load_archive(args.features, train, neural.FrozenConfig())
    experiment = Experiment(train, cached, adapters=not args.control, model_path=args.model_path,
                            device="cpu" if args.control else args.device)
    if args.command == "probe":
        probe_training(train, plan, experiment, args.output_dir)
        return 0
    for fold in args.outer_fold:
        run_fold(train, plan, experiment, fold, Path(args.output_dir) / f"fold{fold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
