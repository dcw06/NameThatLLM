"""Final QLoRA model and offline scoring for the current system blended with adapted Qwen.

The validated procedure is applied once to all 2,699 rows: two inner prompt folds choose the
learning rate, duration and the weight v of (1 - v) * current + v * adapted, exactly as inside each
outer training partition. Adapters and head are then trained on every row and saved.
Scoring loads the backbone once and reads frozen (adapters disabled) and adapted features
from it; the current system's fitted CPU and frozen-Qwen classifiers come from `ntl.ensemble fit`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from . import ensemble, lora, lora_compare, neural
from .constants import LABELS
from .data import fingerprint_frame, read_test, read_train, write_submission
from .folds import inner_prompt_splits

# The outer folds used seed offsets 0-4; the one full-data selection gets its own.
FINAL_SEED_OFFSET = 5


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def final_plan(train: pd.DataFrame, config: lora.AdapterConfig) -> pd.DataFrame:
    """Inner prompt folds over every official row, for the final setting and weight choice."""
    splits, _ = inner_prompt_splits(ensemble._targets(train), train["prompt_id"].to_numpy(),
                                    config.inner_folds, config.seed + FINAL_SEED_OFFSET)
    inner = np.full(len(train), -1)
    for inner_fold, (_, valid) in enumerate(splits):
        inner[valid] = inner_fold
    return train[["id", "prompt_id"]].assign(inner=inner)


def _aligned_plan(train: pd.DataFrame, plan: pd.DataFrame, config: lora.AdapterConfig) -> np.ndarray:
    plan = train[["id"]].merge(plan, on="id", how="left", validate="one_to_one")
    inner = plan["inner"].to_numpy()
    if plan["inner"].isna().any() or set(inner) != set(range(config.inner_folds)):
        raise ValueError("Final plan must assign every row to an inner fold")
    if plan.groupby("prompt_id")["inner"].nunique().max() != 1:
        raise ValueError("Final inner folds must keep prompts together")
    return inner.astype(int)


def run_inner(train: pd.DataFrame, plan: pd.DataFrame, experiment: lora.Experiment, inner_fold: int,
              output_dir: str | Path) -> Path:
    config = experiment.config
    inner = _aligned_plan(train, plan, config)
    fit_rows, eval_rows = np.flatnonzero(inner != inner_fold), np.flatnonzero(inner == inner_fold)
    probabilities, checkpoints = {}, []
    for rate in sorted(config.learning_rates):
        result = experiment.run(fit_rows, eval_rows, rate, config.trained_durations)
        probabilities[(0.0, 0.0)] = result["probabilities"][0.0]
        probabilities.update({(rate, duration): result["probabilities"][duration]
                              for duration in config.trained_durations})
        checkpoints.append({"learning_rate": rate, "checkpoints": result["checkpoints"]})
    frames = []
    for (rate, duration), values in probabilities.items():
        frame = pd.DataFrame(values, columns=lora_compare.PROBABILITIES)
        frame.insert(0, "duration", duration)
        frame.insert(0, "learning_rate", rate)
        frame.insert(0, "setting", lora.setting_name(rate, duration))
        frame.insert(0, "inner_fold", inner_fold)
        frame.insert(0, "id", train["id"].to_numpy()[eval_rows])
        frames.append(frame)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(output / f"inner{inner_fold}.parquet", index=False)
    _json(output / f"inner{inner_fold}.json", {"inner_fold": inner_fold, "fit_rows": len(fit_rows),
                                               "eval_rows": len(eval_rows), "runs": checkpoints})
    return output / f"inner{inner_fold}.parquet"


def choose(train: pd.DataFrame, features: np.ndarray, plan: pd.DataFrame, inner: pd.DataFrame,
           config: lora.AdapterConfig, blend: ensemble.BlendConfig | None = None) -> dict:
    """(Learning rate, duration), then blend weight, from the full-data inner folds only."""
    blend = blend or ensemble.BlendConfig()
    folds = _aligned_plan(train, plan, config)
    truth = ensemble._targets(train)
    position = pd.Series(np.arange(len(train)), index=train["id"])
    settings = {lora.setting_name(*setting): setting for setting in config.settings()}
    scores = {name: [] for name in settings}
    for inner_fold in range(config.inner_folds):
        for name in settings:
            rows = inner[(inner["inner_fold"] == inner_fold) & (inner["setting"] == name)]
            if set(position.loc[rows["id"]]) != set(np.flatnonzero(folds == inner_fold)):
                raise ValueError("Final inner predictions must cover their inner fold exactly")
            predictions = rows[lora_compare.PROBABILITIES].to_numpy().argmax(axis=1)
            scores[name].append(float(balanced_accuracy_score(truth[position.loc[rows["id"]]], predictions)))
    means = {name: float(np.mean(values)) for name, values in scores.items()}
    # Ties keep the shorter duration (zero is the frozen head), then the smaller learning rate.
    chosen = min(settings, key=lambda name: (-means[name], settings[name][1], settings[name][0]))
    rate, duration = settings[chosen]
    current = {}
    for inner_fold in range(config.inner_folds):
        fit, valid = np.flatnonzero(folds != inner_fold), np.flatnonzero(folds == inner_fold)
        probabilities, weight = lora_compare.current_on_split(train, features, fit, valid, blend,
                                                              100 + inner_fold)
        current[inner_fold] = (valid, probabilities, weight)
    weight = lora_compare.choose_weight(train, current, inner.assign(outer_fold=-1), -1, chosen)
    return {"schema_version": 1, "kind": "current-plus-adapted-qwen-recipe", "labels": list(LABELS),
            "setting": chosen, "learning_rate": rate, "duration": duration,
            "adapted_weight": weight["weight"], "current_weight": 0.25,
            "inner_mean_balanced_accuracy_by_setting": means,
            "inner_balanced_accuracy_by_weight": weight["inner_balanced_accuracy"],
            "current_inner_weights": [item[2] for item in current.values()],
            "train_fingerprint": fingerprint_frame(train), "adapter": asdict(config),
            "model_id": neural.MODEL_ID, "model_revision": neural.MODEL_REVISION}


def fit_all(train: pd.DataFrame, experiment: lora.Experiment, recipe: dict, output_dir: str | Path) -> dict:
    """Train adapters and head on every row for the chosen setting; save both."""
    if recipe["train_fingerprint"] != fingerprint_frame(train):
        raise ValueError("Training data differ from the rows used to choose the recipe")
    output = Path(output_dir)
    if (output / "head.npz").exists():
        raise ValueError("Use a new output directory; a fitted adapter must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = np.arange(len(train))
    duration = recipe["duration"]
    result = experiment.run(rows, rows[:0], recipe["learning_rate"], [duration] if duration > 0 else [])
    head = result["head"]
    np.savez(output / "head.npz", mean=head.mean.cpu().numpy(), scale=head.scale.cpu().numpy(),
             weight=head.linear.weight.detach().cpu().numpy(), bias=head.linear.bias.detach().cpu().numpy())
    if experiment.adapters:
        experiment.model.save_pretrained(output / "adapter")
    _json(output / "recipe.json", recipe)
    log = {"checkpoints": result["checkpoints"], "rows": len(train), "elapsed_seconds": time.perf_counter() - started}
    _json(output / "fit.json", log)
    return log


def load_adapter(base, adapter_dir: Path, device):
    """Wrap the backbone as training did and load the saved adapter weights onto its device.

    PeftModel.from_pretrained re-dispatches CPU-mapped backbones to the default accelerator.
    """
    from peft import PeftConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    model = get_peft_model(base, PeftConfig.from_pretrained(str(adapter_dir)))
    weights = load_file(str(adapter_dir / "adapter_model.safetensors"), device=str(device))
    result = set_peft_model_state_dict(model, weights)
    loaded = {name for name, _ in model.named_parameters() if "lora_" in name}
    if result.unexpected_keys or len(loaded) != len(weights):
        raise ValueError("Saved adapter weights do not match the adapter structure")
    return model


def predict(final_dir: str | Path, blend_dir: str | Path, test: pd.DataFrame, model_path: str | Path,
            output: str | Path, device: str = "auto") -> pd.DataFrame:
    """Score every current test row: current system (frozen features) blended with adapted Qwen."""
    import torch

    final_dir = Path(final_dir)
    recipe = json.loads((final_dir / "recipe.json").read_text())
    if recipe.get("kind") != "current-plus-adapted-qwen-recipe" or recipe.get("labels") != list(LABELS):
        raise ValueError("Unsupported adapted-Qwen recipe")
    # joblib is trusted project output written by `ntl.ensemble fit` in this same run.
    current = joblib.load(Path(blend_dir) / "blend.joblib")
    weight, _, frozen = ensemble.load_recipe(current["recipe"])
    if weight != recipe["current_weight"]:
        raise ValueError("Current-system weight differs from the adapted recipe")
    frozen = neural.FrozenConfig(**{**asdict(frozen), "device": device})
    started = time.perf_counter()
    tokenizer, base, device, _, _ = neural.load_backbone(model_path, frozen)
    model = load_adapter(base, final_dir / "adapter", device).eval()
    stored = np.load(final_dir / "head.npz")
    head = lora.head_module(torch, {key: stored[key] for key in ("mean", "scale", "weight", "bias")}, device).eval()
    texts = test["text"].tolist()

    def pooled(batch):
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=frozen.max_length,
                            return_tensors="pt").to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        return neural.masked_layer_mean(output.hidden_states, encoded["attention_mask"], frozen.pool_layers)

    with torch.inference_mode():
        # Frozen features exactly as validated: adapters off, batch size 2, file order.
        with model.disable_adapter():
            frozen_features = np.concatenate([pooled(texts[start:start + frozen.batch_size]).cpu().numpy()
                                              for start in range(0, len(texts), frozen.batch_size)])
        # Adapted features as in the adapter validation: length-sorted batches of eval_batch.
        lengths = np.asarray(tokenizer(texts, truncation=True, max_length=frozen.max_length,
                                       return_length=True)["length"])
        order, size = np.argsort(lengths, kind="stable"), recipe["adapter"]["eval_batch"]
        adapted = np.empty((len(texts), len(LABELS)))
        for start in range(0, len(texts), size):
            positions = order[start:start + size]
            logits = head(pooled([texts[index] for index in positions]))
            adapted[positions] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    with threadpool_limits(limits=2):
        cpu = ensemble.aligned_probabilities(current["cpu"], test["text"])
        qwen = neural._probabilities(current["qwen_head"], frozen_features)
    current_scores = ensemble._blend(cpu, qwen, weight)
    final = ensemble._blend(current_scores, adapted, recipe["adapted_weight"])
    if not np.isfinite(final).all() or not np.allclose(final.sum(axis=1), 1.0):
        raise ValueError("Final scores must be finite probability vectors")
    submission = write_submission(test["id"], np.asarray(LABELS)[final.argmax(axis=1)].tolist(), output)
    output = Path(output)
    _json(output.with_name(f"{output.stem}_profile.json"),
          {"rows": len(test), "adapted_weight": recipe["adapted_weight"], "setting": recipe["setting"],
           "elapsed_seconds": time.perf_counter() - started})
    return submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_command = commands.add_parser("plan")
    plan_command.add_argument("--train", type=Path, required=True)
    plan_command.add_argument("--output", type=Path, required=True)
    for name in ("inner", "choose", "fit"):
        sub = commands.add_parser(name)
        sub.add_argument("--train", type=Path, required=True)
        sub.add_argument("--plan", type=Path, required=True)
        sub.add_argument("--features", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        if name != "choose":
            sub.add_argument("--model-path", type=Path, required=True)
            sub.add_argument("--device", default="cuda:0")
    commands.choices["inner"].add_argument("--inner-fold", type=int, required=True)
    commands.choices["fit"].add_argument("--recipe", type=Path, required=True)
    predict_command = commands.add_parser("predict")
    for argument in ("--final-dir", "--blend-dir", "--test", "--model-path", "--output"):
        predict_command.add_argument(argument, type=Path, required=True)
    predict_command.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.command == "predict":
        submission = predict(args.final_dir, args.blend_dir, read_test(args.test), args.model_path, args.output,
                             args.device)
        print(json.dumps({"submission": str(args.output), "rows": len(submission)}))
        return 0
    train = read_train(args.train)
    config = lora.AdapterConfig()
    if args.command == "plan":
        final_plan(train, config).to_csv(args.output, index=False)
        return 0
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    cached, _ = ensemble.load_archive(args.features, train, neural.FrozenConfig())
    if args.command == "choose":
        inner = pd.concat([pd.read_parquet(path) for path in sorted(args.output_dir.glob("inner*.parquet"))])
        recipe = choose(train, cached, plan, inner, config)
        _json(args.output_dir / "recipe.json", recipe)
        print(json.dumps({key: recipe[key] for key in ("setting", "adapted_weight", "inner_mean_balanced_accuracy_by_setting")}))
        return 0
    experiment = lora.Experiment(train, cached, adapters=True, model_path=args.model_path, device=args.device)
    if args.command == "inner":
        run_inner(train, plan, experiment, args.inner_fold, args.output_dir)
        return 0
    log = fit_all(train, experiment, json.loads(args.recipe.read_text()), args.output_dir / "final")
    print(json.dumps({"rows": log["rows"], "elapsed_seconds": log["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
