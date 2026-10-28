"""Compare QLoRA-adapted Qwen (and its adapters-disabled control) with the validated system.

The current CPU-Qwen system's outer scores come from its nested validation. On each outer
fold's inner folds (those that chose the adapter setting) the current system is refit with
its own nested weight selection, so the weight v of (1 - v) * current + v * adapted is
chosen from inner rows only; v = 0 keeps the current system. Every system scores all 2,699
rows and is compared with the current system by a paired prompt bootstrap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from . import ensemble, neural
from .constants import LABELS
from .data import read_train
from .lora import AdapterConfig, validate_plan
from .selection import first_best

PROBABILITIES = [f"probability::{label}" for label in LABELS]
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_runs(directory: str | Path, train: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[dict]]:
    """Outer probabilities aligned to train rows, inner probabilities, and per-fold logs."""
    directory = Path(directory)
    outer = pd.concat([pd.read_parquet(directory / f"fold{fold}" / "outer.parquet") for fold in range(5)])
    inner = pd.concat([pd.read_parquet(directory / f"fold{fold}" / "inner.parquet").assign(outer_fold=fold)
                       for fold in range(5)])
    logs = [json.loads((directory / f"fold{fold}" / "log.json").read_text()) for fold in range(5)]
    if len(outer) != len(train) or set(outer["id"]) != set(train["id"]) or outer["id"].duplicated().any():
        raise ValueError(f"{directory} must score every official row exactly once")
    aligned = outer.set_index("id").loc[train["id"], PROBABILITIES].to_numpy()
    return aligned, inner, logs


def current_on_split(train: pd.DataFrame, features: np.ndarray, fit: np.ndarray, valid: np.ndarray,
                     config: ensemble.BlendConfig, seed_offset: int) -> tuple[np.ndarray, float]:
    """The current system refit on `fit` rows, with its own nested weight choice, scoring `valid`."""
    texts, labels = train["text"].to_numpy(dtype=object), train["model"].to_numpy(dtype=object)
    truth, prompts = ensemble._targets(train), train["prompt_id"].to_numpy()
    weight, _ = ensemble._select_weight(texts[fit], labels[fit], features[fit], truth[fit], prompts[fit],
                                        config, seed_offset)
    cpu = ensemble._cpu_probabilities(texts[fit], labels[fit], texts[valid], config.cpu)
    qwen = ensemble._qwen_probabilities(features[fit], truth[fit], features[valid], config.qwen)
    return ensemble._blend(cpu, qwen, weight), weight


def current_inner(train: pd.DataFrame, features: np.ndarray, plan: pd.DataFrame, outer_fold: int,
                  config: ensemble.BlendConfig) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    """Refit the current system on each inner training split of one outer fold."""
    column, outer = plan[f"inner_{outer_fold}"].to_numpy(), plan["outer_fold"].to_numpy()
    result = {}
    for inner_fold in sorted(set(column[outer != outer_fold])):
        fit, valid = np.flatnonzero((outer != outer_fold) & (column != inner_fold)), np.flatnonzero(column == inner_fold)
        probabilities, weight = current_on_split(train, features, fit, valid, config,
                                                 10 * (outer_fold + 1) + inner_fold)
        result[inner_fold] = (valid, probabilities, weight)
    return result


def choose_weight(train: pd.DataFrame, current: dict, inner: pd.DataFrame, outer_fold: int, setting: str) -> dict:
    truth = ensemble._targets(train)
    position = pd.Series(np.arange(len(train)), index=train["id"])
    scores = {weight: [] for weight in WEIGHTS}
    for inner_fold, (valid, current_probabilities, _) in current.items():
        rows = inner[(inner["outer_fold"] == outer_fold) & (inner["inner_fold"] == inner_fold)
                     & (inner["setting"] == setting)]
        row_positions = position.loc[rows["id"]].to_numpy()
        if len(row_positions) != len(valid) or set(row_positions) != set(valid):
            raise ValueError("Inner adapted predictions must cover the inner validation rows exactly")
        index = {row: offset for offset, row in enumerate(valid)}
        adapted = np.zeros_like(current_probabilities)
        adapted[[index[value] for value in row_positions]] = rows[PROBABILITIES].to_numpy()
        for weight in WEIGHTS:
            blended = ensemble._blend(current_probabilities, adapted, weight)
            scores[weight].append(float(balanced_accuracy_score(truth[valid], blended.argmax(axis=1))))
    means = {weight: float(np.mean(values)) for weight, values in scores.items()}
    # Ties keep the smaller adapted weight, i.e. stay with the validated system.
    return {"weight": first_best(WEIGHTS, means.__getitem__),
            "inner_balanced_accuracy": {str(weight): values for weight, values in scores.items()}}


def compare(train: pd.DataFrame, features: np.ndarray, plan: pd.DataFrame, current_oof: pd.DataFrame,
            runs: dict[str, str | Path], output_dir: str | Path, config: ensemble.BlendConfig | None = None,
            samples: int = 10000) -> dict:
    config = config or ensemble.BlendConfig()
    plan = validate_plan(train, plan, AdapterConfig())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    truth, prompts, folds = ensemble._targets(train), train["prompt_id"].to_numpy(), plan["outer_fold"].to_numpy()
    current_frame = current_oof.set_index("id").loc[train["id"]]
    if (current_frame["true_model"].to_numpy() != train["model"].to_numpy()).any():
        raise ValueError("Current-system OOF labels differ from the official rows")
    if (current_frame["fold"].to_numpy() != folds).any():
        raise ValueError("Current-system OOF must come from the plan's outer folds")
    current = current_frame[[f"blend::{label}" for label in LABELS]].to_numpy()
    current_inner_cache = {fold: current_inner(train, features, plan, fold, config) for fold in range(5)}
    probabilities, selection = {"current": current}, {}
    for name, directory in runs.items():
        adapted, inner, logs = load_runs(directory, train)
        probabilities[name] = adapted
        blended, chosen = np.empty_like(current), []
        for fold in range(5):
            choice = choose_weight(train, current_inner_cache[fold], inner, fold, logs[fold]["selected_setting"])
            mask = folds == fold
            blended[mask] = ensemble._blend(current[mask], adapted[mask], choice["weight"])
            chosen.append({"outer_fold": fold, "selected_setting": logs[fold]["selected_setting"],
                           "current_inner_weight": [item[2] for item in current_inner_cache[fold].values()],
                           **choice})
        probabilities[f"{name}+current"] = blended
        selection[name] = chosen
    predictions = {name: values.argmax(axis=1) for name, values in probabilities.items()}
    systems = {name: ensemble._system_metrics(truth, values, folds) for name, values in predictions.items()}
    paired = {f"{name}_vs_current": ensemble.prompt_bootstrap(truth, predictions["current"], values, prompts,
                                                               samples, config.seed)
              for name, values in predictions.items() if name != "current"}
    if {"lora", "control"} <= set(predictions):
        paired["lora_vs_control"] = ensemble.prompt_bootstrap(truth, predictions["control"], predictions["lora"],
                                                              prompts, samples, config.seed)
    frame = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"}).assign(outer_fold=folds)
    for name, values in predictions.items():
        frame[f"prediction::{name}"] = np.asarray(LABELS)[values]
    frame.to_parquet(output / "predictions.parquet", index=False)
    result = {"systems": systems, "paired": paired, "selection": selection,
              "reference_current_balanced_accuracy": systems["current"]["balanced_accuracy"]}
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--current-oof", type=Path, required=True, help="blend-cv oof.parquet")
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "DIRECTORY"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    train = read_train(args.train)
    features, _ = ensemble.load_archive(args.features, train, neural.FrozenConfig())
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    result = compare(train, features, plan, pd.read_parquet(args.current_oof), dict(args.run), args.output_dir)
    print(json.dumps({"balanced_accuracy": {name: values["balanced_accuracy"]
                                            for name, values in result["systems"].items()},
                      "weights": {name: [item["weight"] for item in items] for name, items in result["selection"].items()},
                      "paired": {name: [round(values["balanced_accuracy_delta"], 4)] + [
                          round(bound, 4) for bound in values["delta_95_interval"]]
                                 for name, values in result["paired"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
