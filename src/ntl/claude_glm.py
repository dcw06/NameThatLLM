"""Claude-GLM specialist: frozen Qwen features (control) vs Qwen + StyleDistance (candidate).

A seven-class head over the five Claude models and the two GLMs is trained on those rows
of each training partition only. It redistributes the current system's probability mass
on the seven: for those classes p' = (1 - s) p + s M q, where M is the current system's
mass on the seven and q the specialist's distribution; every other class keeps its
probability, and s = 0 is the current system exactly. Inner prompt folds of each outer
training partition choose s from {0, 0.25, 0.5, 0.75, 1}, scored on the full 18 classes.
Every passage of the original five prompt folds is scored and compared with the current
system by a paired prompt bootstrap.

Both heads are the frozen Qwen recipe's classifier (standardization, then C=0.005
logistic regression). The candidate appends StyleDistance, scaled after standardization
so that its block carries the same total variance as the Qwen block; the Qwen block is
treated exactly as in the control.

For the topic robustness check, `--groups` regroups the rows (whole prompts move
together), as `topic_cv` did: the current system's own inner weight choice, the
specialist's inner folds and the bootstrap then all hold out and resample whole topics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from . import ensemble, neural
from .constants import LABELS
from .data import fingerprint_frame, read_test, read_train, write_submission
from .encoders import ENCODERS, extract_embeddings
from .frozen_gemini import load_feature_archive, save_feature_archive
from .lora import AdapterConfig, validate_plan
from .lora_compare import current_inner, current_on_split
from .lora_final import _aligned_plan
from .modeling import aligned_probabilities

SPECIALIST = ("claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
              "glm-5.3", "glm-5.3-flash")
INDEX = np.array([LABELS.index(label) for label in SPECIALIST])
# Canonical label index -> specialist class index (-1 outside the specialist).
_POSITION = np.full(len(LABELS), -1)
_POSITION[INDEX] = np.arange(len(SPECIALIST))
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
FEATURE_SETS = {"qwen": ("qwen",), "qwen+styledistance": ("qwen", "styledistance")}
BLEND = [f"blend::{label}" for label in LABELS]
RECIPE_KIND = "claude-glm-specialist-recipe"
# The validated candidate; the control (Qwen only) gave no gain.
FINAL_FEATURE_SET = "qwen+styledistance"
# Final weight choice on the full-data inner folds, as lora_final seeds them.
FINAL_SEED_OFFSET = 100


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _column_weights(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return features * weights


def specialist_head(widths: tuple[int, ...], frozen: neural.FrozenConfig) -> Pipeline:
    """The frozen recipe's head; every block after the first is scaled to the first's total variance."""
    weights = np.concatenate([np.full(width, np.sqrt(widths[0] / width)) for width in widths])
    return Pipeline([
        ("scale", StandardScaler()),
        ("balance", FunctionTransformer(_column_weights, kw_args={"weights": weights})),
        ("logistic", LogisticRegression(C=frozen.C, max_iter=2000, random_state=frozen.seed)),
    ])


def specialist_probabilities(blocks: dict[str, np.ndarray], feature_set: str, truth: np.ndarray,
                             fit: np.ndarray, rows: np.ndarray, frozen: neural.FrozenConfig) -> np.ndarray:
    """Fit on the specialist-class rows among `fit`; seven-class probabilities for `rows`."""
    names = FEATURE_SETS[feature_set]
    features = np.hstack([blocks[name] for name in names])
    fit = fit[np.isin(truth[fit], INDEX)]
    targets = _POSITION[truth[fit]]
    if set(targets) != set(range(len(SPECIALIST))):
        raise ValueError("Every specialist class needs training rows")
    head = specialist_head(tuple(blocks[name].shape[1] for name in names), frozen)
    with threadpool_limits(limits=2):
        head.fit(features[fit], targets)
        probabilities = head.predict_proba(features[rows])
    if not np.array_equal(head.classes_, np.arange(len(SPECIALIST))):
        raise ValueError("Specialist head must contain every specialist class")
    return probabilities


def redistribute(current: np.ndarray, specialist: np.ndarray, weight: float) -> np.ndarray:
    """(1 - s) p + s M q on the specialist's classes; every other class keeps its probability."""
    result = np.array(current, dtype=float, copy=True)
    mass = result[:, INDEX].sum(axis=1, keepdims=True)
    result[:, INDEX] = (1 - weight) * result[:, INDEX] + weight * mass * specialist
    return result


def groups_sha256(train: pd.DataFrame) -> str:
    """The row grouping (prompt or topic) every split and the bootstrap use."""
    pairs = sorted(zip(train["id"].astype(str), train["prompt_id"].astype(str)))
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()


def regroup(train: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Rows regrouped, e.g. by topic, as topic_cv does; whole prompts move together."""
    mapping = groups.set_index("id")["prompt_id"]
    if mapping.index.duplicated().any() or set(mapping.index) != set(train["id"]):
        raise ValueError("Groups must cover every training row exactly once")
    group = train["id"].map(mapping).astype(str).to_numpy()
    if pd.Series(group).groupby(train["prompt_id"].to_numpy()).nunique().max() != 1:
        raise ValueError("Groups must keep each prompt's passages together")
    return train.assign(prompt_id=group)


def _aligned_current(current_oof: pd.DataFrame, train: pd.DataFrame, folds: np.ndarray) -> np.ndarray:
    frame = current_oof.set_index("id").loc[train["id"]]
    if (frame["true_model"].to_numpy() != train["model"].to_numpy()).any():
        raise ValueError("Current-system OOF labels differ from the official rows")
    if (frame["fold"].to_numpy() != folds).any():
        raise ValueError("Current-system OOF must come from the plan's outer folds")
    if (frame["prompt_id"].astype(str).to_numpy() != train["prompt_id"].astype(str).to_numpy()).any():
        raise ValueError("Current-system OOF must come from the same row grouping (prompts or topics)")
    return frame[BLEND].to_numpy()


def build_current_inner(train: pd.DataFrame, features: np.ndarray, plan: pd.DataFrame, outer_fold: int,
                        config: ensemble.BlendConfig | None = None) -> pd.DataFrame:
    """The current system refit, with its own nested weight, on each inner split of one outer fold."""
    config = config or ensemble.BlendConfig()
    plan = validate_plan(train, plan, AdapterConfig())
    frames = []
    for inner_fold, (valid, probabilities, weight) in current_inner(train, features, plan, outer_fold,
                                                                    config).items():
        frame = pd.DataFrame(probabilities, columns=BLEND)
        frame.insert(0, "groups_sha256", groups_sha256(train))
        frame.insert(0, "weight", weight)
        frame.insert(0, "inner_fold", inner_fold)
        frame.insert(0, "outer_fold", outer_fold)
        frame.insert(0, "id", train["id"].to_numpy()[valid])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _inner_current(cache: pd.DataFrame, train: pd.DataFrame, plan: pd.DataFrame, outer_fold: int) -> dict:
    """{inner fold: (row positions, probabilities)}, checked to cover exactly that inner fold."""
    position = pd.Series(np.arange(len(train)), index=train["id"])
    column = plan[f"inner_{outer_fold}"].to_numpy()
    result = {}
    for inner_fold in sorted(set(column[column >= 0])):
        rows = cache[(cache["outer_fold"] == outer_fold) & (cache["inner_fold"] == inner_fold)]
        positions = position.loc[rows["id"]].to_numpy()
        if len(positions) != (column == inner_fold).sum() or set(positions) != set(np.flatnonzero(column == inner_fold)):
            raise ValueError("Cached current-system inner predictions must cover each inner fold exactly")
        result[inner_fold] = (positions, rows[BLEND].to_numpy())
    return result


def evaluate(train: pd.DataFrame, blocks: dict[str, np.ndarray], plan: pd.DataFrame, current_oof: pd.DataFrame,
             current_cache: pd.DataFrame, output_dir: str | Path, *, frozen: neural.FrozenConfig | None = None,
             samples: int = 10000, seed: int = 2026, provenance: dict | None = None) -> dict:
    frozen = frozen or neural.FrozenConfig()
    plan = validate_plan(train, plan, AdapterConfig())
    output = Path(output_dir)
    if (output / "comparison.json").exists():
        raise ValueError("Use a new output directory; an existing comparison must not be overwritten")
    if set(current_cache["groups_sha256"]) != {groups_sha256(train)}:
        raise ValueError("Cached current-system inner predictions were built under a different row grouping")
    output.mkdir(parents=True, exist_ok=True)
    truth, prompts = ensemble._targets(train), train["prompt_id"].to_numpy()
    folds = plan["outer_fold"].to_numpy()
    current = _aligned_current(current_oof, train, folds)
    in_group = np.isin(truth, INDEX)
    probabilities, selection, specialist_only = {"current": current}, {}, {}
    for name in FEATURE_SETS:
        system, specialist, chosen = np.empty_like(current), np.empty((len(train), len(SPECIALIST))), []
        for fold in range(5):
            column = plan[f"inner_{fold}"].to_numpy()
            scores = {weight: [] for weight in WEIGHTS}
            for inner_fold, (valid, inner_current) in _inner_current(current_cache, train, plan, fold).items():
                fit = np.flatnonzero((folds != fold) & (column != inner_fold))
                inner_specialist = specialist_probabilities(blocks, name, truth, fit, valid, frozen)
                for weight in WEIGHTS:
                    blended = redistribute(inner_current, inner_specialist, weight)
                    scores[weight].append(float(balanced_accuracy_score(truth[valid], blended.argmax(axis=1))))
            means = {weight: float(np.mean(values)) for weight, values in scores.items()}
            # Ties keep the smaller specialist weight, i.e. stay with the current system.
            weight = min(WEIGHTS, key=lambda value: (-means[value], value))
            rows = np.flatnonzero(folds == fold)
            specialist[rows] = specialist_probabilities(blocks, name, truth, np.flatnonzero(folds != fold), rows,
                                                        frozen)
            system[rows] = redistribute(current[rows], specialist[rows], weight)
            current_weights = (current_cache[current_cache["outer_fold"] == fold]
                               .groupby("inner_fold")["weight"].first().tolist())
            chosen.append({"outer_fold": fold, "weight": weight, "current_inner_weight": current_weights,
                           "inner_balanced_accuracy": {str(key): values for key, values in scores.items()}})
        probabilities[f"current+{name}"] = system
        selection[name] = chosen
        # Seven-way separation on the true Claude and GLM rows, against the current system's own.
        rows = np.flatnonzero(in_group)
        specialist_only[name] = float(balanced_accuracy_score(truth[rows], INDEX[specialist[rows].argmax(axis=1)]))
        np.save(output / f"specialist-{name}.npy", specialist)
    group_rows = np.flatnonzero(in_group)
    specialist_only["current"] = float(balanced_accuracy_score(
        truth[group_rows], INDEX[current[group_rows][:, INDEX].argmax(axis=1)]))
    predictions = {name: values.argmax(axis=1) for name, values in probabilities.items()}
    systems = {name: ensemble._system_metrics(truth, values, folds) for name, values in predictions.items()}
    paired = {f"{name}_vs_current": ensemble.prompt_bootstrap(truth, predictions["current"], values, prompts,
                                                              samples, seed)
              for name, values in predictions.items() if name != "current"}
    paired["current+qwen+styledistance_vs_current+qwen"] = ensemble.prompt_bootstrap(
        truth, predictions["current+qwen"], predictions["current+qwen+styledistance"], prompts, samples, seed)
    changes = {}
    for name, values in predictions.items():
        if name == "current":
            continue
        gained = (predictions["current"] != truth) & (values == truth)
        lost = (predictions["current"] == truth) & (values != truth)
        changes[name] = {"gained": int(gained.sum()), "lost": int(lost.sum()),
                         "by_class": {label: {"gained": int(gained[truth == index].sum()),
                                              "lost": int(lost[truth == index].sum())}
                                      for index, label in enumerate(LABELS)
                                      if gained[truth == index].any() or lost[truth == index].any()}}
    frame = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"}).assign(outer_fold=folds)
    for name, values in predictions.items():
        frame[f"prediction::{name}"] = np.asarray(LABELS)[values]
    frame.to_parquet(output / "predictions.parquet", index=False)
    result = {"specialist_classes": list(SPECIALIST), "weights": list(WEIGHTS), "systems": systems,
              "paired": paired, "selection": selection, "changes": changes,
              "seven_class_balanced_accuracy_on_true_rows": specialist_only,
              "groups": int(train["prompt_id"].nunique()), "groups_sha256": groups_sha256(train),
              "train_fingerprint": fingerprint_frame(train), "provenance": provenance or {}}
    _json(output / "comparison.json", result)
    return result


def recipe_sha256(recipe: dict) -> str:
    """Identity of the current system's frozen recipe, as parsed JSON."""
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def freeze_recipe(train: pd.DataFrame, blocks: dict[str, np.ndarray], final_plan: pd.DataFrame,
                  current_recipe_sha256: str, *, frozen: neural.FrozenConfig | None = None,
                  blend: ensemble.BlendConfig | None = None, validation: dict | None = None) -> dict:
    """Choose the final specialist weight once, on inner prompt folds over every official row."""
    frozen, blend = frozen or neural.FrozenConfig(), blend or ensemble.BlendConfig()
    inner = _aligned_plan(train, final_plan, AdapterConfig())
    truth = ensemble._targets(train)
    scores, current_weights = {weight: [] for weight in WEIGHTS}, []
    for inner_fold in sorted(set(inner)):
        fit, valid = np.flatnonzero(inner != inner_fold), np.flatnonzero(inner == inner_fold)
        current, current_weight = current_on_split(train, blocks["qwen"], fit, valid, blend,
                                                   FINAL_SEED_OFFSET + inner_fold)
        current_weights.append(current_weight)
        specialist = specialist_probabilities(blocks, FINAL_FEATURE_SET, truth, fit, valid, frozen)
        for weight in WEIGHTS:
            predictions = redistribute(current, specialist, weight).argmax(axis=1)
            scores[weight].append(float(balanced_accuracy_score(truth[valid], predictions)))
    means = {weight: float(np.mean(values)) for weight, values in scores.items()}
    weight = min(WEIGHTS, key=lambda value: (-means[value], value))
    styledistance = ENCODERS["styledistance"]
    return {"schema_version": 1, "kind": RECIPE_KIND, "labels": list(LABELS),
            "specialist_classes": list(SPECIALIST), "feature_set": FINAL_FEATURE_SET, "weight": weight,
            "inner_balanced_accuracy_by_weight": {str(key): values for key, values in scores.items()},
            "current_inner_weights": current_weights, "C": frozen.C, "seed": frozen.seed,
            "qwen_model_id": neural.MODEL_ID, "qwen_model_revision": neural.MODEL_REVISION,
            "styledistance_model_id": styledistance["model_id"],
            "styledistance_model_revision": styledistance["model_revision"], "styledistance_max_length": 512,
            "current_recipe_sha256": current_recipe_sha256, "train_fingerprint": fingerprint_frame(train),
            "nested_validation": validation or {}}


def _checked_recipe(recipe: dict) -> dict:
    styledistance = ENCODERS["styledistance"]
    if (recipe.get("schema_version") != 1 or recipe.get("kind") != RECIPE_KIND
            or recipe.get("labels") != list(LABELS) or recipe.get("specialist_classes") != list(SPECIALIST)
            or recipe.get("feature_set") != FINAL_FEATURE_SET or recipe.get("weight") not in WEIGHTS
            or recipe.get("styledistance_model_revision") != styledistance["model_revision"]
            or recipe.get("qwen_model_revision") != neural.MODEL_REVISION):
        raise ValueError("Unsupported Claude-GLM specialist recipe")
    return recipe


def fit_final(train: pd.DataFrame, blend_dir: str | Path, styledistance_path: str | Path, recipe: dict,
              output_dir: str | Path, device: str = "auto") -> dict:
    """Fit the specialist on every official Claude and GLM row: the blend fit's own Qwen
    features, and StyleDistance extracted here."""
    recipe = _checked_recipe(recipe)
    if recipe["train_fingerprint"] != fingerprint_frame(train):
        raise ValueError("Training data differ from the rows used to freeze the specialist recipe")
    blend = joblib.load(Path(blend_dir) / "blend.joblib")  # trusted output of `ntl.ensemble fit` in this run
    if recipe_sha256(blend["recipe"]) != recipe["current_recipe_sha256"]:
        raise ValueError("The fitted current system differs from the one the specialist was chosen for")
    output = Path(output_dir)
    if (output / "specialist.joblib").exists():
        raise ValueError("Use a new output directory; a fitted specialist must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _, _, qwen_config = ensemble.load_recipe(blend["recipe"])
    qwen, _ = ensemble.load_archive(Path(blend_dir) / "train-features.npz", train, qwen_config)
    styledistance, profile = extract_embeddings(train["text"].tolist(), styledistance_path, "styledistance",
                                                device=_device(device))
    frozen = neural.FrozenConfig(C=recipe["C"], seed=recipe["seed"])
    blocks = {"qwen": qwen, "styledistance": styledistance}
    truth = ensemble._targets(train)
    names = FEATURE_SETS[FINAL_FEATURE_SET]
    rows = np.flatnonzero(np.isin(truth, INDEX))
    head = specialist_head(tuple(blocks[name].shape[1] for name in names), frozen)
    with threadpool_limits(limits=2):
        head.fit(np.hstack([blocks[name] for name in names])[rows], _POSITION[truth[rows]])
    if not np.array_equal(head.classes_, np.arange(len(SPECIALIST))):
        raise ValueError("Specialist head must contain every specialist class")
    metadata = {"weight": recipe["weight"], "train_fingerprint": fingerprint_frame(train),
                "specialist_rows": len(rows), "styledistance_profile": profile["profile"],
                "fit_seconds": time.perf_counter() - started,
                "versions": {package: importlib.metadata.version(package)
                             for package in ("numpy", "scikit-learn", "scipy")}}
    # joblib is trusted project output; never load a third-party pickle with predict_final.
    joblib.dump({"head": head, "recipe": recipe, "metadata": metadata}, output / "specialist.joblib")
    _json(output / "fit.json", metadata)
    return metadata


def _device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def predict_final(specialist_dir: str | Path, blend_dir: str | Path, test: pd.DataFrame, qwen_path: str | Path,
                  styledistance_path: str | Path, output: str | Path, device: str = "auto") -> pd.DataFrame:
    """Score every current test row: the current blend's probabilities, redistributed on the
    Claude and GLM classes by the specialist."""
    specialist = joblib.load(Path(specialist_dir) / "specialist.joblib")  # trusted output of `fit` in this run
    recipe = _checked_recipe(specialist["recipe"])
    blend = joblib.load(Path(blend_dir) / "blend.joblib")
    if recipe_sha256(blend["recipe"]) != recipe["current_recipe_sha256"]:
        raise ValueError("The fitted current system differs from the one the specialist was chosen for")
    blend_weight, _, qwen_config = ensemble.load_recipe(blend["recipe"])
    started = time.perf_counter()
    qwen_config = neural.FrozenConfig(**{**asdict(qwen_config), "device": device})
    qwen, qwen_profile = neural.extract_features(test["text"].tolist(), qwen_path, qwen_config)
    styledistance, styledistance_profile = extract_embeddings(test["text"].tolist(), styledistance_path,
                                                              "styledistance", device=_device(device))
    with threadpool_limits(limits=2):
        cpu = aligned_probabilities(blend["cpu"], test["text"])
        current = ensemble._blend(cpu, neural._probabilities(blend["qwen_head"], qwen), blend_weight)
        q = specialist["head"].predict_proba(np.hstack([qwen, styledistance]))
    if not np.array_equal(specialist["head"].classes_, np.arange(len(SPECIALIST))):
        raise ValueError("Specialist head must contain every specialist class")
    final = redistribute(current, q, recipe["weight"])
    if not np.isfinite(final).all() or not np.allclose(final.sum(axis=1), 1.0):
        raise ValueError("Final scores must be finite probability vectors")
    submission = write_submission(test["id"], np.asarray(LABELS)[final.argmax(axis=1)].tolist(), output)
    output = Path(output)
    _json(output.with_name(f"{output.stem}_profile.json"),
          {"rows": len(test), "weight": recipe["weight"], "blend_weight": blend_weight,
           "changed_from_current": int((final.argmax(axis=1) != current.argmax(axis=1)).sum()),
           "elapsed_seconds": time.perf_counter() - started, "qwen_profile": qwen_profile,
           "styledistance_profile": styledistance_profile["profile"]})
    return submission


def _load_blocks(train: pd.DataFrame, qwen_path: Path, styledistance_path: Path) -> dict[str, np.ndarray]:
    qwen, _ = ensemble.load_archive(qwen_path, train, neural.FrozenConfig())
    styledistance, metadata = load_feature_archive(styledistance_path, train)
    if metadata.get("model_id") != "StyleDistance/styledistance":
        raise ValueError("Expected StyleDistance features")
    return {"qwen": qwen, "styledistance": styledistance}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _grouped(train: pd.DataFrame, plan: pd.DataFrame, groups: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows and plan under the chosen grouping (prompts unless `groups` is given)."""
    if groups is None:
        return train, plan
    train = regroup(train, pd.read_csv(groups, dtype={"id": str, "prompt_id": str}, keep_default_na=False))
    return train, plan.assign(prompt_id=plan["id"].map(train.set_index("id")["prompt_id"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    features = commands.add_parser("features", help="Frozen StyleDistance embeddings for every training row")
    features.add_argument("--train", type=Path, required=True)
    features.add_argument("--model-path", type=Path, required=True)
    features.add_argument("--output", type=Path, required=True)
    features.add_argument("--device", default="cpu")
    inner = commands.add_parser("current-inner", help="Refit the current system on one outer fold's inner splits")
    inner.add_argument("--train", type=Path, required=True)
    inner.add_argument("--plan", type=Path, required=True)
    inner.add_argument("--qwen-features", type=Path, required=True)
    inner.add_argument("--outer-fold", type=int, required=True)
    inner.add_argument("--output", type=Path, required=True)
    inner.add_argument("--groups", type=Path, help="CSV of id, prompt_id regrouping rows (e.g. by topic)")
    compare = commands.add_parser("evaluate", help="Control and candidate specialists against the current system")
    compare.add_argument("--train", type=Path, required=True)
    compare.add_argument("--plan", type=Path, required=True)
    compare.add_argument("--qwen-features", type=Path, required=True)
    compare.add_argument("--styledistance-features", type=Path, required=True)
    compare.add_argument("--current-oof", type=Path, required=True)
    compare.add_argument("--current-inner", type=Path, nargs="+", required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--groups", type=Path, help="CSV of id, prompt_id regrouping rows (e.g. by topic)")
    freeze = commands.add_parser("freeze", help="Choose the final specialist weight on full-data inner folds")
    freeze.add_argument("--train", type=Path, required=True)
    freeze.add_argument("--final-plan", type=Path, required=True)
    freeze.add_argument("--qwen-features", type=Path, required=True)
    freeze.add_argument("--styledistance-features", type=Path, required=True)
    freeze.add_argument("--current-recipe", type=Path, required=True)
    freeze.add_argument("--validation", type=Path, nargs="+", default=[], help="comparison.json files")
    freeze.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("fit", help="Fit the specialist on every official Claude and GLM row")
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--blend-dir", type=Path, required=True)
    fit.add_argument("--styledistance-path", type=Path, required=True)
    fit.add_argument("--recipe", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--device", default="auto")
    predict = commands.add_parser("predict", help="Score test.csv with the current system and the specialist")
    predict.add_argument("--specialist-dir", type=Path, required=True)
    predict.add_argument("--blend-dir", type=Path, required=True)
    predict.add_argument("--test", type=Path, required=True)
    predict.add_argument("--model-path", type=Path, required=True)
    predict.add_argument("--styledistance-path", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.command == "predict":
        predict_final(args.specialist_dir, args.blend_dir, read_test(args.test), args.model_path,
                      args.styledistance_path, args.output, args.device)
        return 0
    train = read_train(args.train)
    if args.command == "fit":
        print(json.dumps(fit_final(train, args.blend_dir, args.styledistance_path,
                                   json.loads(args.recipe.read_text()), args.output_dir, args.device), indent=2))
        return 0
    if args.command == "freeze":
        if args.output.exists():
            raise ValueError("Use a new output path; a frozen recipe must not be overwritten")
        blocks = _load_blocks(train, args.qwen_features, args.styledistance_features)
        final_plan = pd.read_csv(args.final_plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
        validation = {}
        for path in args.validation:
            result = json.loads(path.read_text())
            paired = result["paired"][f"current+{FINAL_FEATURE_SET}_vs_current"]
            validation[str(path)] = {
                "groups": result["groups"],
                "balanced_accuracy": {name: values["balanced_accuracy"] for name, values in result["systems"].items()},
                "delta": paired["balanced_accuracy_delta"], "delta_95_interval": paired["delta_95_interval"],
                "selected_weight_per_outer_fold": [item["weight"] for item in result["selection"][FINAL_FEATURE_SET]]}
        recipe = freeze_recipe(train, blocks, final_plan,
                               recipe_sha256(json.loads(args.current_recipe.read_text())), validation=validation)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _json(args.output, recipe)
        print(json.dumps({key: recipe[key] for key in ("weight", "inner_balanced_accuracy_by_weight",
                                                       "current_inner_weights")}, indent=2))
        return 0
    if args.command == "features":
        if args.output.exists():
            raise ValueError("Feature output already exists; preserve it or choose another output path")
        embeddings, metadata = extract_embeddings(train["text"].tolist(), args.model_path, "styledistance",
                                                  device=args.device)
        metadata["train_fingerprint"] = fingerprint_frame(train)
        saved = save_feature_archive(args.output, train["id"].tolist(), embeddings, metadata,
                                     texts=train["text"].tolist())
        args.output.with_suffix(".json").write_text(json.dumps(saved, indent=2) + "\n")
        print(json.dumps(metadata["profile"], indent=2))
        return 0
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    if args.command == "current-inner":
        if args.output.exists():
            raise ValueError("Use a new output path; cached inner predictions must not be overwritten")
        qwen, _ = ensemble.load_archive(args.qwen_features, train, neural.FrozenConfig())
        train, plan = _grouped(train, plan, args.groups)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        build_current_inner(train, qwen, plan, args.outer_fold).to_parquet(args.output, index=False)
        return 0
    blocks = _load_blocks(train, args.qwen_features, args.styledistance_features)
    train, plan = _grouped(train, plan, args.groups)
    cache = pd.concat([pd.read_parquet(path) for path in args.current_inner], ignore_index=True)
    inputs = [args.plan, args.qwen_features, args.styledistance_features, args.current_oof, *args.current_inner]
    provenance = {str(path): _sha256(path) for path in inputs + ([args.groups] if args.groups else [])}
    result = evaluate(train, blocks, plan, pd.read_parquet(args.current_oof), cache, args.output_dir,
                      provenance={"input_sha256": provenance})
    print(json.dumps({"balanced_accuracy": {name: round(values["balanced_accuracy"], 4)
                                            for name, values in result["systems"].items()},
                      "weights": {name: [item["weight"] for item in items]
                                  for name, items in result["selection"].items()},
                      "seven_class": {name: round(value, 4) for name, value in
                                      result["seven_class_balanced_accuracy_on_true_rows"].items()},
                      "paired": {name: [round(values["balanced_accuracy_delta"], 4)]
                                 + [round(bound, 4) for bound in values["delta_95_interval"]]
                                 for name, values in result["paired"].items()},
                      "changes": {name: [values["gained"], values["lost"]]
                                  for name, values in result["changes"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    # Run the imported module, not this __main__ copy, so the fitted head pickles its
    # block-weight function as ntl.claude_glm._column_weights and loads from any entry point.
    from ntl.claude_glm import main as imported_main

    raise SystemExit(imported_main())
