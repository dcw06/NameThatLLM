"""Gemini 3.7 / 3.8 Flash binary scoring experiment (docs/gemini-binary-scoring.md).

One L2 logistic regression at a fixed C = 0.1 on the seven marker rates of the marker study's
frozen extractor (`ntl.gemini_markers`), with no length feature. Each of the five original
prompt folds standardizes and fits on the 240 true-pair passages outside it and scores the 60
inside it. The score is P(3.8 | the pair); above 0.5 predicts 3.8. The pooled held-out
predictions are compared with the incumbent's forced 3.7-or-3.8 choice (179/300).

`freeze` records the specification, inputs and features before any fit; `evaluate` refuses to
run unless they still match, then fits once.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import claude_glm, ensemble, gemini_markers
from .constants import LABELS
from .data import fingerprint_frame, read_train

EXPERIMENT = "gemini-binary-scoring/v1"
FEATURES = gemini_markers.MODELS["style"]
C = 0.1
THRESHOLD = 0.5
SOLVER = {"penalty": "l2", "solver": "lbfgs", "max_iter": 2000, "tol": 1e-6, "random_state": 2026}
REFERENCE_CORRECT = 179
SAMPLES, SEED = 10000, 2026
PAIR, PAIR_INDEX = gemini_markers.PAIR, gemini_markers.PAIR_INDEX
SOURCES = ("src/ntl/gemini_binary.py", "src/ntl/gemini_markers.py")


def settings() -> dict:
    return {"experiment": EXPERIMENT, "token_regex": gemini_markers.TOKEN_RE.pattern,
            "word_lists": gemini_markers.WORD_LISTS, "phrase": "not simply (whitespace-only separator)",
            "punctuation": ["; (U+003B)", ": (U+003A)"], "features": FEATURES, "rate": "100 * count / words",
            "length_feature": None, "C": C, "logistic": SOLVER, "standardization": "training rows only",
            "encoding": {PAIR[0]: 0, PAIR[1]: 1}, "threshold": f"3.8 when score > {THRESHOLD}",
            "reference_correct": REFERENCE_CORRECT, "bootstrap": {"samples": SAMPLES, "seed": SEED}}


def model() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("logistic", LogisticRegression(C=C, **SOLVER))])


def fit(values: np.ndarray, target: np.ndarray, rows: np.ndarray) -> Pipeline:
    """A fresh scaler and classifier fit on `rows` only; nonconvergence or nonfinite values fail."""
    if set(target[rows]) != {0, 1}:
        raise ValueError("Both Gemini versions need training rows")
    if not np.isfinite(values[rows]).all():
        raise ValueError("Marker rates must be finite")
    pipeline = model()
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(values[rows], target[rows])
    logistic = pipeline.named_steps["logistic"]
    if (int(logistic.n_iter_[0]) >= SOLVER["max_iter"] or not np.isfinite(logistic.coef_).all()
            or not np.array_equal(logistic.classes_, [0, 1])):
        raise ValueError("The scoring model failed to converge or gave nonfinite weights")
    return pipeline


def weights(pipeline: Pipeline) -> dict:
    """Intercept and standardized weights, with the scaler and the same weights per 100 words."""
    scale, logistic = pipeline.named_steps["scale"], pipeline.named_steps["logistic"]
    coef = logistic.coef_[0]
    names = [name.removeprefix("rate_") for name in FEATURES]
    return {"intercept": float(logistic.intercept_[0]),
            "standardized": dict(zip(names, map(float, coef))),
            "scaler_mean": dict(zip(names, map(float, scale.mean_))),
            "scaler_scale": dict(zip(names, map(float, scale.scale_))),
            "per_100_words": dict(zip(names, map(float, coef / scale.scale_))),
            "intercept_per_100_words": float(logistic.intercept_[0] - (coef * scale.mean_ / scale.scale_).sum()),
            "n_iter": int(logistic.n_iter_[0])}


def cross_validate(train: pd.DataFrame, folds: np.ndarray, features: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    """Held-out scores for every true-pair passage, and each fold's fitted weights."""
    truth = ensemble._targets(train)
    pair = np.isin(truth, PAIR_INDEX)
    target = (truth == PAIR_INDEX[1]).astype(int)
    prompts = train["prompt_id"].to_numpy()
    if pd.Series(folds[pair]).groupby(prompts[pair]).nunique().max() != 1:
        raise ValueError("Both responses to a prompt must share a fold")
    values = features[list(FEATURES)].to_numpy(np.float64)
    scores, records = np.full(len(train), np.nan), []
    for fold in sorted(set(folds[pair])):
        rows, held = np.flatnonzero(pair & (folds != fold)), np.flatnonzero(pair & (folds == fold))
        if set(prompts[rows]) & set(prompts[held]):
            raise ValueError("A prompt is on both sides of a fold")
        pipeline = fit(values, target, rows)
        scores[held] = pipeline.predict_proba(values[held])[:, 1]
        records.append({"outer_fold": int(fold), "training_rows": int(len(rows)), "held_out_rows": int(len(held)),
                        "training_prompts": int(len(set(prompts[rows]))), **weights(pipeline)})
    if not np.isfinite(scores[pair]).all():
        raise ValueError("Every true-pair passage needs one held-out score")
    return scores, records


def forced_share(probabilities: np.ndarray) -> np.ndarray:
    """The incumbent's preference for 3.8 within the pair: p_3.8 / (p_3.7 + p_3.8)."""
    mass = probabilities[:, PAIR_INDEX].sum(axis=1)
    if (mass <= 0).any():
        raise ValueError("The incumbent must give the pair positive mass")
    return probabilities[:, PAIR_INDEX[1]] / mass


def compare(target: np.ndarray, scores: np.ndarray, incumbent: np.ndarray) -> dict:
    """Binary accuracy, recall per version, and errors corrected against correct predictions lost."""
    model_right = (scores > THRESHOLD).astype(int) == target
    incumbent_right = (incumbent > THRESHOLD).astype(int) == target

    def summary(right: np.ndarray) -> dict:
        return {"correct": int(right.sum()), "accuracy": float(right.mean()),
                "recall": {PAIR[0]: float(right[target == 0].mean()), PAIR[1]: float(right[target == 1].mean())},
                "correct_by_version": {PAIR[0]: int(right[target == 0].sum()), PAIR[1]: int(right[target == 1].sum())}}

    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    return {"rows": int(len(target)), "model": summary(model_right), "incumbent_forced": summary(incumbent_right),
            "errors_corrected": int((~incumbent_right & model_right).sum()),
            "correct_lost": int((incumbent_right & ~model_right).sum()),
            "both_correct": int((incumbent_right & model_right).sum()),
            "both_wrong": int((~incumbent_right & ~model_right).sum()),
            "predicted_3.8": int((scores > THRESHOLD).sum()),
            "brier": float(np.mean((scores - target) ** 2)),
            "log_loss": float(-np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)))}


def _paths(args) -> dict:
    return {"train": str(args.train), "plan": str(args.plan), "reference": str(args.reference),
            "reference_npz": str(args.reference.with_suffix(".npz"))}


def freeze(paths: dict, output: Path) -> dict:
    """Specification, input and source hashes, and the feature table; fits nothing."""
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("The specification is already frozen; it must not be overwritten")
    train = read_train(paths["train"])
    truth = ensemble._targets(train)
    pair = np.isin(truth, PAIR_INDEX)
    features = gemini_markers.feature_table(train[pair])  # rejects empty passages
    output.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output / "features.parquet", index=False)
    manifest = {"settings": json.loads(json.dumps(settings())), "paths": paths,
                "train_fingerprint": fingerprint_frame(train),
                "input_sha256": {path: gemini_markers._sha256(path) for path in paths.values()},
                "source_sha256": {path: gemini_markers._sha256(path) for path in SOURCES},
                "features_sha256": gemini_markers._sha256(output / "features.parquet"),
                "pair_rows": int(pair.sum()), "minimum_words": int(features["words"].min())}
    claude_glm._json(output / "manifest.json", manifest)
    return manifest


def evaluate(output: Path) -> dict:
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    changed = [path for path, digest in {**manifest["input_sha256"], **manifest["source_sha256"]}.items()
               if gemini_markers._sha256(path) != digest]
    if gemini_markers._sha256(output / "features.parquet") != manifest["features_sha256"]:
        changed.append("features.parquet")
    if json.loads(json.dumps(settings())) != manifest["settings"]:
        changed.append("settings")
    if changed:
        raise ValueError(f"The frozen specification no longer matches: {changed}")
    if (output / "comparison.json").exists():
        raise ValueError("Already evaluated; the results must not be overwritten")
    paths = manifest["paths"]
    train = read_train(paths["train"])
    plan = pd.read_csv(paths["plan"], dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    folds = train["id"].map(plan.set_index("id")["outer_fold"]).to_numpy()
    truth = ensemble._targets(train)
    pair = np.isin(truth, PAIR_INDEX)
    features = pd.read_parquet(output / "features.parquet").set_index("id").reindex(train["id"]).reset_index()
    reference = json.loads(Path(paths["reference"]).read_text())
    locked = np.load(paths["reference_npz"], allow_pickle=False)
    if not np.array_equal(locked["ids"], train["id"].to_numpy().astype(str)) or \
            reference["groups_sha256"] != claude_glm.groups_sha256(train):
        raise ValueError("The locked incumbent is for other rows")
    scores, records = cross_validate(train, folds, features)
    if [record["training_rows"] for record in records] != [240] * 5 or \
            [record["held_out_rows"] for record in records] != [60] * 5:
        raise ValueError("Each fold must train on 240 passages and hold out 60")
    target = (truth == PAIR_INDEX[1]).astype(int)
    share = forced_share(locked["probabilities"])
    pooled = compare(target[pair], scores[pair], share[pair])
    if pooled["incumbent_forced"]["correct"] != REFERENCE_CORRECT:
        raise ValueError(f"The incumbent's forced choice gets {pooled['incumbent_forced']['correct']}, not 179")
    by_fold = {str(fold): compare(target[pair & (folds == fold)], scores[pair & (folds == fold)],
                                  share[pair & (folds == fold)]) for fold in range(5)}
    paired = ensemble.prompt_bootstrap(target[pair], (share[pair] > THRESHOLD).astype(int),
                                       (scores[pair] > THRESHOLD).astype(int), train["prompt_id"].to_numpy()[pair],
                                       SAMPLES, SEED)
    reference_fit = weights(fit(features[list(FEATURES)].to_numpy(np.float64), target, np.flatnonzero(pair)))
    frame = train.loc[pair, ["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    frame["outer_fold"] = folds[pair]
    frame["score"] = scores[pair]
    frame["prediction"] = np.where(scores[pair] > THRESHOLD, PAIR[1], PAIR[0])
    frame["incumbent_share"] = share[pair]
    frame["incumbent_prediction"] = np.where(share[pair] > THRESHOLD, PAIR[1], PAIR[0])
    frame["incumbent_18_class"] = np.asarray(LABELS)[locked["probabilities"][pair].argmax(axis=1)]
    frame.to_parquet(output / "scores.parquet", index=False)
    claude_glm._json(output / "weights.json", {"folds": records, "all_300_reference_fit": reference_fit,
                                                "note": "The all-300 fit has no held-out estimate."})
    result = {"experiment": EXPERIMENT, "pooled": pooled, "by_fold": by_fold, "paired_bootstrap": paired,
              "manifest_sha256": gemini_markers._sha256(output / "manifest.json")}
    claude_glm._json(output / "comparison.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("freeze", help="Record the specification and inputs; fits nothing")
    for name in ("train", "plan", "reference", "output"):
        first.add_argument(f"--{name}", type=Path, required=True)
    second = commands.add_parser("evaluate", help="Check the frozen specification, then fit once")
    second.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        manifest = freeze(_paths(args), args.output)
        print(json.dumps({key: manifest[key] for key in ("pair_rows", "minimum_words", "features_sha256")}, indent=2))
    else:
        result = evaluate(args.output)
        pooled = result["pooled"]
        print(json.dumps({"model": pooled["model"], "incumbent_forced": pooled["incumbent_forced"],
                          "errors_corrected": pooled["errors_corrected"], "correct_lost": pooled["correct_lost"],
                          "accuracy_change_interval": result["paired_bootstrap"]["delta_95_interval"]}, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.gemini_binary import main as imported_main

    raise SystemExit(imported_main())
