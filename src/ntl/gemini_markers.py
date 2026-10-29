"""Gemini 3.7 / 3.8 Flash marker study (docs/gemini-marker-study.md, protocol version 2).

Seven frozen marker rates (five word lists, semicolons, colons) and three logistic models on
the 300 true-pair passages: length-only (log W), style-only (the seven rates, the sole
integration candidate) and length + style. In each outer prompt fold the two inner folds
choose each model's C by mean inner binary balanced accuracy, then style-only's a in
{0, 1/4, 1/2} by mean inner 18-class balanced accuracy after the pair-mass update
p'_k = (1 - a) p_k + a M q_k, inside the complete incumbent rebuilt on each inner partition.

`preflight` checks the references, audits the inner caches and the incumbent's
reconstruction, and freezes the specification in a manifest before any new model is fit.
`evaluate` refuses to run unless the frozen manifest still matches, then fits once.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import claude_glm, ensemble, gemini_pair, neural
from .constants import LABELS
from .data import fingerprint_frame, read_train
from .lora import AdapterConfig, validate_plan
from .selection import first_best, snap_tie

PROTOCOL = "gemini-marker-study/v2"
TOKEN_RE = re.compile(r"\b\w+(?:['’\-]\w+)*\b", re.UNICODE)
WORD_LISTS = {
    "action": ("shatter", "ignite", "collapse", "bloom", "demands", "pierce", "erupt"),
    "qualifying": ("merely", "essentially", "partially", "relatively"),  # plus the phrase "not simply"
    "formal": ("wherein", "whereby", "thus", "consequently", "yielding"),
    "systemic": ("paradigm", "framework", "syntax", "subroutines", "equilibrium"),
    "sensory": ("void", "abyss", "boiling", "froth", "molten", "violent", "tears"),
}
LEXICAL = tuple(WORD_LISTS)
FEATURES = (*LEXICAL, "semicolons", "colons")
MODELS = {"length": ("log_words",), "style": tuple(f"rate_{name}" for name in FEATURES),
          "length+style": ("log_words", *(f"rate_{name}" for name in FEATURES))}
CANDIDATE = "style"
C_GRID = (0.01, 0.1, 1.0)
A_GRID = (0.0, 0.25, 0.5)
SEED = 2026
SAMPLES = 10000
GAIN = 0.005
BINS = ((0, 60, "<60"), (60, 100, "60-99"), (100, 150, "100-149"), (150, None, ">=150"))
PAIR, PAIR_INDEX = gemini_pair.PAIR, gemini_pair.PAIR_INDEX
REFERENCES = {"forced_binary_correct": 179, "pair_18_class_correct": 139,
              "prompt_balanced_accuracy": 0.4700745712155108, "topic_balanced_accuracy": 0.423027591349739}
DESCRIPTIVE = {"pair_words": 32672, "mean_words_3.7": 117.13, "mean_words_3.8": 100.69,
               "all_seven_zero": [45, 44], "all_lexical_zero": [73, 65]}
MARKER_HISTORY = {
    "known": [
        "An external rubric pasted into the conversation on 2026-10-28 proposed kinetic verbs (shatter, ignite, "
        "collapse, bloom, demands), qualifiers (merely, not simply, partially, insofar as), formal connectors "
        "(wherein, whereby, consequently, yielding), systemic terms (paradigm, architecture, syntax, subroutines, "
        "equilibrium) and absolutes (forever, completely, total, shattered, abyss), citing passages it had read.",
        "Label-wise paired counts of different, broader dictionaries were then computed over the 150 prompt pairs "
        "(docs/gemini-rubric.md, claim checks); most proposed differences were not supported.",
        "The seven-feature lists and the tokenizer were supplied in the protocol document "
        "(docs/gemini-marker-study.md), together with descriptive counts from an audit program.",
    ],
    "unknown": [
        "How the exact forms demands, subroutines and tears were chosen.",
        "How the sensory list (void, abyss, boiling, froth, molten, violent, tears) was chosen.",
        "How the semicolon and colon features were chosen.",
        "Which alternatives, if any, were tried; the original audit program and a selection log are not recorded.",
    ],
}


# ---------------------------------------------------------------------------------------------
# Features


def extract(text: str) -> dict:
    """Word count W, occurrence counts of the seven features, and their rates per 100 words."""
    matches = list(TOKEN_RE.finditer(text))
    tokens = [match.group(0).casefold() for match in matches]
    counts = {name: sum(token in words for token in tokens) for name, words in WORD_LISTS.items()}
    counts["qualifying"] += sum(
        first == "not" and second == "simply" and text[left.end():right.start()] != ""
        and text[left.end():right.start()].isspace()
        for (first, left), (second, right) in zip(zip(tokens, matches), zip(tokens[1:], matches[1:])))
    counts["semicolons"], counts["colons"] = text.count(";"), text.count(":")
    words = len(tokens)
    result = {"words": words, **{f"count_{name}": counts[name] for name in FEATURES}}
    if words:
        result["log_words"] = float(np.log(words))
        result.update({f"rate_{name}": 100 * counts[name] / words for name in FEATURES})
    return result


def length_bin(words: int) -> str:
    return next(label for low, high, label in BINS if words >= low and (high is None or words < high))


def feature_table(train: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame([extract(text) for text in train["text"]])
    table.insert(0, "id", train["id"].to_numpy())
    if (table["words"] == 0).any():
        raise ValueError("Every official row needs W > 0")
    table["all_seven_zero"] = (table[[f"count_{name}" for name in FEATURES]] == 0).all(axis=1)
    table["all_lexical_zero"] = (table[[f"count_{name}" for name in LEXICAL]] == 0).all(axis=1)
    table["length_bin"] = table["words"].map(length_bin)
    return table


def descriptive(train: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Pair-population counts, for comparison with the protocol's tokenizer audit."""
    truth = ensemble._targets(train)
    by_version = [table[truth == index] for index in PAIR_INDEX]
    pair = table[np.isin(truth, PAIR_INDEX)]
    return {"pair_words": int(pair["words"].sum()),
            "mean_words_3.7": round(float(by_version[0]["words"].mean()), 2),
            "mean_words_3.8": round(float(by_version[1]["words"].mean()), 2),
            "all_seven_zero": [int(part["all_seven_zero"].sum()) for part in by_version],
            "all_lexical_zero": [int(part["all_lexical_zero"].sum()) for part in by_version],
            "minimum_words_all_rows": int(table["words"].min()),
            # Pooled occurrences per 100 pooled words, kept apart from mean passage rates.
            "pooled_rate": {PAIR[position]: {name: 100 * float(part[f"count_{name}"].sum() / part["words"].sum())
                                             for name in FEATURES} for position, part in enumerate(by_version)},
            "mean_passage_rate": {PAIR[position]: {name: float(part[f"rate_{name}"].mean()) for name in FEATURES}
                                  for position, part in enumerate(by_version)}}


# ---------------------------------------------------------------------------------------------
# Models, integration and selection


def fit_scores(values: np.ndarray, target: np.ndarray, fit: np.ndarray, rows: np.ndarray, C: float
               ) -> tuple[np.ndarray, dict]:
    """A fresh scaler and L2 logistic regression fit on `fit`; P(3.8) for `rows`. A fit that does not
    converge, or gives anything nonfinite, fails the run."""
    if set(target[fit]) != {0, 1}:
        raise ValueError("Both Gemini versions need fitting rows")
    model = Pipeline([("scale", StandardScaler()),
                      ("logistic", LogisticRegression(C=C, solver="lbfgs", max_iter=2000, tol=1e-6,
                                                      random_state=SEED))])
    inputs = np.asarray(values, dtype=np.float64)
    if not np.isfinite(inputs[fit]).all() or not np.isfinite(inputs[rows]).all():
        raise ValueError("Marker inputs must be finite")
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(inputs[fit], target[fit])
        scores = model.predict_proba(inputs[rows])[:, 1]
    logistic = model.named_steps["logistic"]
    iterations = int(logistic.n_iter_[0])
    if (not np.array_equal(logistic.classes_, [0, 1]) or iterations >= 2000
            or not np.isfinite(logistic.coef_).all() or not np.isfinite(scores).all()):
        raise ValueError("A marker model failed to converge or gave nonfinite values")
    return scores, {"C": C, "n_iter": iterations, "converged": True, "fit_rows": int(len(fit))}


def integrate(incumbent: np.ndarray, scores: np.ndarray, weight: float) -> np.ndarray:
    """p'_3.8 = (1 - a) p_3.8 + a M S and p'_3.7 = (1 - a) p_3.7 + a M (1 - S); a = 0 returns the
    supplied vectors themselves. Rows with a nonfinite S (W = 0 at inference) keep their vector."""
    if weight == 0:
        return incumbent
    eligible = np.isfinite(scores)
    result = np.array(incumbent, dtype=float, copy=True)
    result[eligible] = claude_glm.redistribute(
        incumbent[eligible], np.column_stack([1 - scores[eligible], scores[eligible]]), weight, PAIR_INDEX)
    others = np.setdiff1d(np.arange(len(LABELS)), PAIR_INDEX)
    if (not np.isfinite(result).all() or (result < 0).any()
            or np.abs(result.sum(axis=1) - 1).max() > 1e-12
            or np.abs(result[:, PAIR_INDEX].sum(axis=1) - incumbent[:, PAIR_INDEX].sum(axis=1)).max() > 1e-12
            or not np.array_equal(result[:, others], incumbent[:, others])
            or not np.array_equal(result[~eligible], incumbent[~eligible])):
        raise ValueError("The pair-mass update broke a probability invariant")
    return result


def binary_share(probabilities: np.ndarray) -> tuple[np.ndarray, int]:
    """q = p_3.8 / M, with q = 0.5 where M is exactly zero; also the number of such rows."""
    mass = probabilities[:, PAIR_INDEX].sum(axis=1)
    zero = mass == 0
    share = np.full(len(probabilities), 0.5)
    share[~zero] = probabilities[~zero, PAIR_INDEX[1]] / mass[~zero]
    return share, int(zero.sum())


def _binary_ba(target: np.ndarray, scores: np.ndarray) -> float:
    return float(balanced_accuracy_score(target, scores > 0.5))


def _ba(truth: np.ndarray, probabilities: np.ndarray) -> float:
    return float(balanced_accuracy_score(truth, probabilities.argmax(axis=1)))


def run_fold(train: pd.DataFrame, blocks: dict[str, np.ndarray], features: pd.DataFrame, plan: pd.DataFrame,
             current: np.ndarray, current_cache: pd.DataFrame, locked: np.ndarray, fold: int,
             frozen: neural.FrozenConfig) -> dict:
    """The declared procedure for one outer fold: C per model, then style-only's a, then one outer fit."""
    truth, folds = ensemble._targets(train), plan["outer_fold"].to_numpy()
    pair = np.isin(truth, PAIR_INDEX)
    target = (truth == PAIR_INDEX[1]).astype(int)
    inputs = {name: features[list(columns)].to_numpy(np.float64) for name, columns in MODELS.items()}
    rebuilt, s, s_scores = gemini_pair.inner_incumbents(train, blocks, plan, current_cache, fold, frozen)
    fits, inner_binary, inner_style = [], {name: {C: [] for C in C_GRID} for name in MODELS}, []
    partitions = []
    for inner_fold, fit, valid, base, specialist in rebuilt:
        if np.isin(valid, np.flatnonzero(folds == fold)).any() or np.isin(fit, valid).any():
            raise ValueError("An inner partition overlaps its validation rows or the outer holdout")
        pair_fit, pair_valid = fit[pair[fit]], valid[pair[valid]]
        if set(target[pair_valid]) != {0, 1}:
            raise ValueError("Both Gemini versions need inner validation rows")
        partitions.append({"inner_fold": int(inner_fold), "fit_pair_rows": int(len(pair_fit)),
                           "valid_pair_rows": int(len(pair_valid)), "fit_rows": int(len(fit)),
                           "valid_rows": int(len(valid)),
                           "fit_pair_ids_sha256": _ids(train["id"].to_numpy()[pair_fit]),
                           "valid_pair_ids_sha256": _ids(train["id"].to_numpy()[pair_valid])})
        incumbent = claude_glm.redistribute(base, specialist, s)
        style = {}
        for name in MODELS:
            for C in C_GRID:
                rows = valid if name == CANDIDATE else pair_valid
                scores, record = fit_scores(inputs[name], target, pair_fit, rows, C)
                fits.append({"stage": "inner", "inner_fold": int(inner_fold), "model": name, **record})
                binary = scores[pair[valid]] if name == CANDIDATE else scores
                inner_binary[name][C].append(_binary_ba(target[pair_valid], binary))
                if name == CANDIDATE:
                    style[C] = scores
        inner_style.append((valid, incumbent, style))
    means = {name: {C: float(np.mean(values)) for C, values in grid.items()} for name, grid in inner_binary.items()}
    chosen_C = {name: first_best(C_GRID, means[name].__getitem__) for name in MODELS}
    a_scores = {a: float(np.mean([_ba(truth[valid], integrate(incumbent, style[chosen_C[CANDIDATE]], a))
                                  for valid, incumbent, style in inner_style])) for a in A_GRID}
    a = first_best(A_GRID, a_scores.__getitem__)
    rows, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
    pair_fit, pair_rows = fit[pair[fit]], rows[pair[rows]]
    specialist = claude_glm.specialist_probabilities(blocks, gemini_pair.INCUMBENT, truth, fit, rows, frozen)
    rebuilt_outer = claude_glm.redistribute(current[rows], specialist, s)
    difference = float(np.abs(rebuilt_outer - locked[rows]).max())
    if difference > 1e-12 or (rebuilt_outer.argmax(axis=1) != locked[rows].argmax(axis=1)).any():
        raise ValueError(f"Fold {fold}: the rebuilt incumbent does not match its lock")
    binary, style_all = {}, None
    for name in MODELS:
        scores, record = fit_scores(inputs[name], target, pair_fit, rows if name == CANDIDATE else pair_rows,
                                    chosen_C[name])
        fits.append({"stage": "outer", "model": name, **record})
        if name == CANDIDATE:
            style_all, binary[name] = scores, scores[pair[rows]]
        else:
            binary[name] = scores
    candidate = integrate(locked[rows], style_all, a)
    if a == 0 and not np.array_equal(candidate, locked[rows]):
        raise ValueError("a = 0 must return the incumbent vectors unchanged")
    return {"fold": fold, "rows": rows, "pair_rows": pair_rows, "binary": binary, "style_all": style_all,
            "candidate": candidate,
            "record": {"outer_fold": fold, "s": s, "s_inner_mean_balanced_accuracy": {str(k): v for k, v in
                                                                                     s_scores.items()},
                       "inner_binary_balanced_accuracy": {name: {f"{C:g}": values for C, values in grid.items()}
                                                          for name, grid in inner_binary.items()},
                       "inner_mean_binary_balanced_accuracy": {name: {f"{C:g}": value for C, value in grid.items()}
                                                               for name, grid in means.items()},
                       "selected_C": chosen_C, "a_inner_mean_balanced_accuracy": {f"{k:g}": v for k, v in
                                                                                  a_scores.items()},
                       "selected_a": a, "incumbent_lock_max_abs_difference": difference,
                       "outer_rows": int(len(rows)), "outer_pair_rows": int(len(pair_rows)),
                       "outer_fit_pair_rows": int(len(pair_fit)), "inner_partitions": partitions, "fits": fits}}


# ---------------------------------------------------------------------------------------------
# Reports


def binary_metrics(target: np.ndarray, scores: np.ndarray, reference: np.ndarray | None = None) -> dict:
    predicted = (scores > 0.5).astype(int)
    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    result = {"rows": int(len(target)), "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
              "accuracy": float((predicted == target).mean()),
              "recall": {PAIR[0]: float((predicted[target == 0] == 0).mean()),
                         PAIR[1]: float((predicted[target == 1] == 1).mean())},
              "correct": {PAIR[0]: int((predicted[target == 0] == 0).sum()),
                          PAIR[1]: int((predicted[target == 1] == 1).sum())},
              "brier": float(np.mean((scores - target) ** 2)),
              "log_loss": float(-np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)))}
    if reference is not None:
        before, after = (reference > 0.5).astype(int) == target, predicted == target
        result["versus_incumbent_forced"] = {"gained": int((~before & after).sum()), "lost": int((before & ~after).sum())}
    return result


def _bin_report(target: np.ndarray, bins: np.ndarray, prompts: np.ndarray, predictors: dict) -> dict:
    result = {}
    for _, _, label in BINS:
        mask = bins == label
        counts = {PAIR[0]: int((target[mask] == 0).sum()), PAIR[1]: int((target[mask] == 1).sum())}
        both = all(counts.values())
        result[label] = {"rows": int(mask.sum()), "prompts": int(len(set(prompts[mask]))), "class_rows": counts,
                         "balanced_accuracy": {name: (float(balanced_accuracy_score(target[mask], scores[mask] > 0.5))
                                                      if both else None) for name, scores in predictors.items()}}
    return result


def _binary_bootstrap(target: np.ndarray, first: np.ndarray, second: np.ndarray, groups: np.ndarray) -> dict:
    return ensemble.prompt_bootstrap(target, (first > 0.5).astype(int), (second > 0.5).astype(int), groups,
                                     SAMPLES, SEED)


def system_report(truth: np.ndarray, folds: np.ndarray, groups: np.ndarray, incumbent: np.ndarray,
                  candidate: np.ndarray, gate: str) -> dict:
    first, second = incumbent.argmax(axis=1), candidate.argmax(axis=1)
    systems = {name: ensemble._system_metrics(truth, values, folds) for name, values in
               (("incumbent", first), ("candidate", second))}
    paired = ensemble.prompt_bootstrap(truth, first, second, groups, SAMPLES, SEED)
    delta = snap_tie(systems["candidate"]["balanced_accuracy"] - systems["incumbent"]["balanced_accuracy"])
    fold_changes = [snap_tie(after - before) for after, before in zip(systems["candidate"]["fold_balanced_accuracy"],
                                                                      systems["incumbent"]["fold_balanced_accuracy"])]
    if gate == "prompt":
        passes = delta >= GAIN and paired["delta_95_interval"][0] > 0
    else:
        passes = delta > 0 and sum(change >= 0 for change in fold_changes) >= 3
    gained, lost = (first != truth) & (second == truth), (first == truth) & (second != truth)
    in_pair = lambda values: np.isin(values, PAIR_INDEX)  # noqa: E731
    routing = {label: {"pair_to_other": int(((truth == index) & in_pair(first) & ~in_pair(second)).sum()),
                       "other_to_pair": int(((truth == index) & ~in_pair(first) & in_pair(second)).sum())}
               for index, label in enumerate(LABELS)}
    return {"systems": systems, "paired": paired,
            "gate": {"stage": gate, "delta": delta, "fold_changes": fold_changes,
                     "positive_folds": int(sum(change > 0 for change in fold_changes)),
                     "zero_folds": int(sum(change == 0 for change in fold_changes)),
                     "negative_folds": int(sum(change < 0 for change in fold_changes)),
                     "fold_rows": [int((folds == fold).sum()) for fold in range(5)], "passes": bool(passes)},
            "recall_change": {label: systems["candidate"]["recall"][label] - systems["incumbent"]["recall"][label]
                              for label in LABELS},
            "changes": {"gained": int(gained.sum()), "lost": int(lost.sum()),
                        "by_class": {label: {"gained": int(gained[truth == index].sum()),
                                             "lost": int(lost[truth == index].sum())}
                                     for index, label in enumerate(LABELS)}},
            "swaps": gemini_pair._swaps(truth, first, second), "routing": routing}


# ---------------------------------------------------------------------------------------------
# Provenance, preflight and evaluation


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ids(values) -> str:
    return hashlib.sha256(json.dumps([str(value) for value in values]).encode()).hexdigest()


def settings() -> dict:
    return {"protocol": PROTOCOL, "token_regex": TOKEN_RE.pattern, "word_lists": WORD_LISTS,
            "phrase": "not simply (whitespace-only separator)", "features": FEATURES, "models": MODELS,
            "candidate": CANDIDATE, "C_grid": C_GRID, "a_grid": A_GRID, "seed": SEED,
            "logistic": {"penalty": "l2", "solver": "lbfgs", "max_iter": 2000, "tol": 1e-6, "class_weight": None},
            "threshold": "3.8 when S > 0.5", "tie_tolerance": 1e-12, "bootstrap": {"samples": SAMPLES, "seed": SEED},
            "gain": GAIN, "length_bins": [label for _, _, label in BINS], "references": REFERENCES,
            "descriptive_audit": DESCRIPTIVE}


def _sources(root: Path) -> dict:
    return {str(path.relative_to(root)): _sha256(path) for path in sorted((root / "src" / "ntl").glob("*.py"))}


def _environment(root: Path) -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "uv_lock_sha256": _sha256(root / "uv.lock"),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl")}}


def _stage_inputs(stage: dict) -> list[str]:
    return [stage["plan"], stage["current_oof"], stage["reference"], stage["reference_npz"], *stage["current_inner"],
            *([stage["groups"]] if stage.get("groups") else [])]


def _load_stage(train: pd.DataFrame, stage: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    plan = pd.read_csv(stage["plan"], dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    grouped, plan = claude_glm._grouped(train, plan, Path(stage["groups"]) if stage.get("groups") else None)
    plan = validate_plan(grouped, plan, AdapterConfig())
    folds = plan["outer_fold"].to_numpy()
    current = claude_glm._aligned_current(pd.read_parquet(stage["current_oof"]), grouped, folds)
    cache = pd.concat([pd.read_parquet(path) for path in stage["current_inner"]], ignore_index=True)
    if set(cache["groups_sha256"]) != {claude_glm.groups_sha256(grouped)}:
        raise ValueError("Inner caches were built under a different row grouping")
    reference = json.loads(Path(stage["reference"]).read_text())
    locked = np.load(stage["reference_npz"], allow_pickle=False)
    if (not np.array_equal(locked["ids"], grouped["id"].to_numpy().astype(str))
            or reference["groups_sha256"] != claude_glm.groups_sha256(grouped)):
        raise ValueError("The locked incumbent is for other rows or another grouping")
    return grouped, plan, current, cache, locked["probabilities"]


def _cache_audit(existing: list[str], rebuilt: list[str]) -> dict:
    """The saved inner caches against an independent rebuild by the same code."""
    old = pd.concat([pd.read_parquet(path) for path in existing], ignore_index=True)
    new = pd.concat([pd.read_parquet(path) for path in rebuilt], ignore_index=True)
    key = ["outer_fold", "inner_fold", "id"]
    old, new = old.sort_values(key).reset_index(drop=True), new.sort_values(key).reset_index(drop=True)
    if not old[key + ["groups_sha256"]].equals(new[key + ["groups_sha256"]]):
        raise ValueError("The rebuilt caches cover different rows")
    return {"rows": int(len(old)), "weights_equal": bool((old["weight"].to_numpy() == new["weight"].to_numpy()).all()),
            "probability_max_abs_difference": float(np.abs(old[claude_glm.BLEND].to_numpy()
                                                           - new[claude_glm.BLEND].to_numpy()).max()),
            "rebuilt_sha256": {path: _sha256(path) for path in rebuilt}}


def preflight(root: Path, train_path: Path, blocks_paths: dict, pair_export: Path, stages: dict, output: Path,
              audit: dict[str, list[str]]) -> dict:
    """References, construction audit and feature table; then the frozen manifest. Fits no new model."""
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("The specification is already frozen; it must not be overwritten")
    train = read_train(train_path)
    if fingerprint_frame(train) != "d76a05125fad49d450e61ec43e78ac5fee9d6a291607386a73ce1046511a301a":
        raise ValueError("Official training data changed")
    truth = ensemble._targets(train)
    table = feature_table(train)
    found = descriptive(train, table)
    mismatched = {key: (found[key], value) for key, value in DESCRIPTIVE.items() if found[key] != value}
    if mismatched:
        raise ValueError(f"Descriptive tokenizer audit differs: {mismatched}")
    export = pd.read_csv(pair_export, dtype={"id": str, "prompt_id": str})
    pair = np.isin(truth, PAIR_INDEX)
    if (set(export["id"]) != set(train["id"][pair]) or len(export) != 300
            or not (export.set_index("id").loc[train["id"][pair], "text"].to_numpy() == train["text"][pair].to_numpy()).all()):
        raise ValueError("The pair export does not match the official pair rows")
    blocks = claude_glm._load_blocks(train, blocks_paths["qwen"], blocks_paths["styledistance"])
    frozen = neural.FrozenConfig()
    checks, inputs = {}, {}
    for name, stage in stages.items():
        grouped, plan, current, cache, locked = _load_stage(train, stage)
        reference = json.loads(Path(stage["reference"]).read_text())
        score = _ba(truth, locked)
        expected = REFERENCES["prompt_balanced_accuracy" if name == "prompts" else "topic_balanced_accuracy"]
        if score != expected:
            raise ValueError(f"{name}: locked incumbent scores {score}, not {expected}")
        record = {"balanced_accuracy": score, "s_by_fold": reference["s_by_fold"]}
        if name == "prompts":
            share, zero = binary_share(locked)
            target = (truth == PAIR_INDEX[1]).astype(int)
            forced = int(((share[pair] > 0.5).astype(int) == target[pair]).sum())
            full = int((locked[pair].argmax(axis=1) == truth[pair]).sum())
            if forced != REFERENCES["forced_binary_correct"] or full != REFERENCES["pair_18_class_correct"]:
                raise ValueError(f"Pair references differ: forced {forced}, 18-class {full}")
            record.update(forced_binary_correct=forced, pair_18_class_correct=full, zero_pair_mass_rows=zero)
            folds = plan["outer_fold"].to_numpy()
            reconstruction = []
            for fold in range(5):
                _, s, _ = gemini_pair.inner_incumbents(grouped, blocks, plan, cache, fold, frozen)
                rows, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
                specialist = claude_glm.specialist_probabilities(blocks, gemini_pair.INCUMBENT, truth, fit, rows,
                                                                 frozen)
                rebuilt = claude_glm.redistribute(current[rows], specialist, s)
                reconstruction.append({"fold": fold, "s": s, "max_abs_difference": float(np.abs(rebuilt - locked[rows]).max()),
                                       "argmax_identical": bool((rebuilt.argmax(axis=1) == locked[rows].argmax(axis=1)).all()),
                                       "pair_rows": int(pair[rows].sum()), "rows": int(len(rows))})
            if ([item["s"] for item in reconstruction] != reference["s_by_fold"]
                    or any(item["max_abs_difference"] > 1e-12 or not item["argmax_identical"] for item in reconstruction)):
                raise ValueError("The rebuilt incumbent does not match its lock")
            record["reconstruction"] = reconstruction
        if name in audit:
            record["cache_audit"] = _cache_audit(stage["current_inner"], audit[name])
            if record["cache_audit"]["probability_max_abs_difference"] > 1e-12 or not record["cache_audit"]["weights_equal"]:
                raise ValueError(f"{name}: the saved inner caches are not reproduced by their recipe")
        folds_frame = plan[["id", "prompt_id", "outer_fold"] + [f"inner_{fold}" for fold in range(5)]]
        record["fold_identity_sha256"] = _ids(folds_frame.astype(str).to_numpy().ravel())
        record["outer_fold_rows"] = [int((plan["outer_fold"] == fold).sum()) for fold in range(5)]
        checks[name] = record
        inputs.update({path: _sha256(path) for path in _stage_inputs(stage)})
    inputs.update({str(path): _sha256(path) for path in (train_path, pair_export, *blocks_paths.values())})
    output.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output / "features.parquet", index=False)
    manifest = {"protocol": PROTOCOL, "settings": settings(), "train": str(train_path),
                "train_fingerprint": fingerprint_frame(train),
                "row_order_sha256": _ids(train["id"]), "label_order_sha256": _ids(train["model"]),
                "stages": stages, "blocks": {key: str(value) for key, value in blocks_paths.items()},
                "pair_export": str(pair_export), "input_sha256": inputs, "source_sha256": _sources(root),
                "environment": _environment(root), "features_sha256": _sha256(output / "features.parquet"),
                "descriptive": found, "checks": checks, "marker_selection_history": MARKER_HISTORY,
                "construction": "Inner caches: lora_compare.current_on_split fits the CPU and Qwen heads and the "
                                "blend weight on inner fitting rows only and scores the inner validation fold; "
                                "reproduced independently by a fresh rebuild (cache_audit)."}
    claude_glm._json(output / "manifest.json", manifest)
    return manifest


def _verify_frozen(root: Path, manifest: dict, output: Path) -> None:
    changed = [path for path, digest in manifest["input_sha256"].items() if _sha256(path) != digest]
    changed += [path for path, digest in manifest["source_sha256"].items() if _sha256(root / path) != digest]
    if _sources(root).keys() != manifest["source_sha256"].keys():
        changed.append("source file set")
    if _environment(root) != manifest["environment"]:
        changed.append("environment")
    if json.loads(json.dumps(settings())) != manifest["settings"]:
        changed.append("settings")
    if _sha256(output / "features.parquet") != manifest["features_sha256"]:
        changed.append("features.parquet")
    if changed:
        raise ValueError(f"The frozen specification no longer matches: {changed}")


def evaluate(root: Path, output: Path, stage_name: str) -> dict:
    """The single declared evaluation of one stage, against the frozen manifest."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    _verify_frozen(root, manifest, output)
    directory = output / stage_name
    if (directory / "comparison.json").exists():
        raise ValueError("This stage has already been evaluated; it must not be overwritten")
    if stage_name == "topics8":
        prompts = json.loads((output / "prompts" / "comparison.json").read_text())
        if not prompts["system"]["gate"]["passes"]:
            raise ValueError("The topic stage runs only after the prompt gate passes")
    stage = manifest["stages"][stage_name]
    train = read_train(Path(manifest["train"]))
    blocks = claude_glm._load_blocks(train, manifest["blocks"]["qwen"], manifest["blocks"]["styledistance"])
    grouped, plan, current, cache, locked = _load_stage(train, stage)
    features = pd.read_parquet(output / "features.parquet").set_index("id").loc[grouped["id"]].reset_index()
    truth, folds = ensemble._targets(grouped), plan["outer_fold"].to_numpy()
    groups = grouped["prompt_id"].to_numpy()
    pair = np.isin(truth, PAIR_INDEX)
    target = (truth == PAIR_INDEX[1]).astype(int)
    frozen = neural.FrozenConfig()
    candidate = np.array(locked, copy=True)
    binary = {name: np.full(len(grouped), np.nan) for name in MODELS}
    style_all = np.full(len(grouped), np.nan)
    records = []
    for fold in range(5):
        result = run_fold(grouped, blocks, features, plan, current, cache, locked, fold, frozen)
        candidate[result["rows"]] = result["candidate"]
        style_all[result["rows"]] = result["style_all"]
        for name, scores in result["binary"].items():
            binary[name][result["pair_rows"]] = scores
        records.append(result["record"])
    incumbent_share, zero_incumbent = binary_share(locked)
    candidate_share, zero_candidate = binary_share(candidate)
    predictors = {**{name: values[pair] for name, values in binary.items()},
                  "incumbent_forced": incumbent_share[pair], "integrated_forced": candidate_share[pair]}
    reference = predictors["incumbent_forced"]
    t, g, f = target[pair], groups[pair], folds[pair]
    pooled = {name: binary_metrics(t, scores, None if name == "incumbent_forced" else reference)
              for name, scores in predictors.items()}
    by_fold = {str(fold): {name: binary_metrics(t[f == fold], scores[f == fold]) for name, scores in predictors.items()}
               for fold in range(5)}
    contrasts = {"length+style_vs_length": _binary_bootstrap(t, predictors["length"], predictors["length+style"], g),
                 **{f"{name}_vs_incumbent_forced": _binary_bootstrap(t, reference, predictors[name], g)
                    for name in (*MODELS, "integrated_forced")}}
    bins = features["length_bin"].to_numpy()[pair]
    system = system_report(truth, folds, groups, locked, candidate, "prompt" if stage_name == "prompts" else "topic")
    directory.mkdir(parents=True, exist_ok=True)
    frame = grouped[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"}).assign(outer_fold=folds)
    frame["prediction::incumbent"] = np.asarray(LABELS)[locked.argmax(axis=1)]
    frame["prediction::candidate"] = np.asarray(LABELS)[candidate.argmax(axis=1)]
    frame["style_S"] = style_all
    for name, values in binary.items():
        frame[f"binary::{name}"] = values
    frame.to_parquet(directory / "predictions.parquet", index=False)
    np.savez(directory / "probabilities.npz", ids=grouped["id"].to_numpy().astype(str), incumbent=locked,
             candidate=candidate)
    result = {"protocol": PROTOCOL, "stage": stage_name, "selection": records,
              "binary": {"pooled": pooled, "by_fold": by_fold, "contrasts": contrasts,
                         "principal_contrast": "length+style_vs_length",
                         "length_bins": _bin_report(t, bins, g, predictors),
                         "zero_pair_mass_rows": {"incumbent": zero_incumbent, "candidate": zero_candidate}},
              "system": system, "manifest_sha256": _sha256(output / "manifest.json")}
    claude_glm._json(directory / "comparison.json", result)
    return result


def _stages(args) -> dict:
    stages = {"prompts": {"plan": str(args.plan), "current_oof": str(args.current_oof),
                          "current_inner": [str(path) for path in args.current_inner], "reference": str(args.reference),
                          "reference_npz": str(args.reference.with_suffix(".npz"))}}
    stages["topics8"] = {"plan": str(args.topic_plan), "groups": str(args.topic_groups),
                         "current_oof": str(args.topic_current_oof),
                         "current_inner": [str(path) for path in args.topic_current_inner],
                         "reference": str(args.topic_reference),
                         "reference_npz": str(args.topic_reference.with_suffix(".npz"))}
    return stages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("preflight", help="Check references and freeze the specification; fits nothing new")
    for name in ("train", "pair-export", "qwen-features", "styledistance-features", "plan", "current-oof",
                 "reference", "topic-plan", "topic-groups", "topic-current-oof", "topic-reference"):
        first.add_argument(f"--{name}", type=Path, required=True)
    first.add_argument("--current-inner", type=Path, nargs=5, required=True)
    first.add_argument("--topic-current-inner", type=Path, nargs=5, required=True)
    first.add_argument("--audit-current-inner", type=Path, nargs=5, required=True,
                       help="an independent rebuild of the prompt inner caches")
    first.add_argument("--output", type=Path, required=True)
    second = commands.add_parser("evaluate", help="The single declared evaluation of one stage")
    second.add_argument("--output", type=Path, required=True)
    second.add_argument("--stage", choices=("prompts", "topics8"), required=True)
    args = parser.parse_args(argv)
    root = Path.cwd()
    if args.command == "preflight":
        manifest = preflight(root, args.train, {"qwen": args.qwen_features, "styledistance": args.styledistance_features},
                             args.pair_export, _stages(args), args.output,
                             {"prompts": [str(path) for path in args.audit_current_inner]})
        print(json.dumps({"descriptive": {key: manifest["descriptive"][key] for key in DESCRIPTIVE},
                          "checks": manifest["checks"]}, indent=2))
    else:
        result = evaluate(root, args.output, args.stage)
        system, binary = result["system"], result["binary"]
        print(json.dumps({
            "balanced_accuracy": {name: values["balanced_accuracy"] for name, values in system["systems"].items()},
            "gate": system["gate"], "interval": system["paired"]["delta_95_interval"],
            "selected": [{"fold": item["outer_fold"], "s": item["s"], "C": item["selected_C"], "a": item["selected_a"]}
                         for item in result["selection"]],
            "binary_balanced_accuracy": {name: values["balanced_accuracy"] for name, values in binary["pooled"].items()},
            "principal_contrast": {key: binary["contrasts"]["length+style_vs_length"][key]
                                   for key in ("balanced_accuracy_delta", "delta_95_interval")},
            "swaps": system["swaps"], "changes": {k: v for k, v in system["changes"].items() if k != "by_class"},
        }, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.gemini_markers import main as imported_main

    raise SystemExit(imported_main())
