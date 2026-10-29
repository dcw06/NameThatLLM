"""GLM 5.3 against GLM 5.3 Flash, with matched external passages (docs/glm-pair.md, stage 1).

Two binary logistic regressions on the existing CPU features and regularization: a control
fit on the official training passages of the pair, and the same plus matched external
passages (official Z.AI provider for both versions, shared prompts, equal counts per prompt,
verified identifiers, topic-linked prompts only with their competition prompt on the training
side). Both are scored on the official validation passages of the original five prompt folds
and compared with the incumbent's forced choice between the two versions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from . import claude_glm, ensemble, external_data
from .constants import LABELS
from .data import read_train
from .modeling import BaselineConfig, build_pipeline

PAIR = ("glm-5.3", "glm-5.3-flash")
PAIR_INDEX = np.array([LABELS.index(label) for label in PAIR])
SOURCE_TAG, PROVIDER = "zaiglm", "Z.AI"
SAMPLES, SEED = 10000, 2026


def matched_external(external: pd.DataFrame, links: pd.Series) -> pd.DataFrame:
    """Verified Z.AI passages of both versions from shared prompts, the same number per version per
    prompt (the first by row_id), with each row's topic link ("" when none)."""
    rows = external[(external["source_tag"] == SOURCE_TAG) & (external["provider"] == PROVIDER)
                    & external["model"].isin(PAIR)]
    rows = rows[external_data.verified(rows)].sort_values("row_id")
    chosen = []
    for _, group in rows.groupby("prompt_id"):
        by_version = [group[group["model"] == label] for label in PAIR]
        count = min(len(part) for part in by_version)
        chosen.extend(part.iloc[:count] for part in by_version if count)
    if not chosen:
        return rows.iloc[:0].assign(link="")
    matched = pd.concat(chosen).sort_values(["prompt_id", "model", "row_id"]).reset_index(drop=True)
    return matched.assign(link=matched["row_id"].map(links).fillna("").to_numpy())


def eligible(matched: pd.DataFrame, prompt_fold: pd.Series, fold: int) -> np.ndarray:
    """Matched rows usable when fitting for `fold`: unlinked, or linked to a training-side prompt."""
    linked_fold = matched["link"].map(prompt_fold)
    return ((matched["link"] == "") | (linked_fold.notna() & (linked_fold != fold))).to_numpy()


def fit_scores(texts: np.ndarray, targets: np.ndarray, rows_texts: np.ndarray, seed: int = SEED) -> np.ndarray:
    """P(Flash) from the CPU baseline's features and classifier, fit on `texts`, `targets` (1 = Flash)."""
    if set(targets) != {0, 1}:
        raise ValueError("Both GLM versions need training passages")
    model = build_pipeline(BaselineConfig(seed=seed))
    with threadpool_limits(limits=2):
        model.fit(texts, targets)
        scores = model.predict_proba(rows_texts)
    if not np.array_equal(model.named_steps["classifier"].classes_, [0, 1]) or not np.isfinite(scores).all():
        raise ValueError("The pair classifier must give finite probabilities for both versions")
    return scores[:, 1]


def summary(target: np.ndarray, scores: np.ndarray) -> dict:
    predicted = (scores > 0.5).astype(int)
    return {"balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
            "accuracy": float((predicted == target).mean()),
            "recall": {PAIR[0]: float((predicted[target == 0] == 0).mean()),
                       PAIR[1]: float((predicted[target == 1] == 1).mean())},
            "swaps": {f"{PAIR[0]}->{PAIR[1]}": int(((target == 0) & (predicted == 1)).sum()),
                      f"{PAIR[1]}->{PAIR[0]}": int(((target == 1) & (predicted == 0)).sum())}}


def paired(target: np.ndarray, first: np.ndarray, second: np.ndarray, groups: np.ndarray) -> dict:
    """Second against first: corrected / introduced and the whole-prompt bootstrap of balanced accuracy."""
    right_first, right_second = (first > 0.5).astype(int) == target, (second > 0.5).astype(int) == target
    bootstrap = ensemble.prompt_bootstrap(target, (first > 0.5).astype(int), (second > 0.5).astype(int), groups,
                                          SAMPLES, SEED)
    return {"corrected": int((~right_first & right_second).sum()), "introduced": int((right_first & ~right_second).sum()),
            "delta": bootstrap["balanced_accuracy_delta"], "interval": bootstrap["delta_95_interval"],
            "fraction_at_most_zero": bootstrap["fraction_delta_at_most_zero"]}


def evaluate(train: pd.DataFrame, plan: pd.DataFrame, matched: pd.DataFrame, locked: np.ndarray,
             output_dir: str | Path) -> dict:
    output = Path(output_dir)
    if (output / "comparison.json").exists():
        raise ValueError("Use a new output directory; an existing comparison must not be overwritten")
    truth = ensemble._targets(train)
    folds = train["id"].map(plan.set_index("id")["outer_fold"]).to_numpy()
    prompt_fold = pd.Series(folds, index=train["prompt_id"].to_numpy()).groupby(level=0).first()
    pair = np.isin(truth, PAIR_INDEX)
    target = (truth == PAIR_INDEX[1]).astype(int)
    texts = train["text"].to_numpy(dtype=object)
    external_texts = matched["text"].to_numpy(dtype=object)
    external_target = (matched["model"].to_numpy() == PAIR[1]).astype(int)
    control, augmented = np.full(len(train), np.nan), np.full(len(train), np.nan)
    usage = []
    for fold in range(5):
        fit, rows = np.flatnonzero(pair & (folds != fold)), np.flatnonzero(pair & (folds == fold))
        if set(train["prompt_id"].to_numpy()[fit]) & set(train["prompt_id"].to_numpy()[rows]):
            raise ValueError("A prompt is on both sides of a fold")
        use = eligible(matched, prompt_fold, fold)
        control[rows] = fit_scores(texts[fit], target[fit], texts[rows])
        augmented[rows] = fit_scores(np.concatenate([texts[fit], external_texts[use]]),
                                     np.concatenate([target[fit], external_target[use]]), texts[rows])
        usage.append({"fold": fold, "official_rows": int(len(fit)), "external_rows": int(use.sum()),
                      "external_by_version": {label: int((use & (matched["model"].to_numpy() == label)).sum())
                                              for label in PAIR},
                      "held_back_linked_rows": int((~use).sum())})
    mass = locked[:, PAIR_INDEX].sum(axis=1)
    incumbent = locked[:, PAIR_INDEX[1]] / mass
    t, groups, f = target[pair], train["prompt_id"].to_numpy()[pair], folds[pair]
    predictors = {"incumbent_forced": incumbent[pair], "control": control[pair], "augmented": augmented[pair]}
    direct = (np.isin(locked.argmax(axis=1), PAIR_INDEX) & (locked.argmax(axis=1) != truth))[pair]
    result = {
        "pair": list(PAIR), "rows": int(pair.sum()), "external": usage,
        "pooled": {name: summary(t, scores) for name, scores in predictors.items()},
        "by_fold": {str(fold): {name: summary(t[f == fold], scores[f == fold]) for name, scores in predictors.items()}
                    for fold in range(5)},
        "augmented_vs_control": paired(t, predictors["control"], predictors["augmented"], groups),
        "augmented_vs_incumbent_forced": paired(t, predictors["incumbent_forced"], predictors["augmented"], groups),
        "control_vs_incumbent_forced": paired(t, predictors["incumbent_forced"], predictors["control"], groups),
        # The incumbent's direct 18-class swaps between the two versions, and how many each model calls right.
        "incumbent_direct_swaps": {"count": int(direct.sum()),
                                   **{name: int(((scores[direct] > 0.5).astype(int) == t[direct]).sum())
                                      for name, scores in predictors.items() if name != "incumbent_forced"}},
    }
    pooled = result["pooled"]
    result["gate"] = {"rule": "augmented binary balanced accuracy above both the control and the incumbent's forced choice",
                      "passes": bool(pooled["augmented"]["balanced_accuracy"] > pooled["control"]["balanced_accuracy"]
                                     and pooled["augmented"]["balanced_accuracy"]
                                     > pooled["incumbent_forced"]["balanced_accuracy"])}
    output.mkdir(parents=True, exist_ok=True)
    frame = train.loc[pair, ["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    frame["outer_fold"] = folds[pair]
    for name, scores in predictors.items():
        frame[f"p_flash::{name}"] = scores
    frame.to_parquet(output / "scores.parquet", index=False)
    claude_glm._json(output / "comparison.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "plan", "external", "pool", "reference", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    train = read_train(args.train)
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    pool = pd.read_parquet(args.pool).set_index("id")
    matched = matched_external(external_data.load_external(args.external), pool["link"].fillna(""))
    reference = json.loads(args.reference.read_text())
    locked = np.load(args.reference.with_suffix(".npz"), allow_pickle=False)
    if (not np.array_equal(locked["ids"], train["id"].to_numpy().astype(str))
            or reference["groups_sha256"] != claude_glm.groups_sha256(train)):
        raise ValueError("The locked incumbent is for other rows or another grouping")
    result = evaluate(train, plan, matched, locked["probabilities"], args.output_dir)
    print(json.dumps({"matched_external": {"rows": len(matched), "by_version": matched["model"].value_counts().to_dict(),
                                           "linked": int((matched["link"] != "").sum())},
                      "pooled": {name: {key: value[key] for key in ("balanced_accuracy", "recall", "swaps")}
                                 for name, value in result["pooled"].items()},
                      "augmented_vs_control": result["augmented_vs_control"],
                      "augmented_vs_incumbent_forced": result["augmented_vs_incumbent_forced"],
                      "incumbent_direct_swaps": result["incumbent_direct_swaps"], "gate": result["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.glm_pair import main as imported_main

    raise SystemExit(imported_main())
