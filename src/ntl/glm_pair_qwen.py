"""GLM 5.3 against GLM Flash on Qwen + StyleDistance, with the 316 matched external passages
(docs/glm-pair-qwen.md).

The control and augmented binary classifiers use the Claude-GLM specialist's recipe on identical
Qwen + StyleDistance features; external passages enter training only, under the same matching,
equal counts per version and topic-link exclusions as the CPU test (`ntl.glm_pair`). Scores are
on the official validation passages of the original five prompt folds.

The external passages' StyleDistance features are extracted here exactly as the official ones
were (CPU, FP32), after a bit-for-bit reproduction check; their Qwen features come from the
private Kaggle kernel built by scripts/prepare_external_qwen_features.py, which also re-extracts
16 official passages so that environment can be compared with the cached features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import claude_glm, ensemble, external_data, glm_pair, neural
from .data import read_train
from .encoders import extract_embeddings
from .frozen_gemini import load_feature_archive, save_feature_archive

REFERENCE_ROWS = 32  # the first official passages, re-extracted in their original batches
REFERENCE_TOLERANCE = 1e-6
WIDTHS = (8192, 768)


def specialist_scores(features: np.ndarray, targets: np.ndarray, rows: np.ndarray, frozen: neural.FrozenConfig) -> np.ndarray:
    """P(Flash) from the Claude-GLM specialist's recipe (Qwen block, then StyleDistance scaled to it)."""
    if set(targets) != {0, 1}:
        raise ValueError("Both GLM versions need training passages")
    head = claude_glm.specialist_head(WIDTHS, frozen)
    with threadpool_limits(limits=2):
        head.fit(features, targets)
        scores = head.predict_proba(rows)
    if not np.array_equal(head.classes_, [0, 1]) or not np.isfinite(scores).all():
        raise ValueError("The pair classifier must give finite probabilities for both versions")
    return scores[:, 1]


def evaluate(train: pd.DataFrame, plan: pd.DataFrame, official: np.ndarray, matched: pd.DataFrame,
             external: np.ndarray, locked: np.ndarray, output_dir: str | Path) -> dict:
    """`official` and `external` are hstacked Qwen + StyleDistance features, aligned to `train` and `matched`."""
    output = Path(output_dir)
    if (output / "comparison.json").exists():
        raise ValueError("Use a new output directory; an existing comparison must not be overwritten")
    if official.shape[1] != sum(WIDTHS) or external.shape != (len(matched), sum(WIDTHS)):
        raise ValueError("Features must be Qwen (8,192) then StyleDistance (768) for every row")
    frozen = neural.FrozenConfig()
    truth = ensemble._targets(train)
    folds = train["id"].map(plan.set_index("id")["outer_fold"]).to_numpy()
    prompt_fold = pd.Series(folds, index=train["prompt_id"].to_numpy()).groupby(level=0).first()
    pair = np.isin(truth, glm_pair.PAIR_INDEX)
    target = (truth == glm_pair.PAIR_INDEX[1]).astype(int)
    external_target = (matched["model"].to_numpy() == glm_pair.PAIR[1]).astype(int)
    control, augmented = np.full(len(train), np.nan), np.full(len(train), np.nan)
    usage = []
    for fold in range(5):
        fit, rows = np.flatnonzero(pair & (folds != fold)), np.flatnonzero(pair & (folds == fold))
        if set(train["prompt_id"].to_numpy()[fit]) & set(train["prompt_id"].to_numpy()[rows]):
            raise ValueError("A prompt is on both sides of a fold")
        use = glm_pair.eligible(matched, prompt_fold, fold)
        control[rows] = specialist_scores(official[fit], target[fit], official[rows], frozen)
        augmented[rows] = specialist_scores(np.vstack([official[fit], external[use]]),
                                            np.concatenate([target[fit], external_target[use]]), official[rows], frozen)
        usage.append({"fold": fold, "official_rows": int(len(fit)), "external_rows": int(use.sum()),
                      "external_by_version": {label: int((use & (matched["model"].to_numpy() == label)).sum())
                                              for label in glm_pair.PAIR},
                      "held_back_linked_rows": int((~use).sum())})
    share = locked[:, glm_pair.PAIR_INDEX[1]] / locked[:, glm_pair.PAIR_INDEX].sum(axis=1)
    t, groups, f = target[pair], train["prompt_id"].to_numpy()[pair], folds[pair]
    predictors = {"incumbent_forced": share[pair], "control": control[pair], "augmented": augmented[pair]}
    direct = (np.isin(locked.argmax(axis=1), glm_pair.PAIR_INDEX) & (locked.argmax(axis=1) != truth))[pair]
    versus_control = glm_pair.paired(t, predictors["control"], predictors["augmented"], groups)
    pooled = {name: glm_pair.summary(t, scores) for name, scores in predictors.items()}
    result = {
        "pair": list(glm_pair.PAIR), "rows": int(pair.sum()), "external": usage, "pooled": pooled,
        "by_fold": {str(fold): {name: glm_pair.summary(t[f == fold], scores[f == fold])
                                for name, scores in predictors.items()} for fold in range(5)},
        "augmented_vs_control": versus_control,
        "augmented_vs_incumbent_forced": glm_pair.paired(t, predictors["incumbent_forced"], predictors["augmented"], groups),
        "control_vs_incumbent_forced": glm_pair.paired(t, predictors["incumbent_forced"], predictors["control"], groups),
        "incumbent_direct_swaps": {"count": int(direct.sum()),
                                   **{name: int(((scores[direct] > 0.5).astype(int) == t[direct]).sum())
                                      for name, scores in predictors.items() if name != "incumbent_forced"}},
        "gate": {"rule": "augmented beats its control with the 95% interval above zero, and exceeds the "
                         "incumbent's forced choice",
                 "passes": bool(versus_control["interval"][0] > 0
                                and pooled["augmented"]["balanced_accuracy"]
                                > pooled["incumbent_forced"]["balanced_accuracy"])},
    }
    output.mkdir(parents=True, exist_ok=True)
    frame = train.loc[pair, ["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    frame["outer_fold"] = folds[pair]
    for name, scores in predictors.items():
        frame[f"p_flash::{name}"] = scores
    frame.to_parquet(output / "scores.parquet", index=False)
    claude_glm._json(output / "comparison.json", result)
    return result


def _matched(args) -> pd.DataFrame:
    pool = pd.read_parquet(args.pool).set_index("id")
    return glm_pair.matched_external(external_data.load_external(args.external), pool["link"].fillna(""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("styledistance", help="Reproduce official features, then embed the matched passages")
    for name in ("train", "external", "pool", "model-path", "cached", "output"):
        first.add_argument(f"--{name}", type=Path, required=True)
    second = commands.add_parser("evaluate", help="Control against augmented on Qwen + StyleDistance")
    for name in ("train", "plan", "external", "pool", "qwen-features", "styledistance-features",
                 "external-qwen", "external-qwen-reference", "external-styledistance", "reference", "output-dir"):
        second.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    train = read_train(args.train)
    matched = _matched(args)
    if args.command == "styledistance":
        if args.output.exists():
            raise ValueError("The archive already exists; it must not be overwritten")
        cached, _ = load_feature_archive(args.cached, train)
        again, _ = extract_embeddings(train["text"].tolist()[:REFERENCE_ROWS], args.model_path, "styledistance",
                                      device="cpu")
        difference = float(np.abs(again - cached[:REFERENCE_ROWS]).max())
        if difference > REFERENCE_TOLERANCE:
            raise ValueError(f"StyleDistance extraction does not reproduce the cached features ({difference:.2e})")
        embeddings, metadata = extract_embeddings(matched["text"].tolist(), args.model_path, "styledistance",
                                                  device="cpu")
        metadata["reproduction"] = {"official_rows": REFERENCE_ROWS, "max_abs_difference": difference}
        save_feature_archive(args.output, matched["row_id"].tolist(), embeddings, metadata,
                             texts=matched["text"].tolist())
        print(json.dumps({"rows": len(matched), "reproduction_max_abs_difference": difference}))
        return 0
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    blocks = claude_glm._load_blocks(train, args.qwen_features, args.styledistance_features)
    frame = matched.rename(columns={"row_id": "id"})[["id", "text"]]
    external_qwen, provenance = ensemble.load_archive(args.external_qwen, frame, neural.FrozenConfig())
    external_style, _ = load_feature_archive(args.external_styledistance, frame)
    # The Kaggle run's re-extraction of official passages, against their cached features.
    with np.load(args.external_qwen_reference, allow_pickle=False) as archive:
        reference_ids = [str(value) for value in archive["ids"]]
    reference_frame = train.set_index("id").loc[reference_ids].reset_index()[["id", "text"]]
    again, _ = ensemble.load_archive(args.external_qwen_reference, reference_frame, neural.FrozenConfig())
    cached = blocks["qwen"][[train.index[train["id"] == value][0] for value in reference_ids]]
    cosine = (again * cached).sum(axis=1) / (np.linalg.norm(again, axis=1) * np.linalg.norm(cached, axis=1))
    environment = {"rows": len(reference_ids), "max_abs_difference": float(np.abs(again - cached).max()),
                   "relative_difference": float(np.linalg.norm(again - cached) / np.linalg.norm(cached)),
                   "minimum_cosine": float(cosine.min())}
    reference = json.loads(args.reference.read_text())
    locked = np.load(args.reference.with_suffix(".npz"), allow_pickle=False)
    if (not np.array_equal(locked["ids"], train["id"].to_numpy().astype(str))
            or reference["groups_sha256"] != claude_glm.groups_sha256(train)):
        raise ValueError("The locked incumbent is for other rows or another grouping")
    result = evaluate(train, plan, np.hstack([blocks["qwen"], blocks["styledistance"]]), matched,
                      np.hstack([external_qwen, external_style]), locked["probabilities"], args.output_dir)
    result["external_qwen_environment_check"] = environment
    claude_glm._json(Path(args.output_dir) / "comparison.json", result)
    print(json.dumps({"pooled": {name: {key: value[key] for key in ("balanced_accuracy", "recall", "swaps")}
                                 for name, value in result["pooled"].items()},
                      "augmented_vs_control": result["augmented_vs_control"],
                      "augmented_vs_incumbent_forced": result["augmented_vs_incumbent_forced"],
                      "incumbent_direct_swaps": result["incumbent_direct_swaps"], "gate": result["gate"],
                      "external_qwen_device": provenance.get("profile", {}).get("device_name"),
                      "external_qwen_environment_check": environment}, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.glm_pair_qwen import main as imported_main

    raise SystemExit(imported_main())
