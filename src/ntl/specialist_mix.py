"""Does a further frozen encoder add to the Claude-GLM specialist?

The incumbent is the validated system: the current CPU + frozen-Qwen blend whose
Claude-GLM probability mass the Qwen + StyleDistance specialist redistributes at the
weight s its inner folds choose (`claude_glm.evaluate`). A candidate head appends one
more frozen encoder block, block-variance balanced, with the same C. The two seven-class
distributions are mixed, q_mix = (1 - a) q_existing + a q_augmented, and fed through the
same integration at the same s, so a = 0 is the incumbent exactly.

Inner prompt folds of each outer training partition choose s first, as the incumbent
does, then the encoder and a from the unchanged reference and every encoder at
a in {0.25, 0.5, 1}. Ties keep the reference, then the smaller a, then the encoder
listed first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from . import claude_glm, ensemble, neural
from .constants import LABELS
from .data import fingerprint_frame, read_train
from .frozen_gemini import load_feature_archive
from .lora import AdapterConfig, validate_plan
from .selection import first_best
from .style_encoders import ENCODERS

INCUMBENT = "qwen+styledistance"
MIX_WEIGHTS = (0.25, 0.5, 1.0)
REFERENCE = "reference"
# Confusions the experiment targets: (true, predicted).
CONFUSIONS = (("claude-fable-5", "glm-5.3"), ("claude-fable-5", "glm-5.3-flash"),
              ("glm-5.3", "claude-fable-5"), ("glm-5.3-flash", "claude-fable-5"),
              ("glm-5.3", "glm-5.3-flash"), ("glm-5.3-flash", "glm-5.3"))


def option_name(encoder: str | None, weight: float) -> str:
    return REFERENCE if encoder is None or weight == 0 else f"{encoder}@{weight:g}"


def options(encoders: tuple[str, ...]) -> list[tuple[str | None, float]]:
    """In tie order: the reference, then by increasing a, then in the order the encoders are listed."""
    return [(None, 0.0)] + [(encoder, weight) for weight in MIX_WEIGHTS for encoder in encoders]


def mixed(existing: np.ndarray, augmented: dict[str, np.ndarray], encoder: str | None, weight: float) -> np.ndarray:
    if encoder is None or weight == 0:
        return existing
    return (1 - weight) * existing + weight * augmented[encoder]


def _ba(truth: np.ndarray, probabilities: np.ndarray) -> float:
    return float(balanced_accuracy_score(truth, probabilities.argmax(axis=1)))


def _confusions(truth: np.ndarray, predictions: np.ndarray) -> dict[str, dict]:
    result = {}
    for true, predicted in CONFUSIONS:
        rows = truth == LABELS.index(true)
        count = int((predictions[rows] == LABELS.index(predicted)).sum())
        result[f"{true}->{predicted}"] = {"count": count, "share": count / int(rows.sum())}
    return result


def evaluate(train: pd.DataFrame, blocks: dict[str, np.ndarray], encoders: tuple[str, ...], plan: pd.DataFrame,
             current_oof: pd.DataFrame, current_cache: pd.DataFrame, output_dir: str | Path, *,
             frozen: neural.FrozenConfig | None = None, samples: int = 10000, seed: int = 2026,
             provenance: dict | None = None) -> dict:
    frozen = frozen or neural.FrozenConfig()
    if (not encoders or len(set(encoders)) != len(encoders)
            or any(name not in blocks or name in ("qwen", "styledistance") or "+" in name for name in encoders)):
        raise ValueError("Name each added encoder's features once; Qwen and StyleDistance are the incumbent's")
    plan = validate_plan(train, plan, AdapterConfig())
    output = Path(output_dir)
    if (output / "comparison.json").exists():
        raise ValueError("Use a new output directory; an existing comparison must not be overwritten")
    if set(current_cache["groups_sha256"]) != {claude_glm.groups_sha256(train)}:
        raise ValueError("Cached current-system inner predictions were built under a different row grouping")
    output.mkdir(parents=True, exist_ok=True)
    truth, prompts = ensemble._targets(train), train["prompt_id"].to_numpy()
    folds = plan["outer_fold"].to_numpy()
    current = claude_glm._aligned_current(current_oof, train, folds)
    grid = options(encoders)
    incumbent, candidate = np.empty_like(current), np.empty_like(current)
    by_option = {option_name(*option): np.empty_like(current) for option in grid}
    selection = []
    for fold in range(5):
        column = plan[f"inner_{fold}"].to_numpy()
        inner = []
        for inner_fold, (valid, inner_current) in claude_glm._inner_current(current_cache, train, plan,
                                                                            fold).items():
            fit = np.flatnonzero((folds != fold) & (column != inner_fold))
            existing = claude_glm.specialist_probabilities(blocks, INCUMBENT, truth, fit, valid, frozen)
            augmented = {encoder: claude_glm.specialist_probabilities(blocks, f"{INCUMBENT}+{encoder}", truth,
                                                                      fit, valid, frozen)
                         for encoder in encoders}
            inner.append((valid, inner_current, existing, augmented))
        # The incumbent's own choice of s, exactly as claude_glm.evaluate makes it.
        s_scores = {s: float(np.mean([_ba(truth[valid], claude_glm.redistribute(base, existing, s))
                                      for valid, base, existing, _ in inner]))
                    for s in claude_glm.WEIGHTS}
        s = first_best(claude_glm.WEIGHTS, s_scores.__getitem__)
        scores = {option_name(*option): [_ba(truth[valid], claude_glm.redistribute(
            base, mixed(existing, augmented, *option), s)) for valid, base, existing, augmented in inner]
            for option in grid}
        means = {name: float(np.mean(values)) for name, values in scores.items()}
        # Options are in tie order, and the first of equal scores is kept.
        chosen = first_best(grid, lambda option: means[option_name(*option)])
        rows = np.flatnonzero(folds == fold)
        fit = np.flatnonzero(folds != fold)
        existing = claude_glm.specialist_probabilities(blocks, INCUMBENT, truth, fit, rows, frozen)
        augmented = {encoder: claude_glm.specialist_probabilities(blocks, f"{INCUMBENT}+{encoder}", truth, fit,
                                                                  rows, frozen)
                     for encoder in encoders}
        incumbent[rows] = claude_glm.redistribute(current[rows], existing, s)
        candidate[rows] = claude_glm.redistribute(current[rows], mixed(existing, augmented, *chosen), s)
        for option in grid:
            by_option[option_name(*option)][rows] = claude_glm.redistribute(
                current[rows], mixed(existing, augmented, *option), s)
        selection.append({"outer_fold": fold, "s": s, "selected": option_name(*chosen),
                          "inner_mean_balanced_accuracy": means,
                          "inner_balanced_accuracy": scores, "s_inner_mean_balanced_accuracy":
                              {str(key): value for key, value in s_scores.items()}})
    predictions = {"incumbent": incumbent.argmax(axis=1), "candidate": candidate.argmax(axis=1)}
    systems = {name: ensemble._system_metrics(truth, values, folds) for name, values in predictions.items()}
    paired = ensemble.prompt_bootstrap(truth, predictions["incumbent"], predictions["candidate"], prompts,
                                       samples, seed)
    outside = ~np.isin(truth, claude_glm.INDEX)
    gained = (predictions["incumbent"] != truth) & (predictions["candidate"] == truth)
    lost = (predictions["incumbent"] == truth) & (predictions["candidate"] != truth)
    frame = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"}).assign(outer_fold=folds)
    for name, values in predictions.items():
        frame[f"prediction::{name}"] = np.asarray(LABELS)[values]
    frame.to_parquet(output / "predictions.parquet", index=False)
    result = {
        "incumbent": INCUMBENT, "encoders": list(encoders), "mix_weights": list(MIX_WEIGHTS),
        "systems": systems, "paired": paired, "selection": selection,
        "specialist_recall": {name: {label: systems[name]["recall"][label] for label in claude_glm.SPECIALIST}
                              for name in systems},
        "confusions": {name: _confusions(truth, values) for name, values in predictions.items()},
        "changes": {"gained": int(gained.sum()), "lost": int(lost.sum()),
                    "gained_outside_specialist": int((gained & outside).sum()),
                    "lost_outside_specialist": int((lost & outside).sum())},
        # Each option applied in every fold: a diagnostic, never a selection.
        "option_balanced_accuracy": {name: _ba(truth, values) for name, values in by_option.items()},
        "groups": int(train["prompt_id"].nunique()), "groups_sha256": claude_glm.groups_sha256(train),
        "train_fingerprint": fingerprint_frame(train), "provenance": provenance or {},
    }
    claude_glm._json(output / "comparison.json", result)
    return result


def load_encoder(name: str, path: Path, train: pd.DataFrame) -> np.ndarray:
    """An added encoder's archive, which must come from that pinned encoder."""
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder {name!r}; expected one of {sorted(ENCODERS)}")
    features, metadata = load_feature_archive(path, train)
    spec = ENCODERS[name]
    if (metadata.get("model_id"), metadata.get("model_revision")) != (spec["model_id"], spec["model_revision"]):
        raise ValueError(f"{path} does not hold {name} features at the pinned revision")
    return features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--qwen-features", type=Path, required=True)
    parser.add_argument("--styledistance-features", type=Path, required=True)
    parser.add_argument("--encoder", nargs=2, action="append", metavar=("NAME", "ARCHIVE"), required=True,
                        help="an added encoder's feature archive, e.g. luar artifacts/.../luar-train.npz")
    parser.add_argument("--current-oof", type=Path, required=True)
    parser.add_argument("--current-inner", type=Path, nargs="+", required=True)
    parser.add_argument("--groups", type=Path, help="CSV of id, prompt_id regrouping rows (e.g. by topic)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    train = read_train(args.train)
    blocks = claude_glm._load_blocks(train, args.qwen_features, args.styledistance_features)
    for name, path in args.encoder:
        blocks[name] = load_encoder(name, Path(path), train)
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    train, plan = claude_glm._grouped(train, plan, args.groups)
    cache = pd.concat([pd.read_parquet(path) for path in args.current_inner], ignore_index=True)
    inputs = [args.plan, args.qwen_features, args.styledistance_features, args.current_oof, *args.current_inner,
              *(Path(path) for _, path in args.encoder), *([args.groups] if args.groups else [])]
    result = evaluate(train, blocks, tuple(name for name, _ in args.encoder), plan,
                      pd.read_parquet(args.current_oof), cache, args.output_dir,
                      provenance={"input_sha256": {str(path): claude_glm._sha256(path) for path in inputs}})
    paired = result["paired"]
    print(json.dumps({
        "balanced_accuracy": {name: round(values["balanced_accuracy"], 4) for name, values in result["systems"].items()},
        "delta": round(paired["balanced_accuracy_delta"], 4),
        "delta_95_interval": [round(bound, 4) for bound in paired["delta_95_interval"]],
        "selected": [item["selected"] for item in result["selection"]], "s": [item["s"] for item in result["selection"]],
        "changes": result["changes"],
        "option_balanced_accuracy": {name: round(value, 4) for name, value in result["option_balanced_accuracy"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.specialist_mix import main as imported_main

    raise SystemExit(imported_main())
