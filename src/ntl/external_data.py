"""External training passages for the CPU component (docs/external-data.md, stage A).

Rows of the CC BY 4.0 "Name That LLM: OpenRouter + OpenAI Generations" dataset are used only
when their recorded `served_model` matches their label. Each external prompt whose passages
answer a competition prompt's topic (word TF-IDF cosine above the 95th percentile of the
competition's own cross-prompt maxima) is linked to that prompt, and its passages are used
only when the linked prompt is on the training side of the split being fit. In every fit a
class gains at most as many external passages as it has competition passages, in a fixed
seeded order; the existing class-balanced weighting then equalizes the 18 classes.

The augmented CPU model replaces the CPU fit inside the unchanged nested CPU-Qwen blend (its
weight re-chosen in its own inner splits); the Qwen head and the Claude-GLM specialist are
unchanged, and the specialist's s is re-chosen by the inner folds. Every score is on the
original prompt-grouped validation passages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import claude_glm, ensemble, gemini_pair, gpt_specialist, neural
from .constants import LABELS
from .data import read_train
from .lora import AdapterConfig, validate_plan

SOURCE_SHA256 = "b205a6805ed407910d69a1223b62a2a9b2320c438915221fb9256bb8c80cc268"
SERVED = {"gemini-3.7-flash": "google/gemini-3.7-flash", "glm-5.3": "z-ai/glm-5.3",
          "glm-5.3-flash": "z-ai/glm-5.3-flash", "kimi-k3": "moonshotai/kimi-k3",
          "qwen3.8-max": "qwen/qwen3.8-max-0902", "gpt-5.6-luna": "gpt-5.6-luna", "gpt-5.6-sol": "gpt-5.6-sol",
          "gpt-5.6-terra": "gpt-5.6-terra"}
QUANTILE = 0.95
CAP = 1.0
SEED = 2026


@dataclass(frozen=True)
class Pool:
    """Verified external rows: text, label, linked competition prompt ("" when unlinked), and the
    fixed per-class order in which eligible rows are taken."""
    ids: np.ndarray
    texts: np.ndarray
    labels: np.ndarray
    link: np.ndarray
    rank: np.ndarray

    @staticmethod
    def empty() -> Pool:
        nothing = np.array([], dtype=object)
        return Pool(nothing, nothing, nothing, nothing, np.array([], dtype=int))


def verified(external: pd.DataFrame) -> np.ndarray:
    """Rows whose recorded generating-model identifier matches their label."""
    return (external["served_model"].fillna("").to_numpy() == external["model"].map(SERVED).fillna("-").to_numpy())


def topic_links(train: pd.DataFrame, external: pd.DataFrame, quantile: float = QUANTILE) -> tuple[np.ndarray, dict]:
    """The competition prompt each external row is linked to ("" when none), and the threshold."""
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    vectorizer.fit(pd.concat([train["text"], external["text"]]))
    competition, outside = vectorizer.transform(train["text"]), vectorizer.transform(external["text"])
    prompts = train["prompt_id"].to_numpy()
    within = (competition @ competition.T).toarray()
    within[prompts[:, None] == prompts[None, :]] = -1
    tau = float(np.quantile(within.max(axis=1), quantile))
    similarity = (outside @ competition.T).toarray()
    nearest, best = similarity.argmax(axis=1), similarity.max(axis=1)
    link = np.full(len(external), "", dtype=object)
    frame = pd.DataFrame({"prompt": external["prompt_id"].to_numpy(), "match": prompts[nearest], "best": best})
    for _, group in frame[frame["best"] > tau].groupby("prompt"):
        votes = group.groupby("match")["best"].agg(["size", "max"]).sort_values(["size", "max"], ascending=False)
        link[(frame["prompt"] == group["prompt"].iloc[0]).to_numpy()] = votes.index[0]
    return link, {"tau": tau, "quantile": quantile, "rows_above_tau": int((best > tau).sum()),
                  "linked_rows": int((link != "").sum()), "linked_external_prompts": int(frame[link != ""]["prompt"].nunique()),
                  "linked_competition_prompts": int(len(set(link[link != ""])))}


def build_pool(train: pd.DataFrame, external: pd.DataFrame) -> tuple[Pool, dict]:
    keep = verified(external)
    rows = external[keep].sort_values("row_id").reset_index(drop=True)
    link, stats = topic_links(train, rows)
    rank = np.empty(len(rows), dtype=int)
    rng = np.random.default_rng(SEED)
    for label in sorted(set(rows["model"])):
        members = np.flatnonzero(rows["model"].to_numpy() == label)
        rank[members] = rng.permutation(len(members))
    pool = Pool(rows["row_id"].to_numpy(dtype=object), rows["text"].to_numpy(dtype=object),
                rows["model"].to_numpy(dtype=object), link, rank)
    return pool, {**stats, "verified_rows": int(keep.sum()), "excluded_unverified_rows": int((~keep).sum()),
                  "rows_by_label": rows["model"].value_counts().to_dict()}


def eligible_rows(pool: Pool, fit_labels: np.ndarray, fit_prompts: np.ndarray, cap: float = CAP) -> np.ndarray:
    """The external rows one fit may use: linked rows only with their prompt in the fit, and per class at
    most `cap` times the fit's competition rows, in the fixed order."""
    allowed = (pool.link == "") | np.isin(pool.link, np.unique(fit_prompts))
    chosen = []
    for label in sorted(set(pool.labels)):
        limit = int(np.floor(cap * (fit_labels == label).sum()))
        members = np.flatnonzero(allowed & (pool.labels == label))
        chosen.extend(members[np.argsort(pool.rank[members], kind="stable")][:limit])
    return np.sort(np.asarray(chosen, dtype=int))


def augmented_cpu(pool: Pool, usage: list | None = None, cap: float = CAP):
    """A CPU fit (ensemble.default_cpu's signature) that adds the eligible external rows."""
    def cpu(fit_texts, fit_labels, fit_prompts, texts, config):
        chosen = eligible_rows(pool, np.asarray(fit_labels), np.asarray(fit_prompts), cap)
        if usage is not None:
            usage.append({"fit_rows": int(len(fit_texts)), "external_rows": int(len(chosen)),
                          "by_label": pd.Series(pool.labels[chosen]).value_counts().to_dict()})
        if not len(chosen):
            return ensemble._cpu_probabilities(fit_texts, fit_labels, texts, config)
        return ensemble._cpu_probabilities(np.concatenate([fit_texts, pool.texts[chosen]]),
                                           np.concatenate([fit_labels, pool.labels[chosen]]), texts, config)
    return cpu


def outer_blend(train: pd.DataFrame, features: np.ndarray, folds: np.ndarray, fold: int, cpu,
                config: ensemble.BlendConfig | None = None) -> tuple[pd.DataFrame, float]:
    """One outer fold of the nested CPU-Qwen blend, exactly as ensemble.evaluate_blend fits it."""
    config = config or ensemble.BlendConfig()
    texts, labels = train["text"].to_numpy(dtype=object), train["model"].to_numpy(dtype=object)
    truth, prompts = ensemble._targets(train), train["prompt_id"].to_numpy()
    fit, valid = folds != fold, folds == fold
    weight, _ = ensemble._select_weight(texts[fit], labels[fit], features[fit], truth[fit], prompts[fit], config, fold,
                                        cpu)
    cpu_scores = cpu(texts[fit], labels[fit], prompts[fit], texts[valid], config.cpu)
    qwen = ensemble._qwen_probabilities(features[fit], truth[fit], features[valid], config.qwen)
    frame = pd.DataFrame(ensemble._blend(cpu_scores, qwen, weight), columns=claude_glm.BLEND)
    for name, values in (("fold", fold), ("true_model", labels[valid]), ("prompt_id", prompts[valid]),
                         ("id", train["id"].to_numpy()[valid])):
        frame.insert(0, name, values)
    return frame, weight


def evaluate(train: pd.DataFrame, blocks: dict[str, np.ndarray], plan: pd.DataFrame, current: pd.DataFrame,
             cache: pd.DataFrame, locked: np.ndarray, output_dir: str | Path, *, samples: int = 10000,
             seed: int = 2026, provenance: dict | None = None) -> dict:
    """The complete system on the augmented blend: s re-chosen by the inner folds, the unchanged
    Claude-GLM specialist, against the locked incumbent."""
    frozen = neural.FrozenConfig()
    plan = validate_plan(train, plan, AdapterConfig())
    output = Path(output_dir)
    if (output / "comparison.json").exists():
        raise ValueError("Use a new output directory; an existing comparison must not be overwritten")
    truth, folds = ensemble._targets(train), plan["outer_fold"].to_numpy()
    base = claude_glm._aligned_current(current, train, folds)
    candidate = np.empty_like(locked)
    selection = []
    for fold in range(5):
        _, s, s_scores = gemini_pair.inner_incumbents(train, blocks, plan, cache, fold, frozen)
        rows, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        specialist = claude_glm.specialist_probabilities(blocks, gemini_pair.INCUMBENT, truth, fit, rows, frozen)
        candidate[rows] = claude_glm.redistribute(base[rows], specialist, s)
        selection.append({"outer_fold": fold, "s": s, "s_inner_mean_balanced_accuracy":
                          {str(key): value for key, value in s_scores.items()}})
    predictions = {"incumbent": locked.argmax(axis=1), "candidate": candidate.argmax(axis=1),
                   "augmented_blend_only": base.argmax(axis=1)}
    systems = {name: gpt_specialist._metrics(truth, values, folds) for name, values in predictions.items()}
    report = gpt_specialist.compare(truth, folds, train["prompt_id"].to_numpy(), predictions["incumbent"],
                                    predictions["candidate"], samples, seed)
    delta, interval = report["delta"], report["paired"]["delta_95_interval"]
    output.mkdir(parents=True, exist_ok=True)
    frame = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"}).assign(outer_fold=folds)
    for name, values in predictions.items():
        frame[f"prediction::{name}"] = np.asarray(LABELS)[values]
    frame.to_parquet(output / "predictions.parquet", index=False)
    np.savez(output / "probabilities.npz", ids=train["id"].to_numpy().astype(str), incumbent=locked, candidate=candidate)
    result = {"selection": selection, "systems": systems, "candidate_vs_incumbent": report,
              "promotion": {"gate": "prompt", "delta": delta, "interval": interval,
                            "passes": bool(delta >= 0.005 and interval[0] > 0)},
              "groups_sha256": claude_glm.groups_sha256(train), "provenance": provenance or {}}
    claude_glm._json(output / "comparison.json", result)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_external(path: str | Path) -> pd.DataFrame:
    if _sha256(Path(path)) != SOURCE_SHA256:
        raise ValueError(f"{path} is not the audited release of the external dataset")
    return pd.read_parquet(path)


def save_pool(pool: Pool, path: Path) -> None:
    pd.DataFrame({"id": pool.ids, "text": pool.texts, "model": pool.labels, "link": pool.link,
                  "rank": pool.rank}).to_parquet(path, index=False)


def read_pool(path: Path) -> Pool:
    frame = pd.read_parquet(path)
    return Pool(frame["id"].to_numpy(dtype=object), frame["text"].to_numpy(dtype=object),
                frame["model"].to_numpy(dtype=object), frame["link"].fillna("").to_numpy(dtype=object),
                frame["rank"].to_numpy(dtype=int))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("pool", help="Verify, link and order the external rows")
    for name in ("train", "external", "output-dir"):
        first.add_argument(f"--{name}", type=Path, required=True)
    for name in ("inner", "outer"):
        command = commands.add_parser(name, help=f"Build one outer fold's augmented {name} blend")
        for argument in ("train", "plan", "qwen-features", "pool", "output"):
            command.add_argument(f"--{argument}", type=Path, required=True)
        command.add_argument("--fold", type=int, required=True)
        command.add_argument("--empty-pool", action="store_true", help="no external rows: reproduce the saved blend")
    last = commands.add_parser("evaluate", help="The complete system on the augmented blend")
    for name in ("train", "plan", "qwen-features", "styledistance-features", "current-oof", "reference", "output-dir"):
        last.add_argument(f"--{name}", type=Path, required=True)
    last.add_argument("--current-inner", type=Path, nargs=5, required=True)
    args = parser.parse_args(argv)
    train = read_train(args.train)
    if args.command == "pool":
        output = Path(args.output_dir)
        if (output / "pool.parquet").exists():
            raise ValueError("The pool already exists; it must not be overwritten")
        output.mkdir(parents=True, exist_ok=True)
        pool, stats = build_pool(train, load_external(args.external))
        save_pool(pool, output / "pool.parquet")
        claude_glm._json(output / "pool.json", {**stats, "source_sha256": SOURCE_SHA256, "served": SERVED, "cap": CAP,
                                                "seed": SEED, "pool_sha256": _sha256(output / "pool.parquet")})
        print(json.dumps(stats, indent=1))
        return 0
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    if args.command in ("inner", "outer"):
        if Path(args.output).exists():
            raise ValueError(f"{args.output} exists; it must not be overwritten")
        plan = validate_plan(train, plan, AdapterConfig())
        features, _ = ensemble.load_archive(args.qwen_features, train, neural.FrozenConfig())
        pool = Pool.empty() if args.empty_pool else read_pool(args.pool)
        usage = []
        cpu = augmented_cpu(pool, usage)
        if args.command == "inner":
            frame = claude_glm.build_current_inner(train, features, plan, args.fold, cpu=cpu)
        else:
            frame, weight = outer_blend(train, features, plan["outer_fold"].to_numpy(), args.fold, cpu)
            frame["weight"] = weight
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output, index=False)
        Path(args.output).with_suffix(".json").write_text(json.dumps({"fold": args.fold, "usage": usage}, indent=1))
        print(json.dumps({"fold": args.fold, "rows": len(frame), "fits": len(usage),
                          "external_rows_per_fit": [item["external_rows"] for item in usage]}))
        return 0
    blocks = claude_glm._load_blocks(train, args.qwen_features, args.styledistance_features)
    reference = json.loads(args.reference.read_text())
    locked = np.load(args.reference.with_suffix(".npz"), allow_pickle=False)
    if (not np.array_equal(locked["ids"], train["id"].to_numpy().astype(str))
            or reference["groups_sha256"] != claude_glm.groups_sha256(train)):
        raise ValueError("The locked incumbent is for other rows or another grouping")
    cache = pd.concat([pd.read_parquet(path) for path in args.current_inner], ignore_index=True)
    inputs = [args.plan, args.qwen_features, args.styledistance_features, args.current_oof, args.reference,
              *args.current_inner]
    result = evaluate(train, blocks, plan, pd.read_parquet(args.current_oof), cache, locked["probabilities"],
                      args.output_dir, provenance={"input_sha256": {str(path): _sha256(path) for path in inputs}})
    print(json.dumps({"balanced_accuracy": {name: round(values["balanced_accuracy"], 4)
                                            for name, values in result["systems"].items()},
                      "promotion": result["promotion"], "s": [item["s"] for item in result["selection"]],
                      "changes": {key: result["candidate_vs_incumbent"][key] for key in ("gained", "lost")}}, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.external_data import main as imported_main

    raise SystemExit(imported_main())
