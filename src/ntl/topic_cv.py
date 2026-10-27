"""Topic-grouped validation: does the CPU-Qwen blend lose more than CPU on unseen topics?

The data carry no prompt text, so each prompt's topic is the content-word TF-IDF of its
18 responses (label-free). Prompts are clustered into K topics, and whole topics are held
out together in both the outer folds and the blend's inner weight-selection folds. Coarser
K means a larger topic distance between training and held-out prompts. Row-level folds
(prompt leakage) and the original prompt folds bracket the topic conditions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import ensemble, neural
from .data import read_train, validate_train
from .folds import build_fold_manifest, save_fold_manifest


def prompt_topics(train: pd.DataFrame, seed: int = 2026, components: int = 60) -> dict:
    """Label-free topic vectors: one TF-IDF document per prompt, from all its responses."""
    train = validate_train(train)
    documents = train.groupby("prompt_id")["text"].apply("\n".join).sort_index()
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", min_df=2, max_df=0.5,
                                 sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    tfidf = vectorizer.fit_transform(documents.to_numpy())
    components = min(components, tfidf.shape[1] - 1, len(documents) - 1)
    latent = normalize(TruncatedSVD(n_components=components, random_state=seed).fit_transform(tfidf))
    similarity = (tfidf @ tfidf.T).toarray()
    np.fill_diagonal(similarity, 0.0)
    return {"prompts": documents.index.to_numpy(), "tfidf": tfidf, "latent": latent,
            "similarity": similarity, "terms": np.asarray(vectorizer.get_feature_names_out())}


def topic_clusters(topics: dict, k: int, seed: int = 2026) -> dict[str, str]:
    labels = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(topics["latent"])
    return {prompt: f"topic{k}-{label:02d}" for prompt, label in zip(topics["prompts"], labels, strict=True)}


def describe_clusters(topics: dict, groups: dict[str, str], top: int = 8) -> dict[str, dict]:
    prompts = topics["prompts"]
    result = {}
    for group in sorted(set(groups.values())):
        mask = np.asarray([groups[p] == group for p in prompts])
        centroid = np.asarray(topics["tfidf"][mask].mean(axis=0)).ravel()
        result[group] = {"prompts": int(mask.sum()), "terms": topics["terms"][np.argsort(-centroid)[:top]].tolist()}
    return result


def regroup(train: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """Replace prompt_id by its group so every outer and inner split holds out whole groups."""
    regrouped = train.copy()
    regrouped["prompt_id"] = regrouped["prompt_id"].map(groups)
    if regrouped["prompt_id"].isna().any():
        raise ValueError("Every prompt needs a group")
    return regrouped


def nearest_training_similarity(topics: dict, prompt_folds: dict[str, int]) -> np.ndarray:
    """Per prompt, the topic cosine to its most similar prompt in the training partition."""
    folds = np.asarray([prompt_folds[p] for p in topics["prompts"]])
    return np.asarray([topics["similarity"][index, folds != folds[index]].max()
                       for index in range(len(folds))])


def _prompt_folds(train: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, int]:
    folds = train[["id", "prompt_id"]].merge(manifest[["id", "fold"]], on="id", validate="one_to_one")
    per_prompt = folds.groupby("prompt_id")["fold"].agg(["min", "max"])
    if (per_prompt["min"] != per_prompt["max"]).any():
        return {}  # Row-level folds split prompts; nearest training prompt is the prompt itself.
    return per_prompt["min"].to_dict()


def similarity_bins(train: pd.DataFrame, oof: pd.DataFrame, topics: dict, prompt_folds: dict[str, int]) -> dict:
    """On one condition's OOF: does each system's accuracy depend on topic distance?"""
    nearest = pd.Series(nearest_training_similarity(topics, prompt_folds), index=topics["prompts"])
    frame = oof.assign(nearest=oof["id"].map(train.set_index("id")["prompt_id"]).map(nearest))
    for name in ("cpu", "qwen", "blend"):
        frame[f"{name}_correct"] = frame[f"prediction::{name}"] == frame["true_model"]
    frame["tercile"] = pd.qcut(frame["nearest"], 3, labels=["least similar", "middle", "most similar"])
    bins = {str(tercile): {"rows": len(group), "median_similarity": float(group["nearest"].median()),
                           **{name: float(group[f"{name}_correct"].mean()) for name in ("cpu", "qwen", "blend")}}
            for tercile, group in frame.groupby("tercile", observed=True)}
    per_prompt = frame.assign(prompt=frame["id"].map(train.set_index("id")["prompt_id"])).groupby("prompt").agg(
        nearest=("nearest", "first"), cpu=("cpu_correct", "mean"), qwen=("qwen_correct", "mean"),
        blend=("blend_correct", "mean"))
    def correlation(values):
        # Undefined (None) when a system scores every prompt identically.
        if values.nunique() < 2:
            return None
        return float(spearmanr(per_prompt["nearest"], values).statistic)

    return {"terciles": bins, "spearman_nearest_similarity_vs": {
        "qwen_minus_cpu": correlation(per_prompt["qwen"] - per_prompt["cpu"]),
        "blend_minus_cpu": correlation(per_prompt["blend"] - per_prompt["cpu"]),
        "cpu": correlation(per_prompt["cpu"]), "qwen": correlation(per_prompt["qwen"])}}


def run(train: pd.DataFrame, features: np.ndarray, output_dir: str | Path, ks: tuple[int, ...] = (30, 15, 8),
        include_rows: bool = True, config: ensemble.BlendConfig | None = None) -> dict:
    config = config or ensemble.BlendConfig()
    train = validate_train(train).reset_index(drop=True)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory; an existing experiment must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    topics = prompt_topics(train, config.seed)
    conditions = ([("rows", dict(zip(train["prompt_id"], train["prompt_id"])), True)] if include_rows else [])
    conditions.append(("prompts", {p: p for p in topics["prompts"]}, False))
    conditions += [(f"topics-{k}", topic_clusters(topics, k, config.seed), False) for k in ks]
    summary = {"configuration": asdict(config), "conditions": {}}
    for name, groups, by_row in conditions:
        frame = train.assign(prompt_id=train["id"]) if by_row else regroup(train, groups)
        manifest, metadata = build_fold_manifest(frame, n_splits=5, seed=config.seed)
        directory = output / name
        manifest_path = save_fold_manifest(manifest, metadata, directory / "folds")
        comparison = ensemble.evaluate_blend(frame, features, manifest_path, directory / "blend", config)
        prompt_folds = {} if by_row else _prompt_folds(train, manifest)
        nearest = (np.ones(len(topics["prompts"])) if by_row
                   else nearest_training_similarity(topics, prompt_folds))
        record = {
            "groups": int(frame["prompt_id"].nunique()),
            "median_nearest_training_prompt_similarity": float(np.median(nearest)),
            "balanced_accuracy": {system: values["balanced_accuracy"] for system, values in comparison["systems"].items()},
            "fold_balanced_accuracy": {system: values["fold_balanced_accuracy"]
                                       for system, values in comparison["systems"].items()},
            "selected_weight_per_outer_fold": comparison["selected_weight_per_outer_fold"],
            "blend_vs_cpu": comparison["paired"]["blend_vs_cpu"],
            "qwen_vs_cpu": comparison["paired"]["qwen_vs_cpu"],
        }
        if not by_row:
            record["clusters"] = describe_clusters(topics, groups) if name != "prompts" else None
            record["by_topic_distance"] = similarity_bins(
                train, pd.read_parquet(directory / "blend" / "oof.parquet"), topics, prompt_folds)
        summary["conditions"][name] = record
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"condition": name, "median_nearest_similarity": record["median_nearest_training_prompt_similarity"],
                          **record["balanced_accuracy"], "weights": record["selected_weight_per_outer_fold"]}),
              flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[30, 15, 8])
    args = parser.parse_args(argv)
    train = read_train(args.train)
    features, _ = ensemble.load_archive(args.features, train, neural.FrozenConfig())
    run(train, features, args.output_dir, tuple(args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
