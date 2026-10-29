"""DeTeCtive authorship embeddings for the GPT classes (docs/detective-gpt.md).

The pinned `Deepfake_best.pth` checkpoint of heyongxin233/DeTeCtive is a state dict for a stock
RoBERTa encoder (princeton-nlp/unsup-simcse-roberta-base), read with `weights_only=True` and
loaded into `transformers` classes: no authors' code and no remote code. Embeddings follow the
authors' computation: passages truncated and padded to 512 tokens, the last hidden states summed
over all positions (the authors' masked_fill result is discarded, so padding counts) and divided
by the number of real tokens, then L2-normalized.

Stage 1 (diagnostic) measures five-way balanced accuracy on the true GPT passages. Stage 2 (the
decision) runs `ntl.gpt_specialist` with Qwen + StyleDistance + DeTeCtive features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from . import claude_glm, ensemble, gpt_specialist, neural
from .constants import LABELS
from .data import read_train
from .frozen_gemini import load_feature_archive, save_feature_archive

MODEL_ID = "heyongxin233/DeTeCtive"
REVISION = "cefc9833290189b414a683d0775edb900c1832c1"
CHECKPOINT = "Deepfake_best.pth"
CHECKPOINT_SHA256 = "31a47928ea17f1fc872312e9896a0952c1dae48dba5660b9594ee5e05f2487e1"
BASE_ID = "princeton-nlp/unsup-simcse-roberta-base"
BASE_REVISION = "db28710348cf9f33a2be25c505f98f0fbbbfe768"
BASE_FILES = {
    "config.json": "b6fc1fad1e8d45ba913343193a86315f544387bc9bf6f85a566892a24a027a3c",
    "merges.txt": "fe36cab26d4f4421ed725e10a2e9ddb7f799449c603a96e7f29b5a3c82a95862",
    "special_tokens_map.json": "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d",
    "tokenizer_config.json": "80afdc1b9ca7c489e4de769812aa78ef23ce2c7b88ef7e968e07555816bac951",
    "vocab.json": "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e",
}
MAX_LENGTH = 512
POOLING = "sum over all padded positions / real tokens (authors' discarded masked_fill)"
BLOCK = "detective"
FEATURE_SET = f"qwen+styledistance+{BLOCK}"
K = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_encoder(directory: str | Path, *, checkpoint_sha256: str = CHECKPOINT_SHA256,
                 base_files: dict[str, str] | None = None):
    """The tokenizer and the checkpoint's RoBERTa encoder, every file checked against its pin."""
    import torch
    from transformers import AutoTokenizer, RobertaConfig, RobertaModel

    directory = Path(directory)
    base = directory / "base"
    for name, digest in (base_files or BASE_FILES).items():
        if _sha256(base / name) != digest:
            raise ValueError(f"{base / name} does not match its pinned hash")
    if _sha256(directory / CHECKPOINT) != checkpoint_sha256:
        raise ValueError(f"{directory / CHECKPOINT} does not match its published hash")
    state = torch.load(directory / CHECKPOINT, map_location="cpu", weights_only=True)
    if any(not key.startswith("model.") for key in state):
        raise ValueError("The checkpoint must hold only the encoder's state dict")
    model = RobertaModel(RobertaConfig.from_json_file(base / "config.json"))
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()}, strict=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    return tokenizer, model


def pool(hidden, attention_mask):
    """The authors' pooling: every position's state summed, divided by the real-token count, normalized."""
    import torch

    summed = hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    return torch.nn.functional.normalize(summed, dim=-1)


def embed(texts: list[str], tokenizer, model, batch_size: int = 16) -> np.ndarray:
    import torch

    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], max_length=MAX_LENGTH, padding="max_length",
                              truncation=True, return_tensors="pt")
            chunks.append(pool(model(**batch).last_hidden_state, batch["attention_mask"]).numpy())
    embeddings = np.concatenate(chunks).astype(np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be finite")
    return embeddings


def metadata() -> dict:
    return {"model_id": MODEL_ID, "model_revision": REVISION, "checkpoint": CHECKPOINT,
            "checkpoint_sha256": CHECKPOINT_SHA256, "base_model_id": BASE_ID, "base_model_revision": BASE_REVISION,
            "encoder_frozen": True,
            "feature_options": {"pooling": POOLING, "max_length": MAX_LENGTH, "normalization": "l2",
                                "padding": "max_length", "dtype": "float32", "device": "cpu"}}


def load_block(path: str | Path, train: pd.DataFrame) -> np.ndarray:
    features, provenance = load_feature_archive(path, train)
    if (provenance.get("model_id"), provenance.get("model_revision"), provenance.get("checkpoint_sha256")) != \
            (MODEL_ID, REVISION, CHECKPOINT_SHA256):
        raise ValueError(f"{path} does not hold the pinned DeTeCtive checkpoint's features")
    return features


def knn(embeddings: np.ndarray, labels: np.ndarray, fit: np.ndarray, rows: np.ndarray, k: int = K) -> np.ndarray:
    """Majority label of each row's k most cosine-similar fitting rows; ties go to the larger summed
    similarity, then to the smaller label."""
    similarity = embeddings[rows] @ embeddings[fit].T
    nearest = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    predictions = np.empty(len(rows), dtype=int)
    for row, neighbours in enumerate(nearest):
        votes = {}
        for neighbour in neighbours:
            count, total = votes.get(labels[fit[neighbour]], (0, 0.0))
            votes[labels[fit[neighbour]]] = (count + 1, total + similarity[row, neighbour])
        predictions[row] = min(votes, key=lambda label: (-votes[label][0], -votes[label][1], label))
    return predictions


def five_way(train: pd.DataFrame, blocks: dict[str, np.ndarray], folds: np.ndarray, locked: np.ndarray,
             frozen: neural.FrozenConfig) -> dict:
    """Stage 1: five-way balanced accuracy and recall on the true GPT passages, each fold's model
    fit on the true GPT passages outside it."""
    truth = ensemble._targets(train)
    gpt = np.isin(truth, gpt_specialist.GPT_INDEX)
    position = np.full(len(LABELS), -1)
    position[gpt_specialist.GPT_INDEX] = np.arange(len(gpt_specialist.GPT))
    predicted = {"incumbent_forced": locked[:, gpt_specialist.GPT_INDEX].argmax(axis=1)}
    for feature_set in (BLOCK, "qwen+styledistance", FEATURE_SET):
        values = np.full(len(train), -1)
        for fold in range(5):
            rows, fit = np.flatnonzero(gpt & (folds == fold)), np.flatnonzero(folds != fold)
            values[rows] = gpt_specialist.specialist(blocks, feature_set, truth, fit, rows, frozen).argmax(axis=1)
        predicted[f"specialist_{feature_set}"] = values
    values = np.full(len(train), -1)
    for fold in range(5):
        rows, fit = np.flatnonzero(gpt & (folds == fold)), np.flatnonzero(gpt & (folds != fold))
        values[rows] = knn(blocks[BLOCK], position[truth], fit, rows)
    predicted[f"{BLOCK}_knn{K}"] = values
    target = position[truth[gpt]]
    return {name: {"balanced_accuracy": float(balanced_accuracy_score(target, values[gpt])),
                   "recall": {label: float((values[gpt][target == index] == index).mean())
                              for index, label in enumerate(gpt_specialist.GPT)}}
            for name, values in predicted.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("extract", help="Embed every training passage with the pinned checkpoint")
    first.add_argument("--train", type=Path, required=True)
    first.add_argument("--model-dir", type=Path, required=True)
    first.add_argument("--output", type=Path, required=True)
    first.add_argument("--threads", type=int, default=8)
    second = commands.add_parser("evaluate", help="Stage 1 diagnostics and the stage 2 GPT specialist")
    for name in ("train", "plan", "qwen-features", "styledistance-features", "detective-features", "current-oof",
                 "reference", "earlier", "output-dir"):
        second.add_argument(f"--{name}", type=Path, required=True)
    second.add_argument("--current-inner", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    train = read_train(args.train)
    if args.command == "extract":
        import torch

        if args.output.exists():
            raise ValueError("The archive already exists; it must not be overwritten")
        torch.set_num_threads(args.threads)
        tokenizer, model = load_encoder(args.model_dir)
        embeddings = embed(train["text"].tolist(), tokenizer, model)
        saved = save_feature_archive(args.output, train["id"].tolist(), embeddings, metadata(),
                                     texts=train["text"].tolist())
        print(json.dumps({"rows": int(len(embeddings)), "dimension": int(embeddings.shape[1]),
                          "embeddings_sha256": saved["embeddings_sha256"]}))
        return 0
    blocks = claude_glm._load_blocks(train, args.qwen_features, args.styledistance_features)
    blocks[BLOCK] = load_block(args.detective_features, train)
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    folds = train["id"].map(plan.set_index("id")["outer_fold"]).to_numpy()
    reference = json.loads(args.reference.read_text())
    locked = np.load(args.reference.with_suffix(".npz"), allow_pickle=False)
    if (not np.array_equal(locked["ids"], train["id"].to_numpy().astype(str))
            or reference["groups_sha256"] != claude_glm.groups_sha256(train)):
        raise ValueError("The locked incumbent is for other rows or another grouping")
    frozen = neural.FrozenConfig()
    stage1 = five_way(train, blocks, folds, locked["probabilities"], frozen)
    cache = pd.concat([pd.read_parquet(path) for path in args.current_inner], ignore_index=True)
    inputs = [args.plan, args.qwen_features, args.styledistance_features, args.detective_features, args.current_oof,
              args.reference, args.reference.with_suffix(".npz"), args.earlier, *args.current_inner]
    result = gpt_specialist.evaluate(train, blocks, plan, pd.read_parquet(args.current_oof), cache,
                                     locked["probabilities"], args.output_dir, feature_sets=(FEATURE_SET,),
                                     provenance={"input_sha256": {str(path): claude_glm._sha256(path)
                                                                  for path in inputs}})
    earlier = pd.read_parquet(args.earlier).set_index("id").loc[train["id"]]
    candidate = pd.read_parquet(Path(args.output_dir) / "predictions.parquet").set_index("id").loc[train["id"]]
    index = {label: position for position, label in enumerate(LABELS)}
    truth = ensemble._targets(train)
    versus = gpt_specialist.compare(truth, folds, train["prompt_id"].to_numpy(),
                                    earlier["prediction::qwen+styledistance_alone"].map(index).to_numpy(),
                                    candidate["prediction::candidate"].map(index).to_numpy(), 10000, 2026)
    claude_glm._json(Path(args.output_dir) / "stage1.json", {"five_way": stage1, "k": K,
                                                             "versus_qwen_styledistance_gpt_specialist": versus})
    print(json.dumps({
        "stage1_five_way_balanced_accuracy": {name: round(value["balanced_accuracy"], 4) for name, value in stage1.items()},
        "balanced_accuracy": {name: round(values["balanced_accuracy"], 4) for name, values in result["systems"].items()},
        "promotion": result["promotion"], "selected": [item["selected"] for item in result["selection"]],
        "versus_qwen_styledistance_gpt_specialist": {"delta": versus["delta"],
                                                     "interval": versus["paired"]["delta_95_interval"]},
        "changes": {key: result["candidate_vs_incumbent"][key] for key in ("gained", "lost")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.detective import main as imported_main

    raise SystemExit(imported_main())
