"""READER activation-trajectory fingerprints for the GPT classes (docs/reader-gpt.md).

READER's official Qwen3-8B preset: the response alone (no special tokens, no chat template,
at most 1,024 tokens) is read by `Qwen/Qwen3-8B`, and the output of its second decoder block
(before any final norm) is summarized over the response tokens by the length-normalized
DCT-II modes 0 (DC, the token mean) and 1 (AC). Only the embeddings and the first two blocks
are loaded, from the pinned first weight shard, which is what READER's early exit computes.
No authors' code and no remote code are run. Stage 1 measures five-way separation of the
GPT classes, with READER's own probe and the specialist recipe; stage 2 runs
`ntl.gpt_specialist` with Qwen + StyleDistance + READER features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from . import claude_glm, ensemble, gpt_specialist, neural
from .constants import LABELS
from .data import read_train
from .frozen_gemini import load_feature_archive, save_feature_archive

MODEL_ID = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SHARD = "model-00001-of-00005.safetensors"
SHARD_SHA256 = "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f"
FILES = {
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
LAYER = 2
MAX_LENGTH = 1024
BLOCK = "reader"
FEATURE_SET = f"qwen+styledistance+{BLOCK}"
# READER's canonical probe (training/probe.py: fit_source_probe).
PROBE = {"learning_rate": 1e-3, "max_steps": 40, "schedule_horizon": 100, "c_value": 1.0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_proxy(directory: str | Path, *, layer: int = LAYER, files: dict[str, str] | None = None,
               shard_sha256: str = SHARD_SHA256):
    """The tokenizer and a Qwen3 model holding only the embeddings and the first `layer` blocks."""
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

    directory = Path(directory)
    for name, digest in (files or FILES).items():
        if _sha256(directory / name) != digest:
            raise ValueError(f"{directory / name} does not match its pinned hash")
    if _sha256(directory / SHARD) != shard_sha256:
        raise ValueError(f"{directory / SHARD} does not match its published hash")
    settings = json.loads((directory / "config.json").read_text())
    model = Qwen3Model(Qwen3Config(**{**settings, "num_hidden_layers": layer, "layer_types": None}))
    wanted = ("model.embed_tokens.weight", *(f"model.layers.{index}." for index in range(layer)))
    with safe_open(directory / SHARD, framework="pt") as shard:
        state = {key.removeprefix("model."): shard.get_tensor(key).to(torch.float32)
                 for key in shard.keys() if key.startswith(wanted)}
    loaded = model.load_state_dict(state, strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys) != {"norm.weight"}:
        raise ValueError("The shard must hold the embeddings and every retained block, and nothing else is used")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer, model


def dct_basis(lengths, sequence_length: int):
    """READER's length-normalized DCT-II weights for modes 0 and 1, shape (batch, 2, tokens);
    every response starts at position 0 (the response-only input)."""
    import torch

    positions = torch.arange(sequence_length)[None, :]
    valid = positions < lengths[:, None]
    modes = torch.arange(2)
    phase = torch.pi * (positions.float()[:, None, :] + 0.5) * modes.float()[None, :, None] / \
        lengths.float()[:, None, None]
    basis = torch.cos(phase)
    basis[:, 1:] *= math.sqrt(2.0)
    basis = basis / lengths.float()[:, None, None] * valid[:, None, :]
    return basis * (modes[None, :, None] < lengths[:, None, None])


def last_block_output(model, inputs):
    """The output of the model's last retained block, before the final norm."""
    captured = {}
    handle = model.layers[-1].register_forward_hook(lambda module, args, output: captured.setdefault(
        "hidden", output[0] if isinstance(output, (tuple, list)) else output))
    try:
        model(**inputs, use_cache=False)
    finally:
        handle.remove()
    return captured["hidden"]


def extract(texts: list[str], tokenizer, model, batch_size: int = 1) -> tuple[np.ndarray, dict]:
    """(passages, 2, hidden) DC and AC fingerprints, and token-length statistics."""
    import torch

    output, lengths = [], []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer([text or " " for text in texts[start:start + batch_size]], return_tensors="pt",
                               truncation=True, max_length=MAX_LENGTH, padding=True, add_special_tokens=False)
            ends = inputs["attention_mask"].sum(dim=1)
            hidden = last_block_output(model, inputs)
            output.append(torch.einsum("bqt,btd->bqd", dct_basis(ends, hidden.shape[1]), hidden).numpy())
            lengths.extend(ends.tolist())
    features = np.concatenate(output).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Fingerprints must be finite")
    return features, {"min": int(min(lengths)), "median": float(np.median(lengths)), "max": int(max(lengths)),
                      "at_max_length": int(sum(length >= MAX_LENGTH for length in lengths))}


def reader_probe(features: np.ndarray, labels: np.ndarray, classes: int, fit: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """READER's canonical probe, fit on `fit` and predicting `rows`."""
    import torch

    x = torch.as_tensor(np.asarray(features[fit], dtype=np.float32))
    y = torch.as_tensor(labels[fit], dtype=torch.long)
    mean = x.mean(dim=0, keepdim=True)
    scale = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / scale
    model = torch.nn.Linear(x.shape[1], classes)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=PROBE["learning_rate"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PROBE["schedule_horizon"],
                                                          eta_min=PROBE["learning_rate"] * 0.01)
    penalty = 1.0 / (2.0 * PROBE["c_value"] * len(x))
    for _ in range(PROBE["max_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y) + penalty * model.weight.square().sum()
        loss.backward()
        optimizer.step()
        schedule.step()
    with torch.no_grad():
        standardized = (torch.as_tensor(np.asarray(features[rows], dtype=np.float32)) - mean) / scale
        return model(standardized).argmax(dim=1).numpy()


def metadata() -> dict:
    return {"model_id": MODEL_ID, "model_revision": REVISION, "shard": SHARD, "shard_sha256": SHARD_SHA256,
            "layer": LAYER, "input_form": "r", "encoder_frozen": True,
            "feature_options": {"pooling": "DCT-II modes 0 and 1 over all response tokens (DC then AC)",
                                "max_length": MAX_LENGTH, "normalization": "orthonormal DCT-II divided by sqrt(N)",
                                "special_tokens": False, "dtype": "float32", "device": "cpu",
                                "readout": "output of decoder block 2, before the final norm"}}


def load_block(path: str | Path, train: pd.DataFrame) -> np.ndarray:
    features, provenance = load_feature_archive(path, train)
    if (provenance.get("model_id"), provenance.get("model_revision"), provenance.get("layer")) != \
            (MODEL_ID, REVISION, LAYER):
        raise ValueError(f"{path} does not hold the pinned READER fingerprints")
    return features


def five_way(train: pd.DataFrame, blocks: dict[str, np.ndarray], folds: np.ndarray, locked: np.ndarray,
             frozen: neural.FrozenConfig) -> dict:
    """Stage 1: five-way balanced accuracy and recall on the true GPT passages, each fold's model fit on
    the true GPT passages outside it."""
    truth = ensemble._targets(train)
    gpt = np.isin(truth, gpt_specialist.GPT_INDEX)
    position = np.full(len(LABELS), -1)
    position[gpt_specialist.GPT_INDEX] = np.arange(len(gpt_specialist.GPT))
    width = blocks[BLOCK].shape[1] // 2
    components = {"dc-ac": blocks[BLOCK], "dc": blocks[BLOCK][:, :width], "ac": blocks[BLOCK][:, width:]}
    predicted = {"incumbent_forced": locked[:, gpt_specialist.GPT_INDEX].argmax(axis=1)}
    for name, values in components.items():
        result = np.full(len(train), -1)
        for fold in range(5):
            rows, fit = np.flatnonzero(gpt & (folds == fold)), np.flatnonzero(gpt & (folds != fold))
            result[rows] = reader_probe(values, position[truth], len(gpt_specialist.GPT), fit, rows)
        predicted[f"reader_probe_{name}"] = result
    for feature_set in (BLOCK, "qwen+styledistance", FEATURE_SET):
        result = np.full(len(train), -1)
        for fold in range(5):
            rows, fit = np.flatnonzero(gpt & (folds == fold)), np.flatnonzero(folds != fold)
            result[rows] = gpt_specialist.specialist(blocks, feature_set, truth, fit, rows, frozen).argmax(axis=1)
        predicted[f"specialist_{feature_set}"] = result
    target = position[truth[gpt]]
    return {name: {"balanced_accuracy": float(balanced_accuracy_score(target, values[gpt])),
                   "recall": {label: float((values[gpt][target == index] == index).mean())
                              for index, label in enumerate(gpt_specialist.GPT)}}
            for name, values in predicted.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("extract", help="Fingerprint every training passage with the pinned proxy")
    first.add_argument("--train", type=Path, required=True)
    first.add_argument("--model-dir", type=Path, required=True)
    first.add_argument("--output", type=Path, required=True)
    first.add_argument("--threads", type=int, default=8)
    second = commands.add_parser("evaluate", help="Stage 1 diagnostics and the stage 2 GPT specialist")
    for name in ("train", "plan", "qwen-features", "styledistance-features", "reader-features", "current-oof",
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
        tokenizer, model = load_proxy(args.model_dir)
        features, lengths = extract(train["text"].tolist(), tokenizer, model)
        saved = save_feature_archive(args.output, train["id"].tolist(), features.reshape(len(features), -1),
                                     {**metadata(), "response_token_lengths": lengths}, texts=train["text"].tolist())
        print(json.dumps({"rows": int(len(features)), "shape": list(features.shape), "token_lengths": lengths,
                          "embeddings_sha256": saved["embeddings_sha256"]}))
        return 0
    blocks = claude_glm._load_blocks(train, args.qwen_features, args.styledistance_features)
    blocks[BLOCK] = load_block(args.reader_features, train)
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
    inputs = [args.plan, args.qwen_features, args.styledistance_features, args.reader_features, args.current_oof,
              args.reference, args.reference.with_suffix(".npz"), args.earlier, *args.current_inner]
    result = gpt_specialist.evaluate(train, blocks, plan, pd.read_parquet(args.current_oof), cache,
                                     locked["probabilities"], args.output_dir, feature_sets=(FEATURE_SET,),
                                     provenance={"input_sha256": {str(path): claude_glm._sha256(path)
                                                                  for path in inputs}})
    earlier = pd.read_parquet(args.earlier).set_index("id").loc[train["id"]]
    candidate = pd.read_parquet(Path(args.output_dir) / "predictions.parquet").set_index("id").loc[train["id"]]
    index = {label: position for position, label in enumerate(LABELS)}
    versus = gpt_specialist.compare(ensemble._targets(train), folds, train["prompt_id"].to_numpy(),
                                    earlier["prediction::qwen+styledistance_alone"].map(index).to_numpy(),
                                    candidate["prediction::candidate"].map(index).to_numpy(), 10000, 2026)
    claude_glm._json(Path(args.output_dir) / "stage1.json", {"five_way": stage1, "probe": PROBE,
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
    from ntl.reader_fingerprint import main as imported_main

    raise SystemExit(imported_main())
