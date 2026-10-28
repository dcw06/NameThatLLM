"""Frozen LUAR-MUD and LUSR passage embeddings for the Claude-GLM specialist experiment.

Readouts, fixed before validation (docs/claude-glm.md, Added encoders):
- Each passage is encoded on its own; for LUAR that is one passage per episode.
- At most 512 tokens per passage (no training passage exceeds 387, so none is truncated).
- Masked mean over the last hidden state, special tokens included, as both models pool.
- LUAR then applies its trained 768 -> 512 projection. With one passage per episode its
  episode attention and max are identities, so this is its whole forward pass, built
  here from the checkpoint's weights: no remote code and no einops are needed offline.
- Every embedding is L2-normalized, as both models are compared by cosine similarity.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

from .data import fingerprint_frame, read_train
from .frozen_gemini import save_feature_archive

ENCODERS = {
    "luar-mud": {"model_id": "rrivera1849/LUAR-MUD", "model_revision": "f1db50251805ed69b43cf4f72ea2f0e231f36a1c",
                 "model_type": "LUAR", "layers": 6, "hidden": 768, "dimension": 512,
                 "weights_sha256": "443939bb2df7a31f73890e6013ae20021e1fffed47f6142fe813257e6501e63b"},
    "lusr": {"model_id": "rrivera1849/LUSR", "model_revision": "ab7852825ad1be40112bbfdaf1f9e2315a1727b7",
             "model_type": "roberta", "layers": 12, "hidden": 768, "dimension": 768,
             "weights_sha256": "9dc936bd31f349d874a79880549a5a9a9ca0a7ead64bebb83904cac2666cd5b9"},
}
MAX_LENGTH = 512
POOLING = "masked_mean_including_special_tokens"


def validate_snapshot(directory: str | Path, key: str) -> dict:
    """The pinned revision and weights, exactly the recorded files at their recorded sizes (nothing
    unlisted, no code), and the reviewed architecture."""
    spec = ENCODERS[key]
    directory = Path(directory)
    manifest = json.loads((directory / "snapshot_manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("repo_id") != spec["model_id"]
            or manifest.get("revision") != spec["model_revision"] or manifest.get("includes_weights") is not True):
        raise ValueError("Expected a full snapshot of the pinned encoder revision")
    names = set()
    for entry in manifest.get("files", []):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe path in encoder snapshot manifest")
        path = directory / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Missing or changed encoder snapshot file: {relative}")
        names.add(str(relative))
    on_disk = {str(path.relative_to(directory)) for path in directory.rglob("*")
               if path.is_file() and path.relative_to(directory).parts[0] != ".cache"
               and path.name != "snapshot_manifest.json"}
    if "model.safetensors" not in names or any(name.endswith(".py") for name in on_disk | names):
        raise ValueError("The encoders load safetensors weights and never remote code")
    if on_disk != names:
        raise ValueError("Encoder snapshot holds files its manifest does not record")
    digest = hashlib.sha256()
    with (directory / "model.safetensors").open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != spec["weights_sha256"]:
        raise ValueError("Encoder weights differ from the pinned checkpoint")
    config = json.loads((directory / "config.json").read_text())
    backbone = config.get("backbone_config", {}) if key == "luar-mud" else config
    if (config.get("model_type"), backbone.get("num_hidden_layers"), backbone.get("hidden_size")) != (
            spec["model_type"], spec["layers"], spec["hidden"]):
        raise ValueError("Encoder architecture differs from the reviewed checkpoint")
    return manifest


def load(directory: str | Path, key: str, device: str = "cpu"):
    """(tokenizer, RoBERTa backbone, LUAR projection or None), from the checkpoint's own tensors."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, RobertaConfig, RobertaModel

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    state = load_file(str(directory / "model.safetensors"))
    projection = None
    if key == "luar-mud":
        backbone_config = RobertaConfig(**{name: value for name, value in config["backbone_config"].items()
                                           if name != "model_type"})
        backbone_state = {name.removeprefix("transformer."): value for name, value in state.items()
                          if name.startswith("transformer.")}
        projection = torch.nn.Linear(config["backbone_config"]["hidden_size"], config["embedding_size"])
        projection.load_state_dict({"weight": state["linear.weight"], "bias": state["linear.bias"]})
        if len(backbone_state) + 2 != len(state):
            raise ValueError("Unexpected tensors in the LUAR checkpoint")
    else:
        backbone_config = RobertaConfig.from_pretrained(str(directory), local_files_only=True)
        backbone_state = state
    model = RobertaModel(backbone_config, add_pooling_layer=any(name.startswith("pooler.") for name in backbone_state))
    result = model.load_state_dict(backbone_state, strict=False)
    if result.unexpected_keys or [name for name in result.missing_keys if not name.endswith("position_ids")]:
        raise ValueError("The checkpoint does not match the RoBERTa backbone exactly")
    model.to(device).eval().requires_grad_(False)
    if projection is not None:
        projection.to(device).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False,
                                              padding_side="right", use_fast=True)
    return tokenizer, model, projection


def embed(texts: list[str], tokenizer, model, projection, *, batch_size: int, device: str) -> np.ndarray:
    import torch

    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=MAX_LENGTH,
                              return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            if projection is not None:
                vectors = projection(vectors)
            norms = vectors.norm(p=2, dim=1, keepdim=True)
            if not torch.isfinite(vectors).all() or torch.any(norms <= 0):
                raise ValueError("Encoder produced non-finite or zero embeddings")
            chunks.append((vectors / norms).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def extract(texts: list[str], directory: str | Path, key: str, *, batch_size: int = 16,
            device: str = "cpu") -> tuple[np.ndarray, dict]:
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Encoder inputs must be nonempty passage strings")
    spec = ENCODERS[key]
    validate_snapshot(directory, key)
    started = time.perf_counter()
    tokenizer, model, projection = load(directory, key, device)
    try:
        lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
        embeddings = embed(texts, tokenizer, model, projection, batch_size=batch_size, device=device)
    finally:
        del model, projection
        gc.collect()
    if embeddings.shape != (len(texts), spec["dimension"]) or not np.allclose(
            np.linalg.norm(embeddings, axis=1), 1, atol=1e-5):
        raise ValueError("Embeddings must be unit vectors of the reviewed dimension")
    options = {"pooling": POOLING, "max_length": MAX_LENGTH, "normalization": "l2", "padding_side": "right",
               "compute_dtype": "float32", "one_passage_per_episode": True,
               "projection": "luar_768_to_512" if key == "luar-mud" else None, "remote_code": False}
    metadata = {"model_id": spec["model_id"], "model_revision": spec["model_revision"], "encoder_frozen": True,
                "feature_options": options,
                "profile": {"rows": len(texts), "dimension": spec["dimension"], "batch_size": batch_size,
                            "elapsed_seconds": time.perf_counter() - started, "device": device,
                            "truncated_rows": int(sum(length > MAX_LENGTH for length in lengths)),
                            "maximum_input_tokens": int(max(lengths)),
                            "token_length_quantiles": {str(q): float(np.quantile(lengths, q)) for q in (0.5, 0.9, 0.99)},
                            "versions": {package: importlib.metadata.version(package)
                                         for package in ("torch", "transformers", "numpy")}}}
    return embeddings, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--model-key", choices=tuple(ENCODERS), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Feature output already exists; preserve it or choose another output path")
    train = read_train(args.train)
    embeddings, metadata = extract(train["text"].tolist(), args.model_path, args.model_key,
                                   batch_size=args.batch_size, device=args.device)
    metadata["train_fingerprint"] = fingerprint_frame(train)
    saved = save_feature_archive(args.output, train["id"].tolist(), embeddings, metadata, texts=train["text"].tolist())
    args.output.with_suffix(".json").write_text(json.dumps(saved, indent=2) + "\n")
    print(json.dumps(metadata["profile"], indent=2))
    return 0


if __name__ == "__main__":
    from ntl.style_encoders import main as imported_main

    raise SystemExit(imported_main())
