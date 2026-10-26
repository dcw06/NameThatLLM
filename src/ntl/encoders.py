"""Frozen, pinned StyleDistance and Qwen3 embeddings for the Gemini diagnostic."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import read_train
from .frozen_gemini import save_feature_archive
from .gemini import (
    OFFICIAL_BASELINE_COUNTS,
    _initialize_cuda_memory_stats,
    prepare_gemini_evaluation,
)

ENCODERS = {
    "styledistance": {
        "model_id": "StyleDistance/styledistance",
        "model_revision": "b7df5f0b0480773c097ba3121d83ca32b71015ca",
        "model_type": "roberta", "layers": 12, "dimension": 768,
        "pooling": "masked_mean_including_special_tokens", "padding_side": "right",
        "st_modules": ["Transformer", "Pooling"],
    },
    "qwen3-embedding-0.6b": {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "model_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "model_type": "qwen3", "layers": 28, "dimension": 1024,
        "pooling": "last_non_padding_token", "padding_side": "left",
        "st_modules": ["Transformer", "Pooling", "Normalize"],
    },
}


def validate_snapshot(directory: str | Path, model_key: str) -> dict:
    spec = ENCODERS[model_key]
    directory = Path(directory)
    manifest = json.loads((directory / "snapshot_manifest.json").read_text())
    if (manifest.get("schema_version") != 1
            or manifest.get("repo_id") != spec["model_id"]
            or manifest.get("revision") != spec["model_revision"]
            or manifest.get("includes_weights") is not True):
        raise ValueError("Expected a full snapshot of the selected immutable encoder revision")
    entries = manifest.get("files", [])
    names = set()
    if not entries:
        raise ValueError("Encoder snapshot has no recorded files")
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in names:
            raise ValueError("Unsafe or repeated encoder snapshot path")
        names.add(str(relative))
        path = directory / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Missing or changed encoder snapshot file: {relative}")
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("The pinned encoders require safetensors weights")
    config = json.loads((directory / "config.json").read_text())
    actual = (config.get("model_type"), config.get("num_hidden_layers"), config.get("hidden_size"))
    if actual != (spec["model_type"], spec["layers"], spec["dimension"]):
        raise ValueError("Encoder architecture differs from the reviewed checkpoint")
    modules = json.loads((directory / "modules.json").read_text())
    if [item["type"].rsplit(".", 1)[-1] for item in modules] != spec["st_modules"]:
        raise ValueError("Unexpected Sentence Transformers modules; a learned transform may be missing")
    pooling = json.loads((directory / "1_Pooling/config.json").read_text())
    intended = "pooling_mode_mean_tokens" if model_key == "styledistance" else "pooling_mode_lasttoken"
    if pooling.get(intended) is not True:
        raise ValueError("Upstream pooling differs from the reviewed feature recipe")
    for key, enabled in pooling.items():
        if key.startswith("pooling_mode_") and key != intended and enabled:
            raise ValueError("Upstream uses additional pooling modes that must be implemented explicitly")
    return manifest


def masked_mean_pool(hidden, attention_mask):
    """Native StyleDistance pooling counts every unmasked token, including specials."""
    import torch

    mask = attention_mask.unsqueeze(-1).to(torch.float32)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Cannot pool an empty token sequence")
    return (hidden.float() * mask).sum(dim=1) / counts


def last_non_padding_pool(hidden, attention_mask):
    """Select the last actual token for either left- or right-padded batches."""
    import torch

    if torch.any(attention_mask.sum(dim=1) == 0):
        raise ValueError("Cannot pool an empty token sequence")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last = positions.unsqueeze(0).expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()


def extract_embeddings(
    texts: list[str], model_path: str | Path, model_key: str,
    *, batch_size: int = 16, max_length: int = 512, device: str = "cuda:0",
) -> tuple[np.ndarray, dict]:
    if model_key not in ENCODERS:
        raise ValueError("Unknown encoder key")
    if batch_size < 1 or not 16 <= max_length <= 512:
        raise ValueError("Use a positive batch size and a token limit from 16 to 512")
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Encoder inputs must be nonempty passage strings")
    spec = ENCODERS[model_key]
    validate_snapshot(model_path, model_key)
    import torch
    from transformers import AutoModel, AutoTokenizer

    target = torch.device(device)
    if target.type not in {"cuda", "cpu"}:
        raise ValueError("Use CUDA for the experiment or CPU for a small loading check")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; enable a free Kaggle GPU")
        _initialize_cuda_memory_stats(torch, target)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        padding_side=spec["padding_side"], use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Encoder tokenizer needs a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    # The complete embedding backbone is used. There is no layer truncation,
    # language-model output head, quantization, task instruction, or fine-tuning.
    model = AutoModel.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float32,
    ).to(target)
    model.config.use_cache = False
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The encoder must remain completely frozen")
    chunks, lengths = [], []
    try:
        with torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                passages = texts[offset:offset + batch_size]
                lengths.extend(tokenizer(passages, truncation=False, return_length=True)["length"])
                batch = tokenizer(
                    passages, padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt",
                ).to(target)
                hidden = model(**batch).last_hidden_state
                if spec["pooling"] == "last_non_padding_token":
                    vectors = last_non_padding_pool(hidden, batch["attention_mask"])
                else:
                    vectors = masked_mean_pool(hidden, batch["attention_mask"])
                norms = vectors.norm(p=2, dim=1, keepdim=True)
                if not torch.isfinite(vectors).all() or torch.any(norms <= 0):
                    raise ValueError("Encoder produced non-finite or zero embeddings")
                vectors = vectors / norms
                chunks.append(vectors.cpu().numpy())
                del hidden, vectors, batch
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        embeddings = np.concatenate(chunks).astype(np.float32, copy=False)
        if embeddings.shape != (len(texts), spec["dimension"]):
            raise ValueError("Unexpected encoder embedding shape")
        if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5):
            raise ValueError("Frozen embeddings must be finite unit vectors")
        versions = {package: importlib.metadata.version(package) for package in ("torch", "transformers", "numpy")}
        metadata = {
            "model_id": spec["model_id"], "model_revision": spec["model_revision"],
            "encoder_frozen": True,
            "feature_options": {
                "pooling": spec["pooling"], "max_length": max_length,
                "normalization": "l2", "padding_side": spec["padding_side"],
                "compute_dtype": "float32", "instruction_prefix": "",
                "special_tokens_included": True, "quantization": "none",
                "full_backbone": True,
                "use_cache": False,
            },
            "profile": {
                "rows": len(texts), "dimension": embeddings.shape[1],
                "elapsed_seconds": time.perf_counter() - started,
                "device": str(target),
                "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else None,
                "truncated_rows": int(sum(length > max_length for length in lengths)),
                "maximum_input_tokens": int(max(lengths)),
                "token_length_quantiles": {str(q): float(np.quantile(lengths, q)) for q in (0.5, 0.9, 0.99)},
                "versions": versions,
            },
        }
        return embeddings, metadata
    finally:
        del model
        gc.collect()
        if target.type == "cuda":
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--baseline-oof", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--model-key", choices=tuple(ENCODERS), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Feature output already exists; preserve it or choose another output path")
    frame, protocol = prepare_gemini_evaluation(
        read_train(args.train), pd.read_parquet(args.baseline_oof), args.fold_manifest,
        OFFICIAL_BASELINE_COUNTS,
    )
    embeddings, metadata = extract_embeddings(
        frame["text"].tolist(), args.model_path, args.model_key,
        batch_size=args.batch_size, max_length=args.max_length, device=args.device,
    )
    metadata["train_fingerprint"] = protocol["train_fingerprint"]
    metadata["baseline_oof_fingerprint"] = protocol["baseline_oof_fingerprint"]
    saved = save_feature_archive(
        args.output, frame["id"].tolist(), embeddings, metadata, texts=frame["text"].tolist(),
    )
    args.output.with_suffix(".json").write_text(json.dumps(saved, indent=2) + "\n")
    print(json.dumps({"feature_archive": str(args.output), **metadata["profile"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
