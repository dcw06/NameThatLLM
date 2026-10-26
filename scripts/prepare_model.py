#!/usr/bin/env python3
"""Prepare a pinned, offline model snapshot; --verify never accesses the network."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

DEFAULT_REPO = "Qwen/Qwen3-8B-Base"
MANIFEST_NAME = "snapshot_manifest.json"
MODEL_SPECS = {
    DEFAULT_REPO: {
        "revision": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
        "model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36,
    },
    "Qwen/Qwen3-14B-Base": {
        "revision": "0b0bd3732e2c374d483664439ea334928b65f304",
        "model_type": "qwen3", "hidden_size": 5120, "num_hidden_layers": 40,
    },
    "Qwen/Qwen3-4B-Base": {
        "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36,
    },
    "microsoft/deberta-v3-large": {
        "revision": "64a8c8eab3e352a784c658aef62be1662607476f",
        "model_type": "deberta-v2", "hidden_size": 1024, "num_hidden_layers": 24,
    },
    "microsoft/deberta-v3-base": {
        "revision": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        "model_type": "deberta-v2", "hidden_size": 768, "num_hidden_layers": 12,
    },
    "StyleDistance/styledistance": {
        "revision": "b7df5f0b0480773c097ba3121d83ca32b71015ca",
        "model_type": "roberta", "hidden_size": 768, "num_hidden_layers": 12,
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28,
    },
    "Qwen/Qwen3.5-9B": {
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "model_type": "qwen3_5", "hidden_size": 4096, "num_hidden_layers": 32,
        "nested_text": True,
    },
}
METADATA_PATTERNS = [
    "*.json", "tokenizer*", "vocab*", "merges*", "spm.model",
    "README*", "LICENSE*", "license*", "NOTICE*", "*.jinja",
    "added_tokens*", "special_tokens*",
]


class SnapshotError(ValueError):
    """A snapshot cannot safely be prepared or verified."""


def _spec(repo_id: str, revision: str | None) -> tuple[dict[str, Any], str]:
    if repo_id not in MODEL_SPECS:
        raise SnapshotError("Model is outside the reviewed, pinned allowlist.")
    spec = MODEL_SPECS[repo_id]
    revision = spec["revision"] if revision is None else revision
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SnapshotError("Revision must be an immutable 40-character lowercase commit SHA.")
    if revision != spec["revision"]:
        raise SnapshotError("Revision does not match the reviewed pin for this model.")
    return spec, revision


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"Cannot read valid JSON from {path.name}.") from exc
    if not isinstance(data, dict):
        raise SnapshotError(f"{path.name} must contain a JSON object.")
    return data


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SnapshotError("Manifest contains an invalid file path.")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise SnapshotError("Manifest paths must be normalized relative paths.")
    if path.parts[0] == ".cache" or value == MANIFEST_NAME:
        raise SnapshotError("Manifest cannot inventory its cache or itself.")
    return value


def _inventory(directory: Path) -> list[dict[str, Any]]:
    files = []
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if relative.parts[0] == ".cache" or relative.as_posix() == MANIFEST_NAME:
            continue
        if path.is_symlink():
            raise SnapshotError("Snapshot artifacts must be local files, not symbolic links.")
        if path.is_file():
            files.append({"path": relative.as_posix(), "size": path.stat().st_size})
    return sorted(files, key=lambda item: item["path"])


def _has_weights(files: list[dict[str, Any]]) -> bool:
    return any(
        item["path"].endswith(".safetensors")
        or PurePosixPath(item["path"]).name == "pytorch_model.bin"
        for item in files
    )


def _validate_config(directory: Path, spec: dict[str, Any]) -> None:
    config = _read_json(directory / "config.json")
    if config.get("model_type") != spec["model_type"]:
        raise SnapshotError("Downloaded config has an unexpected model_type.")
    text_config = config.get("text_config", {}) if spec.get("nested_text") else config
    for field in ("hidden_size", "num_hidden_layers"):
        if not isinstance(text_config, dict) or text_config.get(field) != spec[field]:
            raise SnapshotError(f"Downloaded config has an unexpected {field}.")


def _validate_shards(directory: Path, files: list[dict[str, Any]]) -> None:
    names = {item["path"] for item in files}
    for item in files:
        if not item["path"].endswith(".safetensors.index.json"):
            continue
        index = _read_json(directory / item["path"])
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SnapshotError("Weight index is missing its tensor-to-shard map.")
        parent = PurePosixPath(item["path"]).parent
        required = {_safe_relative((parent / _safe_relative(v)).as_posix())
                    for v in weight_map.values()}
        if not required.issubset(names):
            raise SnapshotError("Snapshot is missing weight shards referenced by its index.")


def verify_snapshot(
    output_dir: str | Path, *, repo_id: str | None = None, revision: str | None = None
) -> dict[str, Any]:
    """Check identity, inventory, byte sizes and config without importing Hub code."""
    directory = Path(output_dir).expanduser().absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise SnapshotError("Snapshot destination must be an existing local directory.")
    manifest_path = directory / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise SnapshotError("Snapshot manifest must be a local file.")
    manifest = _read_json(manifest_path)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise SnapshotError("Unsupported snapshot manifest schema_version.")
    actual_repo = manifest.get("repo_id")
    if not isinstance(actual_repo, str):
        raise SnapshotError("Snapshot manifest is missing repo_id.")
    if repo_id is not None and actual_repo != repo_id:
        raise SnapshotError("Destination contains a different model; use a separate directory.")
    spec, actual_revision = _spec(actual_repo, manifest.get("revision"))
    if manifest.get("revision") != actual_revision:
        raise SnapshotError("Snapshot manifest is missing its immutable revision.")
    if revision is not None and actual_revision != revision:
        raise SnapshotError("Destination contains a different revision; use a separate directory.")
    if type(manifest.get("includes_weights")) is not bool:
        raise SnapshotError("Manifest includes_weights must be a boolean.")
    if not isinstance(manifest.get("downloaded_at_utc"), str):
        raise SnapshotError("Manifest is missing downloaded_at_utc.")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise SnapshotError("Manifest must contain a nonempty file inventory.")
    declared = {}
    for item in entries:
        if not isinstance(item, dict):
            raise SnapshotError("Invalid manifest inventory entry.")
        name = _safe_relative(item.get("path"))
        size = item.get("size")
        if name in declared or type(size) is not int or size < 0:
            raise SnapshotError("Manifest file paths must be unique and byte sizes nonnegative.")
        declared[name] = size
    actual_files = _inventory(directory)
    actual = {item["path"]: item["size"] for item in actual_files}
    if actual.keys() != declared.keys():
        raise SnapshotError("Snapshot file count or paths differ from its manifest.")
    if actual != declared:
        raise SnapshotError("Snapshot file byte sizes differ from its manifest.")
    if _has_weights(actual_files) != manifest["includes_weights"]:
        raise SnapshotError("Snapshot weight files disagree with includes_weights.")
    _validate_config(directory, spec)
    if manifest["includes_weights"]:
        _validate_shards(directory, actual_files)
    return manifest


def _download_snapshot(**kwargs: Any) -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SnapshotError(
            "Install huggingface_hub to download a snapshot; --verify needs only Python."
        ) from None
    return snapshot_download(**kwargs)


def _atomic_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory,
            prefix=".snapshot_manifest.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, directory / MANIFEST_NAME)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def prepare_snapshot(
    output_dir: str | Path, *, repo_id: str = DEFAULT_REPO,
    revision: str | None = None, metadata_only: bool = False,
    downloader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Download reviewed artifacts; publish a manifest only after validation succeeds."""
    spec, revision = _spec(repo_id, revision)
    directory = Path(output_dir).expanduser().absolute()
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise SnapshotError("Snapshot destination must be a local directory.")
    if directory.exists() and any(directory.iterdir()):
        if not (directory / MANIFEST_NAME).is_file():
            raise SnapshotError(
                "Nonempty destination has no matching manifest; use a separate empty directory."
            )
        verify_snapshot(directory, repo_id=repo_id, revision=revision)
    directory.mkdir(parents=True, exist_ok=True)
    patterns = list(METADATA_PATTERNS)
    if not metadata_only:
        patterns.append("pytorch_model.bin" if repo_id.startswith("microsoft/")
                        else "*.safetensors")
    try:
        (downloader or _download_snapshot)(
            repo_id=repo_id, revision=revision, local_dir=str(directory),
            allow_patterns=patterns, token=False, max_workers=4,
        )
    except SnapshotError:
        raise
    except Exception:
        # Hub exception text can include authenticated URLs; never echo it.
        raise SnapshotError(
            "Snapshot download failed. Check network access and Hub availability. "
            "No new manifest was published; use a fresh directory after a partial download."
        ) from None
    _validate_config(directory, spec)
    files = _inventory(directory)
    includes_weights = _has_weights(files)
    if not metadata_only and not includes_weights:
        raise SnapshotError("Download completed without model weights; no manifest was published.")
    if includes_weights:
        _validate_shards(directory, files)
    manifest = {
        "schema_version": 1, "repo_id": repo_id, "revision": revision,
        "includes_weights": includes_weights, "files": files,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _atomic_manifest(directory, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-id", choices=sorted(MODEL_SPECS))
    parser.add_argument("--revision", help="Reviewed immutable commit SHA; defaults to the model's pin")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--metadata-only", action="store_true", help="Exclude model weight downloads")
    action.add_argument("--verify", action="store_true", help="Verify an existing snapshot entirely offline")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            manifest = verify_snapshot(args.output_dir, repo_id=args.repo_id, revision=args.revision)
        else:
            manifest = prepare_snapshot(
                args.output_dir, repo_id=args.repo_id or DEFAULT_REPO,
                revision=args.revision, metadata_only=args.metadata_only,
            )
    except (SnapshotError, OSError) as exc:
        print(f"Model snapshot error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: manifest[key] for key in
                      ("repo_id", "revision", "includes_weights")}
                     | {"file_count": len(manifest["files"]),
                        "verified_offline": args.verify}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
