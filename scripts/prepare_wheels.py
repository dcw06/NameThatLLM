"""Prepare hash-locked offline wheels on the same Linux/Python image used for scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory: Path) -> dict:
    manifest = json.loads((directory / "wheel_manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported wheel manifest")
    expected_python = [sys.version_info.major, sys.version_info.minor]
    if manifest.get("python") != expected_python:
        raise ValueError("Wheel bundle Python version differs from the current interpreter")
    if manifest.get("machine") != platform.machine() or manifest.get("system") != platform.system():
        raise ValueError("Wheel bundle platform differs from the current interpreter")
    entries = manifest.get("files", [])
    if not entries:
        raise ValueError("Wheel manifest contains no files")
    expected_files = set()
    for item in entries:
        filename = item["path"]
        if Path(filename).name != filename or filename in expected_files:
            raise ValueError("Invalid or repeated wheel bundle path")
        expected_files.add(filename)
        target = directory / filename
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"Missing local wheel artifact: {filename}")
        if target.stat().st_size != item["size"] or sha256(target) != item["sha256"]:
            raise ValueError(f"Wheel bundle artifact changed: {filename}")
    actual_files = {p.name for p in directory.iterdir() if p.name != "wheel_manifest.json"}
    if actual_files != expected_files or "requirements.lock" not in actual_files:
        raise ValueError("Wheel bundle inventory differs from its manifest")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=Path("requirements-kaggle.lock"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("cpu", "neural"), default="neural")
    parser.add_argument("--verify", action="store_true", help="Verify existing files without network")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.verify:
        manifest = verify(output)
        print(f"Verified {len(manifest['files'])} offline wheel bundle files")
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("Prepare this wheel bundle inside an online Kaggle CPU notebook (Linux x86_64).")
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Wheel destination must be empty; use --verify for an existing bundle.")
    requirements = args.requirements.resolve()
    if not requirements.is_file() or "--hash=sha256:" not in requirements.read_text():
        raise SystemExit("Provide the exported, hash-locked requirements-kaggle.lock file.")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(requirements, output / "requirements.lock")
    # The entire dependency graph is present in the uv export. --no-deps prevents
    # resolution outside the lock; pip verifies each selected wheel against its hash.
    subprocess.run(
        [sys.executable, "-m", "pip", "download", "--disable-pip-version-check",
         "--index-url", "https://pypi.org/simple", "--only-binary=:all:",
         "--require-hashes", "--no-deps", "--dest", str(output),
         "--requirement", str(output / "requirements.lock")],
        check=True,
    )
    files = [
        {"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    ]
    manifest = {
        "schema_version": 1,
        "profile": args.profile,
        "python": [sys.version_info.major, sys.version_info.minor],
        "machine": platform.machine(), "system": platform.system(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    temporary = output / ".wheel_manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "wheel_manifest.json")
    verify(output)
    print(f"Saved {sum(f['size'] for f in files) / 2**30:.2f} GiB of locked offline dependencies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
