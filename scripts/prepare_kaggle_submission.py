#!/usr/bin/env python3
"""Prepare a private CPU Code Competition notebook; never push or submit it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "name-that-llm-cpu-first-submission"
DEFAULT_TITLE = "Name That LLM CPU first submission"


RUNTIME = '''\
# Use fresh subprocesses with Kaggle's installed CPU packages. Every learned
# artifact is fitted and consumed during this run; none crosses environments.
import subprocess

RUNTIME_PYTHON = sys.executable
RUNTIME_ENV = dict(os.environ)
RUNTIME_ENV.pop("PYTHONHOME", None)
RUNTIME_ENV["PYTHONPATH"] = str(SOURCE_ROOT)
RUNTIME_ENV["PYTHONNOUSERSITE"] = "1"
RUNTIME_ENV["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

version_probe = """
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import importlib
import json
from pathlib import Path
import platform
import sys

packages = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "scikit-learn": "sklearn", "joblib": "joblib", "pyarrow": "pyarrow",
    "threadpoolctl": "threadpoolctl",
}
versions = {}
for distribution, module in packages.items():
    try:
        importlib.import_module(module)
        versions[distribution] = version(distribution)
    except (ImportError, PackageNotFoundError) as exc:
        raise RuntimeError(f"Kaggle CPU dependency is unavailable: {distribution}") from exc
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "versions": versions,
    "source_sha256": sys.argv[2],
    "uses_kaggle_installed_cpu_packages": True,
    "fresh_model_fit_in_same_environment": True,
}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + chr(10))
print(json.dumps(report, indent=2))
"""
subprocess.run(
    [RUNTIME_PYTHON, "-c", version_probe,
     "/kaggle/working/first_submission_environment.json", SOURCE_SHA256],
    check=True, env=RUNTIME_ENV,
)

def run_runtime(module, arguments):
    subprocess.run([RUNTIME_PYTHON, "-m", module, *arguments],
                   check=True, env=RUNTIME_ENV)
'''


def build_bundle(
    username: str,
    output_dir: str | Path,
    *,
    slug: str = DEFAULT_SLUG,
    title: str = DEFAULT_TITLE,
    run_cv: bool = True,
) -> dict:
    """Write source and metadata locally, without importing Kaggle or credentials."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", username):
        raise ValueError("Provide a Kaggle username without a slash or URL")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise ValueError("Notebook slug must use lowercase letters, digits, and hyphens")
    if not title.strip() or len(title) > 50:
        raise ValueError("Notebook title must contain 1 to 50 characters")

    exporter = runpy.run_path(str(ROOT / "scripts" / "build_kaggle_notebooks.py"))
    sources = exporter["package_sources"]()
    cell = exporter["cell"]
    cells = [
        cell("markdown", '''\
        # Name That LLM: first private CPU submission

        This self-contained notebook uses Kaggle's installed CPU packages,
        records their versions, and fits every learned artifact during this
        run. Attach the official competition data. Use **CPU**, **Internet
        off**, and a **private** notebook. No extra data, weights, API calls,
        or dependency downloads are required.

        Prompt-grouped cross-validation runs before the final fit. Its report
        is a local diagnostic; Kaggle's scored rerun supplies the public score.
        This generated notebook has no measured competition result yet.
        '''),
        cell("code", exporter["bootstrap"](sources)),
        cell("code", RUNTIME),
        cell("code", exporter["DATA_DISCOVERY"]),
        cell("code", f'''\
        RUN_GROUPED_CV = {run_cv!r}
        CPU_MODEL_DIR = "/kaggle/working/cpu-model"
        if RUN_GROUPED_CV:
            run_runtime("ntl", ["baseline-cv", "--train", TRAIN,
                                 "--output-dir", "/kaggle/working/cpu-cv"])
        run_runtime("ntl", ["baseline-train", "--train", TRAIN,
                             "--output-dir", CPU_MODEL_DIR])
        run_runtime("ntl", ["predict", "--model", CPU_MODEL_DIR + "/model.joblib",
                             "--test", TEST, "--output", SUBMISSION])
        '''),
        cell("code", exporter["SUBMISSION_CHECK"]),
        cell("markdown", '''\
        The final artifact is `submission.parquet`, with exactly `id` and
        `model`. The current test file determines every ID and row count;
        Kaggle replaces the visible examples with hidden rows when scoring.
        Retain `first_submission_environment.json` and `cpu-cv/` to review
        the runtime and validation. A saved successful execution does not
        submit predictions automatically.
        '''),
    ]
    output = Path(output_dir).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    exporter["write_notebook"](output / "main.ipynb", cells, gpu=False, internet=False)
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": "main.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_sources": ["name-that-llm"],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "kernel_id": metadata["id"],
        "run_grouped_cv": run_cv,
        "requires_offline_wheel_bundle": False,
        "uses_kaggle_installed_cpu_packages": True,
        "submitted": False,
        "files": [
            {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (output / "main.ipynb", output / "kernel-metadata.json")
        ],
    }
    (output / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output_dir": str(output), "kernel_id": metadata["id"], "run_grouped_cv": run_cv}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Kaggle username; no credentials are read")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "first_submission")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--skip-cv", action="store_true", help="Skip diagnostics for a plumbing-only run")
    args = parser.parse_args(argv)
    try:
        result = build_bundle(
            args.username, args.output_dir, slug=args.slug, title=args.title,
            run_cv=not args.skip_cv,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
