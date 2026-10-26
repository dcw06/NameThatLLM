#!/usr/bin/env python3
"""Export self-contained, offline Kaggle notebooks from the local ntl package.

Uses only Python's standard library. Run again after changing src/ntl so the
notebooks carry the same implementation as the reviewed local source.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def requirements(*, neural: bool) -> list[str]:
    """Keep notebook compatibility checks aligned with the reviewed package pins."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    result = list(project["dependencies"])
    if neural:
        result.extend(project["optional-dependencies"]["neural"])
        result.extend(project["optional-dependencies"]["gpu"])
    return result


def wheel_requirements(*, neural: bool) -> str:
    path = ROOT / ("requirements-kaggle.lock" if neural else "requirements-cpu.lock")
    if not path.is_file():
        raise SystemExit(f"{path.name} is missing; prepare the pinned lock before exporting.")
    return path.read_text(encoding="utf-8")


def dependency_cell(*, neural: bool) -> str:
    expected = hashlib.sha256(wheel_requirements(neural=neural).encode()).hexdigest()
    return (
        f"REQUIRED_PACKAGES = {requirements(neural=neural)!r}\n"
        f"WHEEL_PROFILE = {'neural' if neural else 'cpu'!r}\n"
        f"WHEEL_REQUIREMENTS_SHA256 = {expected!r}\n" + DEPENDENCIES
    )


def cell(kind: str, source: str) -> dict:
    result = {
        "cell_type": kind,
        "metadata": {},
        "source": textwrap.dedent(source).strip() + "\n",
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def package_sources() -> dict[str, str]:
    package = ROOT / "src" / "ntl"
    sources = {
        str(path.relative_to(ROOT / "src")): path.read_text(encoding="utf-8")
        for path in sorted(package.rglob("*.py"))
        if path.is_file() and not path.is_symlink()
    }
    if "ntl/__init__.py" not in sources:
        raise SystemExit("src/ntl/__init__.py is missing; create the package before exporting.")
    return sources


def bootstrap(sources: dict[str, str]) -> str:
    payload = json.dumps(sources, ensure_ascii=True, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f'''\
# Embedded project source; regenerate this notebook after editing src/ntl.
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

# Scoring is offline. These also prevent accidental Hugging Face requests.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SOURCE_JSON = {payload!r}
SOURCE_SHA256 = {digest!r}
assert hashlib.sha256(SOURCE_JSON.encode()).hexdigest() == SOURCE_SHA256
SOURCE_ROOT = Path("/kaggle/working/ntl_source")
for relative, content in json.loads(SOURCE_JSON).items():
    target = SOURCE_ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
sys.path.insert(0, str(SOURCE_ROOT))
importlib.invalidate_caches()
print("Embedded ntl source SHA-256:", SOURCE_SHA256)
'''


DEPENDENCIES = '''\
# Attach the pinned wheel bundle from kaggle_prepare_dependencies.ipynb.
# Set an explicit folder if more than one bundle is attached.
WHEEL_DIR = None  # directory containing wheel_manifest.json and requirements.lock
if WHEEL_DIR is None:
    wheel_candidates = []
    for wheel_manifest in sorted(Path("/kaggle/input").rglob("wheel_manifest.json")):
        try:
            candidate_manifest = json.loads(wheel_manifest.read_text())
        except (OSError, ValueError):
            continue
        if candidate_manifest.get("profile") == WHEEL_PROFILE:
            wheel_candidates.append(wheel_manifest)
    if len(wheel_candidates) == 1:
        WHEEL_DIR = str(wheel_candidates[0].parent)
    elif len(wheel_candidates) > 1:
        raise RuntimeError("Several wheel bundles are attached; set WHEEL_DIR explicitly")
if WHEEL_DIR is None:
    raise RuntimeError(
        f"Attach the prepared {WHEEL_PROFILE} wheel bundle from kaggle_prepare_dependencies.ipynb"
    )
import platform
import subprocess
import venv
bundle_manifest = json.loads((Path(WHEEL_DIR) / "wheel_manifest.json").read_text())
if bundle_manifest.get("profile") != WHEEL_PROFILE:
    raise RuntimeError(f"Attach the {WHEEL_PROFILE} wheel bundle for this notebook")
if (bundle_manifest.get("python") != list(sys.version_info[:2])
        or bundle_manifest.get("machine") != platform.machine()
        or bundle_manifest.get("system") != platform.system()):
    raise RuntimeError("Prepare the wheel bundle on the same Kaggle Python/Linux image as this run")
wheel_requirements_path = Path(WHEEL_DIR) / "requirements.lock"
if not wheel_requirements_path.is_file():
    raise RuntimeError("Wheel bundle is missing requirements.lock")
if hashlib.sha256(wheel_requirements_path.read_bytes()).hexdigest() != WHEEL_REQUIREMENTS_SHA256:
    raise RuntimeError("Wheel requirements differ from this notebook; attach its matching prepared bundle")

# Train and infer in a fresh package environment, independent of packages that
# Kaggle's notebook kernel may already have imported (including audio/vision).
RUNTIME_DIR = Path("/kaggle/working/ntl_env") / (WHEEL_PROFILE + "-" + WHEEL_REQUIREMENTS_SHA256[:12])
# Kaggle can disable ensurepip; the notebook's existing pip can install into
# a pip-less virtual environment through its supported --python option.
venv.EnvBuilder(with_pip=False, system_site_packages=False).create(RUNTIME_DIR)
RUNTIME_PYTHON = str(RUNTIME_DIR / "bin" / "python")
RUNTIME_ENV = dict(os.environ)
RUNTIME_ENV.pop("PYTHONHOME", None)
RUNTIME_ENV["PYTHONPATH"] = str(SOURCE_ROOT)
RUNTIME_ENV["PYTHONNOUSERSITE"] = "1"
RUNTIME_ENV["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
subprocess.run(
    [sys.executable, "-m", "pip", "--python", RUNTIME_PYTHON,
     "install", "--no-index", "--find-links",
     str(WHEEL_DIR), "--require-hashes", "-r", str(wheel_requirements_path)],
    check=True, env=RUNTIME_ENV,
)
version_probe = """
from importlib.metadata import PackageNotFoundError, version
from packaging.requirements import Requirement
import json
import sys
for raw_requirement in json.loads(sys.argv[1]):
    requirement = Requirement(raw_requirement)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    package = requirement.name
    try:
        installed_version = version(package)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Missing {package} in the isolated runtime; recreate the matching wheel bundle."
        ) from exc
    if installed_version not in requirement.specifier:
        raise RuntimeError(
            f"Expected {requirement}; installed {package}=={installed_version}. "
            "Recreate the matching wheel bundle before rerunning."
        )
    print(package, installed_version)
"""
subprocess.run([RUNTIME_PYTHON, "-c", version_probe, json.dumps(REQUIRED_PACKAGES)],
               check=True, env=RUNTIME_ENV)
def run_runtime(module, arguments):
    subprocess.run([RUNTIME_PYTHON, "-m", module, *arguments],
                   check=True, env=RUNTIME_ENV)
print("Isolated runtime:", RUNTIME_PYTHON)
'''


DATA_DISCOVERY = '''\
import csv

# Usually leave this unset. Never discover train.csv recursively: attached
# external datasets may also contain a file with that name.
COMPETITION_DIR = None
official_locations = [
    Path("/kaggle/input/competitions/name-that-llm"),
    Path("/kaggle/input/name-that-llm"),
]
if COMPETITION_DIR is not None:
    official_locations = [Path(COMPETITION_DIR)]
available = [p for p in official_locations
             if (p / "train.csv").is_file() and (p / "test.csv").is_file()]
if len(available) != 1:
    raise RuntimeError(
        "Attach the official Name That LLM competition data. If both official "
        "paths exist, set COMPETITION_DIR explicitly. Found: " + str(available)
    )
DATA_DIR = available[0]
TRAIN = str(DATA_DIR / "train.csv")
TEST = str(DATA_DIR / "test.csv")
SUBMISSION = "/kaggle/working/submission.parquet"
# A failed fresh run must not leave an earlier prediction file available.
Path(SUBMISSION).unlink(missing_ok=True)
def read_input_csv(path, name):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "text"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{name} requires id and text columns")
        rows = list(reader)
    if not rows or any(not row["id"] or not row["text"] for row in rows):
        raise ValueError(f"{name} is empty or contains null IDs/text")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} has duplicate IDs")
    return rows
train_rows = read_input_csv(TRAIN, "train")
test_rows = read_input_csv(TEST, "test")
VALID_LABELS = {
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
}
if {row.get("model") for row in train_rows} != VALID_LABELS:
    raise ValueError("Official training labels do not match the expected 18 labels")
print(f"Official train: {len(train_rows):,} rows; current test: {len(test_rows):,} rows")
print("The visible test is a smoke test; scoring replaces it with the hidden test.")
'''


SUBMISSION_CHECK = '''\
# Validate the file the competition will score, including arbitrary test size.
validation_code = """
import csv
import json
import sys
import pandas as pd
SUBMISSION, TEST, labels_json = sys.argv[1:]
VALID_LABELS = set(json.loads(labels_json))
with open(TEST, encoding="utf-8-sig", newline="") as handle:
    expected_ids = [row["id"] for row in csv.DictReader(handle)]
submission = pd.read_parquet(SUBMISSION)
if list(submission.columns) != ["id", "model"]:
    raise ValueError("submission.parquet must contain exactly id, model in that order")
if len(submission) != len(expected_ids):
    raise ValueError("Submission row count differs from the current test")
if submission.isna().any().any() or submission["id"].duplicated().any():
    raise ValueError("Submission contains null values or duplicate IDs")
if submission["id"].astype("string").tolist() != expected_ids:
    raise ValueError("Submission must preserve the official test IDs and row order")
if not submission["model"].isin(VALID_LABELS).all():
    raise ValueError("Submission contains a model outside the 18 allowed labels")
print(f"Validated {len(submission):,} predictions: {SUBMISSION}")
print(submission["model"].value_counts().to_string())
print(submission.head().to_string(index=False))
"""
subprocess.run([RUNTIME_PYTHON, "-c", validation_code, SUBMISSION, TEST,
                json.dumps(sorted(VALID_LABELS))], check=True, env=RUNTIME_ENV)
'''


MODEL_DISCOVERY = '''\
# Attach the Qwen3-8B-Base snapshot prepared by scripts/prepare_model.py.
# An explicit local folder is safest when several snapshots are attached.
MODEL_DIR = None  # e.g. "/kaggle/input/your-qwen3-8b-base-snapshot"
if MODEL_DIR is None:
    candidates = []
    for manifest_path in Path("/kaggle/input").rglob("snapshot_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if (manifest.get("repo_id") == "Qwen/Qwen3-8B-Base"
                and manifest.get("revision") == "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
                and manifest.get("includes_weights") is True):
            candidates.append(manifest_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(
            "Attach one pinned Qwen3-8B-Base snapshot and set MODEL_DIR explicitly "
            "if automatic discovery is ambiguous. Found: " + str(candidates)
        )
    MODEL_DIR = str(candidates[0])
manifest = json.loads((Path(MODEL_DIR) / "snapshot_manifest.json").read_text())
if (manifest.get("schema_version") != 1
        or manifest.get("repo_id") != "Qwen/Qwen3-8B-Base"
        or manifest.get("revision") != "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
        or manifest.get("includes_weights") is not True):
    raise ValueError("Model snapshot manifest does not match the pinned base model")
for entry in manifest["files"]:
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe path in model snapshot manifest")
    artifact = Path(MODEL_DIR) / relative
    if not artifact.is_file() or artifact.stat().st_size != entry["size"]:
        raise ValueError(f"Missing or incomplete snapshot file: {relative}")
model_config = json.loads((Path(MODEL_DIR) / "config.json").read_text())
if (model_config.get("model_type"), model_config.get("hidden_size"),
        model_config.get("num_hidden_layers")) != ("qwen3", 4096, 36):
    raise ValueError("MODEL_DIR is not a Qwen3-8B architecture")
print("Local model:", MODEL_DIR)
'''


def make_cpu(sources: dict[str, str]) -> list[dict]:
    return [
        cell("markdown", '''\
        # Name That LLM: CPU baseline

        Self-contained project export. Attach the official competition data,
        select **Internet off**, and **Save & Run All**. This fits the local
        TF-IDF/style baseline and writes `submission.parquet`. No API calls or
        model downloads run in this notebook. A successful sample run checks the
        pipeline; only an official submission measures hidden-test accuracy.
        '''),
        cell("code", bootstrap(sources)),
        cell("code", dependency_cell(neural=False)),
        cell("code", DATA_DISCOVERY),
        cell("code", '''\
        def run_cpu(arguments):
            run_runtime("ntl", arguments)
        RUN_GROUPED_CV = False
        if RUN_GROUPED_CV:
            run_cpu(["baseline-cv", "--train", TRAIN,
                     "--output-dir", "/kaggle/working/cpu-cv"])
        '''),
        cell("code", '''\
        CPU_MODEL_DIR = "/kaggle/working/cpu-model"
        run_cpu(["baseline-train", "--train", TRAIN, "--output-dir", CPU_MODEL_DIR])
        run_cpu(["predict", "--model", CPU_MODEL_DIR + "/model.joblib",
                 "--test", TEST, "--output", SUBMISSION])
        '''),
        cell("code", SUBMISSION_CHECK),
        cell("markdown", '''\
        Submit the **saved notebook version's `submission.parquet`** through
        Kaggle. The hidden rerun executes this same code against every hidden
        test passage. The artifacts under `cpu-model/` are fitted on all
        training rows and do not represent a held-out validation result.
        '''),
    ]


def make_gpu(sources: dict[str, str]) -> list[dict]:
    return [
        cell("markdown", '''\
        # Name That LLM: frozen Qwen3-8B-Base diagnostic

        Attach the official competition data and the pinned local base-model
        snapshot. Select **GPU T4 x2** and **Internet off**, then **Save & Run All**.
        This initial experiment extracts frozen features and fits a small
        classifier. Its score has not been measured; it is a diagnostic starting
        point for comparing features, pooling, and validation choices.

        `RUN_GROUPED_CV = False` below keeps the first submission run small.
        Turn it on for honest prompt-grouped validation. The notebook embeds
        project source and never downloads a model or calls a hosted API.
        '''),
        cell("code", bootstrap(sources)),
        cell("code", dependency_cell(neural=True)),
        cell("code", DATA_DISCOVERY),
        cell("code", MODEL_DISCOVERY),
        cell("code", '''\
        def run_neural(arguments):
            run_runtime("ntl.neural", arguments)
        gpu_probe = """
        import torch
        import bitsandbytes as bnb
        from transformers import Qwen3Model
        assert torch.cuda.is_available(), 'Enable a free Kaggle T4 GPU'
        print('GPU:', torch.cuda.get_device_name(0))
        print('Free / total GPU bytes:', torch.cuda.mem_get_info(0))
        layer = bnb.nn.Linear4bit(16, 8, bias=False, compute_dtype=torch.float16,
                                 quant_type='nf4').to('cuda:0')
        inputs = torch.randn(2, 16, device='cuda:0', dtype=torch.float16)
        result = layer(inputs)
        assert torch.isfinite(result).all(), 'NF4 smoke probe returned nonfinite values'
        print('Tiny NF4 forward probe passed:', tuple(result.shape))
        del layer, inputs, result
        torch.cuda.empty_cache()
        """
        subprocess.run([RUNTIME_PYTHON, "-c", gpu_probe], check=True, env=RUNTIME_ENV)
        RUN_GROUPED_CV = False
        NEURAL_MODEL_DIR = "/kaggle/working/qwen-frozen-model"
        FEATURE_ARGS = ["--model-path", str(MODEL_DIR), "--quantization", "nf4",
                        "--batch-size", "2", "--max-length", "512"]
        if RUN_GROUPED_CV:
            run_neural(["cv", "--train", TRAIN,
                        "--output-dir", "/kaggle/working/qwen-frozen-cv", *FEATURE_ARGS])
        '''),
        cell("code", '''\
        run_neural(["train", "--train", TRAIN,
                    "--output-dir", NEURAL_MODEL_DIR, *FEATURE_ARGS])
        run_neural(["predict", "--model-dir", NEURAL_MODEL_DIR,
                    "--test", TEST, "--output", SUBMISSION])
        '''),
        cell("code", SUBMISSION_CHECK),
        cell("markdown", '''\
        Submit the saved notebook's `submission.parquet`. Keep all model inputs
        attached for the offline hidden rerun. The visible example's runtime
        does not establish the runtime on about 4,000 hidden rows; keep a large
        margin below Kaggle's nine-hour limit. Review the completed run and
        release its GPU runtime when finished.
        '''),
    ]


def make_preparation() -> list[dict]:
    helper = ROOT / "scripts" / "prepare_model.py"
    if not helper.is_file():
        raise SystemExit("scripts/prepare_model.py is missing; create it before exporting.")
    source = helper.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()
    return [
        cell("markdown", '''\
        # Prepare the pinned Qwen3-8B-Base model input

        Run this once in a **CPU notebook with Internet on**. It downloads the
        public model at the project's exact revision, writes its license and
        snapshot manifest, and saves it under `/kaggle/working/qwen3-8b-base`.
        No GPU is needed, and the project does not call a paid API.

        After a successful Save & Run All, use Kaggle's Output tab to create a
        **private dataset from the model folder**. Attach that dataset to the
        separate frozen-feature notebook, which runs with **Internet off**.
        Preserve `snapshot_manifest.json` beside `config.json` and all weights.
        Review the upstream Apache-2.0 model license supplied with the snapshot.
        ''') ,
        cell("code", '''\
        # Kaggle normally includes huggingface_hub. Install only if absent;
        # this preparation notebook is the one workflow that uses the network.
        import importlib.util
        import subprocess
        import sys
        if importlib.util.find_spec("huggingface_hub") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from importlib.metadata import version
        print("huggingface_hub", version("huggingface_hub"))
        ''') ,
        cell("code", f'''\
import hashlib
from pathlib import Path
import runpy

PREPARATION_SOURCE = {source!r}
assert hashlib.sha256(PREPARATION_SOURCE.encode()).hexdigest() == {digest!r}
helper_path = Path("/kaggle/working/prepare_model.py")
helper_path.write_text(PREPARATION_SOURCE, encoding="utf-8")
helper_namespace = runpy.run_path(str(helper_path), run_name="ntl_prepare_model")
preparation_result = helper_namespace["main"](["--output-dir", "/kaggle/working/qwen3-8b-base"])
if preparation_result not in (None, 0):
    raise RuntimeError(f"Model preparation failed with exit code {{preparation_result}}")
'''),
        cell("code", '''\
        import json
        snapshot = Path("/kaggle/working/qwen3-8b-base")
        manifest = json.loads((snapshot / "snapshot_manifest.json").read_text())
        print("Repository:", manifest["repo_id"])
        print("Revision:", manifest["revision"])
        print("Files:", len(manifest["files"]))
        print("Includes weights:", manifest["includes_weights"])
        assert manifest["includes_weights"] is True
        print("Create a private Kaggle dataset from:", snapshot)
        '''),
    ]


def make_dependency_preparation() -> list[dict]:
    helper = ROOT / "scripts" / "prepare_wheels.py"
    if not helper.is_file():
        raise SystemExit("scripts/prepare_wheels.py is missing; create it before exporting.")
    source = helper.read_text(encoding="utf-8")
    cpu_lock = wheel_requirements(neural=False)
    neural_lock = wheel_requirements(neural=True)
    return [
        cell("markdown", '''\
        # Prepare the pinned offline dependency bundle

        Run once on **CPU with Internet on**. This downloads Linux wheels for
        the pinned CPU and neural Kaggle environments, including GPU libraries; it
        does not allocate a GPU, train a model, or call a paid API. The bundle
        can take several GB. Keep it separate from the model snapshot output.

        After Save & Run All succeeds, create a **private Kaggle dataset** from
        the `cpu-wheelhouse` and `neural-wheelhouse` output folders. Attach it
        to both offline submission notebooks; each selects its own profile.
        Preserve `wheel_manifest.json`, `requirements.lock`, and every wheel.
        Installation in those notebooks checks the lock and
        uses `pip --no-index --require-hashes`.
        '''),
        cell("code", f'''\
import hashlib
from pathlib import Path
import runpy

working = Path("/kaggle/working")
working.mkdir(parents=True, exist_ok=True)
HELPER_SOURCE = {source!r}
CPU_REQUIREMENTS_LOCK = {cpu_lock!r}
NEURAL_REQUIREMENTS_LOCK = {neural_lock!r}
assert hashlib.sha256(HELPER_SOURCE.encode()).hexdigest() == {hashlib.sha256(source.encode()).hexdigest()!r}
assert hashlib.sha256(CPU_REQUIREMENTS_LOCK.encode()).hexdigest() == {hashlib.sha256(cpu_lock.encode()).hexdigest()!r}
assert hashlib.sha256(NEURAL_REQUIREMENTS_LOCK.encode()).hexdigest() == {hashlib.sha256(neural_lock.encode()).hexdigest()!r}
helper_path = working / "prepare_wheels.py"
helper_path.write_text(HELPER_SOURCE, encoding="utf-8")
helper_namespace = runpy.run_path(str(helper_path), run_name="ntl_prepare_wheels")
PREPARE_NEURAL = True  # False prepares only the smaller CPU baseline bundle.
profiles = [("cpu", CPU_REQUIREMENTS_LOCK, "requirements-cpu.lock")]
if PREPARE_NEURAL:
    profiles.append(("neural", NEURAL_REQUIREMENTS_LOCK, "requirements-kaggle.lock"))
for profile, lock, filename in profiles:
    requirements_path = working / filename
    requirements_path.write_text(lock, encoding="utf-8")
    result = helper_namespace["main"]([
        "--requirements", str(requirements_path),
        "--output-dir", str(working / (profile + "-wheelhouse")),
        "--profile", profile,
    ])
    if result not in (None, 0):
        raise RuntimeError(f"{{profile}} wheel preparation failed with exit code {{result}}")
    print("Create a private Kaggle dataset including:", working / (profile + "-wheelhouse"))
'''),
    ]


def write_notebook(
    path: Path, cells: list[dict], *, gpu: bool, internet: bool = False, check: bool = False
) -> None:
    # Stable cell IDs and deterministic JSON make generated changes reviewable.
    for index, item in enumerate(cells):
        item["id"] = hashlib.sha256(f"{index}:{item['source']}".encode()).hexdigest()[:12]
        if item["cell_type"] == "code":
            tree = ast.parse(item["source"], filename=f"{path}:cell{index}")
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and any(isinstance(target, ast.Name) and target.id in {
                        "version_probe", "gpu_probe", "validation_code"
                    } for target in node.targets)
                ):
                    ast.parse(node.value.value, filename=f"{path}:cell{index}:subprocess")
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"isInternetEnabled": internet, "isGpuEnabled": gpu},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Outdated notebook: {path.name}. Run scripts/build_kaggle_notebooks.py")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "notebooks")
    parser.add_argument("--check", action="store_true", help="Fail if exports differ or code cells do not compile")
    args = parser.parse_args()
    sources = package_sources()
    write_notebook(args.output_dir / "kaggle_prepare_dependencies.ipynb", make_dependency_preparation(), gpu=False, internet=True, check=args.check)
    write_notebook(args.output_dir / "kaggle_prepare_qwen_model.ipynb", make_preparation(), gpu=False, internet=True, check=args.check)
    write_notebook(args.output_dir / "kaggle_cpu_baseline.ipynb", make_cpu(sources), gpu=False, check=args.check)
    write_notebook(args.output_dir / "kaggle_qwen_frozen.ipynb", make_gpu(sources), gpu=True, check=args.check)


if __name__ == "__main__":
    main()
