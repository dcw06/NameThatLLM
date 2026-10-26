#!/usr/bin/env python3
"""Prepare a private DeBERTa Gemini diagnostic notebook; never launch or submit it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "microsoft/deberta-v3-base"
MODEL_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
BASELINE_KERNEL = "daichongwei06/name-that-llm-cpu-first-submission/1"
DEFAULT_SLUG = "name-that-llm-gemini-deberta-diagnostic"
DEFAULT_TITLE = "Name That LLM Gemini DeBERTa diagnostic"


RUNTIME = '''\
# A clean environment prevents Kaggle's optional audio/vision packages from
# affecting Transformers. Only this diagnostic setup stage uses the network.
import subprocess
import venv

WORKING = Path("/kaggle/working")
WORKING.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = WORKING / "gemini_env"
RUNTIME_ENV = dict(os.environ)
RUNTIME_ENV.pop("PYTHONHOME", None)
RUNTIME_ENV["PYTHONPATH"] = str(SOURCE_ROOT)
RUNTIME_ENV["PYTHONNOUSERSITE"] = "1"
RUNTIME_ENV["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
LOCK_PATH = WORKING / "gemini-requirements.lock"
LOCK_PATH.write_text(REQUIREMENTS_LOCK, encoding="utf-8")
assert hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest() == REQUIREMENTS_SHA256
probe_code = """
from importlib.metadata import version
from packaging.requirements import Requirement
import json
from pathlib import Path
import platform
import sys
import torch
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
for raw_requirement in json.loads(sys.argv[2]):
    requirement = Requirement(raw_requirement)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    installed = version(requirement.name)
    if installed not in requirement.specifier:
        raise ValueError(f'Runtime version mismatch: {requirement}; installed {installed}')
assert torch.cuda.is_available(), 'Enable a free Kaggle T4 GPU for this diagnostic'
report = {
    'python': sys.version, 'platform': platform.platform(),
    'gpu': torch.cuda.get_device_name(0),
    'gpu_count': torch.cuda.device_count(),
    'gpu_memory_free_total_bytes': list(torch.cuda.mem_get_info(0)),
    'versions': {name: version(name) for name in [
        'torch', 'transformers', 'numpy', 'pandas', 'scikit-learn', 'sentencepiece',
        'huggingface-hub', 'tokenizers', 'pyarrow', 'safetensors',
    ]},
}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + chr(10))
print(json.dumps(report, indent=2))
"""
def probe_runtime(python, arguments, environment):
    subprocess.run([str(python), *arguments, "-c", probe_code,
                    str(WORKING / "environment.json"), json.dumps(EXPECTED_RUNTIME_PACKAGES)],
                   check=True, env=environment)

RUNTIME_PYTHON = None
RUNTIME_ARGS = []
if RUNTIME_REUSE_ENABLED:
    # Saved binaries can lose execute permission. Use this image's interpreter
    # with -S and only the attached packages, excluding global site packages.
    for candidate in sorted(Path("/kaggle/input").rglob("gemini_env")):
        previous_lock = candidate.parent / "gemini-requirements.lock"
        previous_report = candidate.parent / "environment.json"
        if not previous_lock.is_file() or not previous_report.is_file():
            continue
        if hashlib.sha256(previous_lock.read_bytes()).hexdigest() != REQUIREMENTS_SHA256:
            continue
        try:
            saved_report = json.loads(previous_report.read_text())
            saved_python = saved_report['python'].split()[0].split('.')[:2]
        except (OSError, ValueError, KeyError, AttributeError):
            continue
        if saved_python != [str(sys.version_info.major), str(sys.version_info.minor)]:
            continue
        site_packages = candidate / 'lib' / ('python' + '.'.join(saved_python)) / 'site-packages'
        if not (site_packages / 'torch').is_dir() or not (site_packages / 'nvidia').is_dir():
            continue
        candidate_env = dict(RUNTIME_ENV)
        candidate_env['PYTHONPATH'] = os.pathsep.join([str(SOURCE_ROOT), str(site_packages)])
        try:
            probe_runtime(sys.executable, ['-S'], candidate_env)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            print("Saved runtime is unusable; preparing a fresh environment:", str(error))
            continue
        RUNTIME_PYTHON = sys.executable
        RUNTIME_ARGS = ['-S']
        RUNTIME_ENV = candidate_env
        print("Reusing pinned, CUDA-checked runtime:", RUNTIME_PYTHON)
        print("Attached site packages:", site_packages)
        break
    if RUNTIME_PYTHON is None:
        print("No intact saved runtime matched the lock; using a fresh environment")
if RUNTIME_PYTHON is None:
    # Kaggle's Python can disable ensurepip. Existing notebook pip can manage
    # a pip-less environment through its supported --python option.
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(RUNTIME_DIR)
    RUNTIME_PYTHON = str(RUNTIME_DIR / "bin" / "python")
    subprocess.run(
        [sys.executable, "-m", "pip", "--python", RUNTIME_PYTHON,
         "install", "--index-url", "https://pypi.org/simple",
         "--require-hashes", "-r", str(LOCK_PATH)],
        check=True, env=RUNTIME_ENV,
    )
    probe_runtime(RUNTIME_PYTHON, RUNTIME_ARGS, RUNTIME_ENV)
'''


INPUTS = '''\
# Use only the official training data and the pinned first CPU notebook's OOF
# output. No hidden competition test data or submission artifact is needed.
TRAIN_PATH = None
if TRAIN_PATH is None:
    train_candidates = [
        p for p in [
            Path("/kaggle/input/competitions/name-that-llm/train.csv"),
            Path("/kaggle/input/name-that-llm/train.csv"),
        ] if p.is_file()
    ]
    if len(train_candidates) != 1:
        raise RuntimeError("Attach the official competition data; set TRAIN_PATH if ambiguous")
    TRAIN_PATH = str(train_candidates[0])

BASELINE_DIR = None
if BASELINE_DIR is None:
    baseline_candidates = [
        p.parent for p in Path("/kaggle/input").rglob("oof.parquet")
        if p.parent.name == "cpu-cv"
        and (p.parent / "folds.csv").is_file()
        and (p.parent / "folds.json").is_file()
    ]
    if len(baseline_candidates) != 1:
        raise RuntimeError(
            "Attach the pinned CPU first-submission version 1 outputs, including "
            "cpu-cv/oof.parquet, folds.csv and folds.json; set BASELINE_DIR if ambiguous"
        )
    BASELINE_DIR = str(baseline_candidates[0])
BASELINE_OOF = str(Path(BASELINE_DIR) / "oof.parquet")
FOLD_MANIFEST = str(Path(BASELINE_DIR) / "folds.csv")
print("Official training data:", TRAIN_PATH)
print("Pinned CPU OOF:", BASELINE_OOF)
print("Existing prompt folds:", FOLD_MANIFEST)
preflight_code = """
import json
from pathlib import Path
import sys
from ntl.gemini import validate_experiment_inputs
protocol = validate_experiment_inputs(sys.argv[1], sys.argv[2], sys.argv[3])
Path(sys.argv[4]).write_text(json.dumps(protocol, indent=2) + chr(10))
print(json.dumps(protocol, indent=2))
"""
subprocess.run([
    RUNTIME_PYTHON, *RUNTIME_ARGS, "-c", preflight_code, TRAIN_PATH, BASELINE_OOF,
    FOLD_MANIFEST, str(WORKING / "input_protocol.json"),
], check=True, env=RUNTIME_ENV)
'''


def download_cell() -> str:
    helper_source = (ROOT / "scripts" / "prepare_model.py").read_text()
    helper_hash = hashlib.sha256(helper_source.encode()).hexdigest()
    return f'''\
MODEL_PATH = "/kaggle/working/deberta-v3-base"
if RUNTIME_REUSE_ENABLED:
    reusable_models = []
    for manifest_path in Path("/kaggle/input").rglob("snapshot_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if (manifest.get("repo_id") == {MODEL_ID!r}
                and manifest.get("revision") == {MODEL_REVISION!r}
                and manifest.get("includes_weights") is True):
            reusable_models.append(manifest_path.parent)
    if len(reusable_models) == 1:
        model_probe = (
            "from ntl.gemini import validate_model_snapshot; import sys; "
            "validate_model_snapshot(sys.argv[1]); print('Saved DeBERTa snapshot validated')"
        )
        try:
            subprocess.run([RUNTIME_PYTHON, *RUNTIME_ARGS, "-c", model_probe, str(reusable_models[0])],
                           check=True, env=RUNTIME_ENV)
        except (OSError, subprocess.CalledProcessError) as error:
            print("Saved snapshot cannot be reused; downloading the pinned revision:", str(error))
        else:
            MODEL_PATH = str(reusable_models[0])
            print("Reusing pinned local snapshot:", MODEL_PATH)
MODEL_PREPARATION_SOURCE = {helper_source!r}
assert hashlib.sha256(MODEL_PREPARATION_SOURCE.encode()).hexdigest() == {helper_hash!r}
model_helper_path = WORKING / "prepare_deberta_model.py"
model_helper_path.write_text(MODEL_PREPARATION_SOURCE, encoding="utf-8")
download_code = """
import runpy
namespace = runpy.run_path('/kaggle/working/prepare_deberta_model.py', run_name='ntl_prepare_deberta')
result = namespace['main']([
    '--repo-id', {MODEL_ID!r}, '--revision', {MODEL_REVISION!r},
    '--output-dir', '/kaggle/working/deberta-v3-base',
])
if result not in (None, 0):
    raise RuntimeError('Pinned DeBERTa snapshot preparation failed')
"""
download_env = dict(RUNTIME_ENV)
download_env["HF_HUB_OFFLINE"] = "0"
download_env["TRANSFORMERS_OFFLINE"] = "0"
if MODEL_PATH == "/kaggle/working/deberta-v3-base":
    subprocess.run([RUNTIME_PYTHON, *RUNTIME_ARGS, "-c", download_code], check=True, env=download_env)
# Training remains entirely local even though notebook Internet is enabled
# for the setup stage.
RUNTIME_ENV["HF_HUB_OFFLINE"] = "1"
RUNTIME_ENV["TRANSFORMERS_OFFLINE"] = "1"
'''


def build_bundle(
    username: str,
    output_dir: str | Path,
    *,
    slug: str = DEFAULT_SLUG,
    title: str = DEFAULT_TITLE,
    baseline_kernel: str = BASELINE_KERNEL,
    runtime_kernel: str | None = None,
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", username):
        raise ValueError("Provide a Kaggle username without a slash or URL")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise ValueError("Notebook slug must use lowercase letters, digits, and hyphens")
    if not title.strip() or len(title) > 50:
        raise ValueError("Notebook title must contain 1 to 50 characters")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[a-z0-9-]+/[1-9][0-9]*", baseline_kernel):
        raise ValueError("Baseline kernel must have an explicit owner/slug/version")
    if runtime_kernel is not None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[a-z0-9-]+/[1-9][0-9]*", runtime_kernel):
            raise ValueError("Runtime kernel must have an explicit owner/slug/version")
        if runtime_kernel.rsplit("/", 1)[0] == f"{username}/{slug}":
            raise ValueError("Use a new notebook slug when reusing an earlier runtime")
    exporter = runpy.run_path(str(ROOT / "scripts" / "build_kaggle_notebooks.py"))
    sources = exporter["package_sources"]()
    if "ntl/gemini.py" not in sources:
        raise ValueError("src/ntl/gemini.py must exist before exporting this diagnostic")
    lock = (ROOT / "requirements-kaggle.lock").read_text()
    lock_hash = hashlib.sha256(lock.encode()).hexdigest()
    cell = exporter["cell"]
    cells = [
        cell("markdown", '''\
        # DeBERTa-v3-base: Gemini version diagnostic

        Private diagnostic on the official training data and the first CPU
        notebook's saved out-of-fold predictions. This notebook trains across
        all 600 true Gemini passages using the existing prompt-grouped folds.
        It reports exact-version performance separately on the 570 passages
        that the CPU baseline routed to the Gemini family correctly.

        Select **T4 x2** and **Internet on**. The experiment uses one GPU.
        Setup installs a pinned environment and downloads the public MIT
        DeBERTa-v3-base snapshot. Training uses only local files. The selected
        570-row evaluation subset never controls which Gemini training rows
        are used. No competition submission is created by this diagnostic.
        '''),
        cell("code", exporter["bootstrap"](sources)),
        cell("code", (
            f"REQUIREMENTS_LOCK = {lock!r}\nREQUIREMENTS_SHA256 = {lock_hash!r}\n"
            f"EXPECTED_RUNTIME_PACKAGES = {exporter['requirements'](neural=True)!r}\n"
            f"RUNTIME_REUSE_ENABLED = {runtime_kernel is not None!r}\n" + RUNTIME
        )),
        cell("code", INPUTS),
        cell("code", download_cell()),
        cell("code", '''\
        OUTPUT_DIR = "/kaggle/working/gemini-diagnostic"
        subprocess.run([
            RUNTIME_PYTHON, *RUNTIME_ARGS, "-m", "ntl.gemini",
            "--train", TRAIN_PATH, "--baseline-oof", BASELINE_OOF,
            "--fold-manifest", FOLD_MANIFEST, "--model-path", MODEL_PATH,
            "--output-dir", OUTPUT_DIR,
            "--epochs", "3", "--learning-rate", "2e-5", "--max-length", "512",
            "--batch-size", "8", "--gradient-accumulation", "2",
            "--warmup-ratio", "0.1", "--seed", "2026",
        ], check=True, env=RUNTIME_ENV)
        print((Path(OUTPUT_DIR) / "metrics.json").read_text())
        '''),
        cell("markdown", '''\
        Review `gemini-diagnostic/metrics.json`, per-class reports and confusion
        matrices, plus predictions for all 600 Gemini passages and the selected
        570. These are prompt-grouped diagnostics on a defined conditional
        subset; the selected result does not measure an end-to-end competition
        classifier. The notebook does not produce `submission.parquet` or call
        Kaggle's submission API. Release the runtime when finished.
        '''),
    ]
    output = Path(output_dir).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    exporter["write_notebook"](output / "main.ipynb", cells, gpu=True, internet=True)
    metadata = {
        "id": f"{username}/{slug}", "title": title, "code_file": "main.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "competition_sources": ["name-that-llm"],
        "kernel_sources": list(dict.fromkeys([baseline_kernel] + ([runtime_kernel] if runtime_kernel else []))),
        "dataset_sources": [], "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "kernel_id": metadata["id"],
        "purpose": "prompt-grouped Gemini exact-version diagnostic; no competition submission",
        "baseline_kernel": baseline_kernel, "model_id": MODEL_ID,
        "runtime_kernel": runtime_kernel,
        "model_revision": MODEL_REVISION, "requirements_sha256": lock_hash,
        "submitted": False,
        "files": [
            {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in [output / "main.ipynb", output / "kernel-metadata.json"]
        ],
    }
    (output / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output_dir": str(output), "kernel_id": metadata["id"], "model_revision": MODEL_REVISION}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "gemini_experiment")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--baseline-kernel", default=BASELINE_KERNEL)
    parser.add_argument("--runtime-kernel", help="Optional pinned owner/slug/version with reusable runtime and snapshot outputs")
    args = parser.parse_args(argv)
    try:
        result = build_bundle(
            args.username, args.output_dir, slug=args.slug, title=args.title,
            baseline_kernel=args.baseline_kernel,
            runtime_kernel=args.runtime_kernel,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
