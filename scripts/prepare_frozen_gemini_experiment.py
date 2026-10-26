#!/usr/bin/env python3
"""Prepare private frozen-encoder Gemini diagnostics; never launch or submit them."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_KERNEL = "daichongwei06/name-that-llm-cpu-first-submission/1"
RUNTIME_KERNEL = "daichongwei06/name-that-llm-gemini-deberta-run/1"
DEFAULT_SLUG = "name-that-llm-frozen-gemini-embeddings"
DEFAULT_TITLE = "Name That LLM frozen Gemini embeddings"
MODELS = [
    {
        "key": "styledistance", "repo_id": "StyleDistance/styledistance",
        "revision": "b7df5f0b0480773c097ba3121d83ca32b71015ca",
        "pooling": "masked mean, including special tokens; explicit L2 normalization",
    },
    {
        "key": "qwen3-embedding-0.6b", "repo_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "pooling": "last non-padding token; L2 normalization",
    },
]


LOCATORS = '''\
# Check only the known mount-root layouts for the pinned notebook references.
# Never walk the attached multi-GB Python environment to discover its parent.
import time
SETUP_STARTED = time.monotonic()
def setup_log(message):
    print(f"[{time.monotonic() - SETUP_STARTED:.1f}s] {message}", flush=True)

def kernel_output_roots(reference):
    if reference is None:
        return []
    owner, slug, version = reference.split('/')
    candidates = [
        Path('/kaggle/input/notebooks') / owner / slug,
        Path('/kaggle/input') / slug,
        Path('/kaggle/input') / owner / slug,
        Path('/kaggle/input/notebooks') / slug,
    ]
    found = []
    for root in candidates:
        setup_log(f'Checking pinned notebook {owner}/{slug}/{version} at {root}')
        if root.is_dir():
            found.append(root)
    return found

def runtime_candidates():
    return [root / 'gemini_env' for root in kernel_output_roots(RUNTIME_KERNEL_REFERENCE)
            if (root / 'gemini_env').is_dir()]

setup_log('Starting bounded input discovery and runtime preparation')
'''


INPUT_LOCATORS = '''\
TRAIN_PATH = None
for candidate in [
    Path('/kaggle/input/competitions/name-that-llm/train.csv'),
    Path('/kaggle/input/name-that-llm/train.csv'),
]:
    if candidate.is_file():
        TRAIN_PATH = str(candidate)
        break
if TRAIN_PATH is None:
    raise RuntimeError('The official Name That LLM training input is missing')

baseline_candidates = [
    root / 'cpu-cv' for root in kernel_output_roots(BASELINE_KERNEL_REFERENCE)
    if all((root / 'cpu-cv' / filename).is_file()
           for filename in ['oof.parquet', 'folds.csv', 'folds.json'])
]
if not baseline_candidates:
    raise RuntimeError('Pinned CPU notebook output lacks cpu-cv/oof.parquet and fold manifests')
BASELINE_DIR = str(baseline_candidates[0])
BASELINE_OOF = str(Path(BASELINE_DIR) / 'oof.parquet')
FOLD_MANIFEST = str(Path(BASELINE_DIR) / 'folds.csv')
setup_log('Official train: ' + TRAIN_PATH)
setup_log('Pinned baseline OOF: ' + BASELINE_OOF)
'''


def pinned_kernel(value: str, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[a-z0-9-]+/[1-9][0-9]*", value):
        raise ValueError(f"{name} must have an explicit owner/slug/version")
    return value


def scratch_runtime(source: str) -> str:
    """Reuse the tested runtime probe while keeping newly installed binaries temporary."""
    source = source.replace(
        'RUNTIME_DIR = WORKING / "gemini_env"',
        'RUNTIME_DIR = Path("/tmp/ntl_frozen_gemini_env")',
    )
    source = source.replace(
        'RUNTIME_ENV["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"',
        'RUNTIME_ENV["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"\n'
        'RUNTIME_ENV["PIP_CACHE_DIR"] = "/tmp/ntl_frozen_pip_cache"',
    )
    if 'RUNTIME_DIR = Path("/tmp/ntl_frozen_gemini_env")' not in source:
        raise ValueError("Shared runtime template changed; review the scratch runtime location")
    traversal = 'for candidate in sorted(Path("/kaggle/input").rglob("gemini_env")):'
    if traversal not in source:
        raise ValueError("Shared runtime discovery template changed; review bounded input discovery")
    source = source.replace(traversal, 'for candidate in runtime_candidates():')
    source = source.replace(
        'def probe_runtime(python, arguments, environment):\n',
        'def probe_runtime(python, arguments, environment):\n'
        '    setup_log("Starting pinned-version/CUDA runtime probe (180-second limit)")\n',
    )
    source = source.replace(
        '[str(python), *arguments, "-c", probe_code,',
        '[str(python), *arguments, "-u", "-c", probe_code,',
    )
    source = source.replace(
        'check=True, env=environment)',
        'check=True, env=environment, timeout=180)\n'
        '    setup_log("Pinned-version/CUDA runtime probe passed")',
    )
    source = source.replace(
        'except (OSError, subprocess.CalledProcessError, ValueError) as error:',
        'except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as error:',
    )
    source = source.replace(
        'import torch\nfrom transformers',
        'print("Runtime probe: importing torch", flush=True)\nimport torch\n'
        'print("Runtime probe: importing Transformers", flush=True)\nfrom transformers',
    )
    source = source.replace(
        'assert torch.cuda.is_available(),',
        'print("Runtime probe: checking CUDA", flush=True)\nassert torch.cuda.is_available(),',
    )
    source = source.replace(
        'if RUNTIME_PYTHON is None:\n    # Kaggle',
        'if RUNTIME_PYTHON is None:\n    setup_log("Preparing fresh temporary pinned environment")\n    # Kaggle',
    )
    return source


def preparation_cell() -> str:
    helper_source = (ROOT / "scripts" / "prepare_model.py").read_text()
    return f'''\
# Download only the two pinned public checkpoints. Pretrained weights stay in
# temporary storage; output uploads contain features and diagnostic reports.
MODELS = {MODELS!r}
PREPARATION_SOURCE = {helper_source!r}
assert hashlib.sha256(PREPARATION_SOURCE.encode()).hexdigest() == {hashlib.sha256(helper_source.encode()).hexdigest()!r}
MODEL_HELPER = Path("/tmp/ntl_prepare_frozen_models.py")
MODEL_HELPER.write_text(PREPARATION_SOURCE, encoding="utf-8")
prepare_code = """
import runpy
import sys
namespace = runpy.run_path(sys.argv[1], run_name='ntl_prepare_frozen_snapshot')
result = namespace['main']([
    '--repo-id', sys.argv[2], '--revision', sys.argv[3], '--output-dir', sys.argv[4],
])
if result not in (None, 0):
    raise RuntimeError('Pinned frozen-encoder preparation failed')
"""
download_env = dict(RUNTIME_ENV)
download_env['HF_HUB_OFFLINE'] = '0'
download_env['TRANSFORMERS_OFFLINE'] = '0'
MODEL_PATHS = {{}}
for model in MODELS:
    snapshot = Path('/tmp/ntl_frozen_models') / model['key']
    subprocess.run([
        RUNTIME_PYTHON, *RUNTIME_ARGS, '-c', prepare_code, str(MODEL_HELPER),
        model['repo_id'], model['revision'], str(snapshot),
    ], check=True, env=download_env)
    MODEL_PATHS[model['key']] = str(snapshot)
RUNTIME_ENV['HF_HUB_OFFLINE'] = '1'
RUNTIME_ENV['TRANSFORMERS_OFFLINE'] = '1'
'''


EXPERIMENT = '''\
FEATURE_DIR = WORKING / 'features'
RESULT_DIR = WORKING / 'frozen-gemini'
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
comparison = {}
for model in MODELS:
    key = model['key']
    features = FEATURE_DIR / (key + '.npz')
    output = RESULT_DIR / key
    # Frozen inference uses float32, no instruction prefix, a 512-token limit
    # and explicit L2 normalization for both encoders. No encoder is fine-tuned.
    subprocess.run([
        RUNTIME_PYTHON, *RUNTIME_ARGS, '-m', 'ntl.encoders',
        '--train', TRAIN_PATH, '--baseline-oof', BASELINE_OOF,
        '--fold-manifest', FOLD_MANIFEST, '--model-key', key,
        '--model-path', MODEL_PATHS[key], '--output', str(features),
        '--device', 'cuda:0', '--batch-size', '16', '--max-length', '512',
    ], check=True, env=RUNTIME_ENV)
    subprocess.run([
        RUNTIME_PYTHON, *RUNTIME_ARGS, '-m', 'ntl.frozen_gemini',
        '--train', TRAIN_PATH, '--baseline-oof', BASELINE_OOF,
        '--fold-manifest', FOLD_MANIFEST, '--features', str(features),
        '--output-dir', str(output), '--C-grid', '0.001', '0.01', '0.1', '1', '10',
        '--inner-folds', '3', '--seed', '2026', '--max-iter', '2000',
    ], check=True, env=RUNTIME_ENV)
    comparison[key] = json.loads((output / 'metrics.json').read_text())
    print(key, json.dumps(comparison[key], indent=2))
(RESULT_DIR / 'comparison.json').write_text(json.dumps(comparison, indent=2) + chr(10))
print('Completed both frozen-encoder diagnostics:', RESULT_DIR)
'''


def build_bundle(
    username: str,
    output_dir: str | Path,
    *,
    slug: str = DEFAULT_SLUG,
    title: str = DEFAULT_TITLE,
    baseline_kernel: str = BASELINE_KERNEL,
    runtime_kernel: str | None = RUNTIME_KERNEL,
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", username):
        raise ValueError("Provide a Kaggle username without a slash or URL")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise ValueError("Notebook slug must use lowercase letters, digits, and hyphens")
    if not title.strip() or len(title) > 50:
        raise ValueError("Notebook title must contain 1 to 50 characters")
    pinned_kernel(baseline_kernel, "Baseline kernel")
    if runtime_kernel is not None:
        pinned_kernel(runtime_kernel, "Runtime kernel")
        if runtime_kernel.rsplit("/", 1)[0] == f"{username}/{slug}":
            raise ValueError("Runtime reuse requires a different notebook slug")
    exporter = runpy.run_path(str(ROOT / "scripts" / "build_kaggle_notebooks.py"))
    shared = runpy.run_path(str(ROOT / "scripts" / "prepare_gemini_experiment.py"))
    snapshot_specs = runpy.run_path(str(ROOT / "scripts" / "prepare_model.py"))["MODEL_SPECS"]
    for model in MODELS:
        if snapshot_specs.get(model["repo_id"], {}).get("revision") != model["revision"]:
            raise ValueError("Snapshot helper must contain the reviewed frozen-encoder pins")
    sources = exporter["package_sources"]()
    if not {"ntl/encoders.py", "ntl/frozen_gemini.py"}.issubset(sources):
        raise ValueError("Frozen encoder and evaluator modules must exist before export")
    lock = (ROOT / "requirements-kaggle.lock").read_text()
    lock_hash = hashlib.sha256(lock.encode()).hexdigest()
    cell = exporter["cell"]
    protocol = {
        "models": MODELS, "encoder_training": False, "dtype": "float32",
        "max_length": 512, "batch_size": 16, "instruction_prefix": None,
        "l2_normalize_both_encoders": True,
        "training_population": "All 600 true Gemini rows in the fitting prompt folds",
        "reporting_cohort": "All 600 true Gemini rows and the fixed 570 correctly routed by CPU",
        "outer_folds": "Existing five prompt-grouped CPU folds",
        "C_grid": [0.001, 0.01, 0.1, 1, 10], "inner_folds": 3,
        "C_selection": "Balanced accuracy on inner prompt-grouped folds only",
        "seed": 2026, "competition_submission": False,
    }
    cells = [
        cell("markdown", '''\
        # Frozen StyleDistance and Qwen3 embeddings for Gemini versions

        Private diagnostic: frozen `StyleDistance/styledistance` and
        `Qwen/Qwen3-Embedding-0.6B` embeddings followed by regularized logistic
        regression. Existing outer prompt folds define held-out predictions;
        three inner prompt folds choose regularization independently in each
        outer fitting set. Report all 600 true Gemini rows and the same fixed
        570-row correctly-routed cohort from the first CPU baseline.

        Select **T4 x2**, **Internet on**, and **Private**. One GPU is used.
        The notebook first tries the pinned successful diagnostic's environment,
        then prepares a clean temporary environment if reuse is unavailable.
        New environments and downloaded weights stay under `/tmp`; only source,
        feature caches, metadata and diagnostic reports are saved as output.
        No encoder is fine-tuned and no competition submission is created.
        '''),
        cell("code", exporter["bootstrap"](sources)),
        cell("code", (
            f"BASELINE_KERNEL_REFERENCE = {baseline_kernel!r}\n"
            f"RUNTIME_KERNEL_REFERENCE = {runtime_kernel!r}\n" + LOCATORS
        )),
        cell("code", (
            f"REQUIREMENTS_LOCK = {lock!r}\nREQUIREMENTS_SHA256 = {lock_hash!r}\n"
            f"EXPECTED_RUNTIME_PACKAGES = {exporter['requirements'](neural=True)!r}\n"
            f"RUNTIME_REUSE_ENABLED = {runtime_kernel is not None!r}\n"
            + scratch_runtime(shared["RUNTIME"])
        )),
        cell("code", INPUT_LOCATORS + shared["INPUTS"][shared["INPUTS"].index('preflight_code ='):]),
        cell("code", (
            f"PROTOCOL = {protocol!r}\n"
            "(WORKING / 'frozen_protocol.json').write_text(json.dumps(PROTOCOL, indent=2) + chr(10))\n"
        )),
        cell("code", preparation_cell()),
        cell("code", EXPERIMENT),
        cell("markdown", '''\
        Review `frozen-gemini/comparison.json`, per-model metrics, selected-row
        predictions, per-class recall changes and confusion matrices. Features
        are cached under `features/`. The 570-row result measures exact-version
        performance conditional on the CPU router selecting the correct family;
        it does not establish an end-to-end competition score. Release the GPU
        runtime when both diagnostics finish.
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
        "schema_version": 1, "kernel_id": metadata["id"], "protocol": protocol,
        "baseline_kernel": baseline_kernel, "runtime_kernel": runtime_kernel,
        "requirements_sha256": lock_hash, "submitted": False,
        "files": [
            {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in [output / "main.ipynb", output / "kernel-metadata.json"]
        ],
    }
    (output / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output_dir": str(output), "kernel_id": metadata["id"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "frozen_gemini_experiment")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--baseline-kernel", default=BASELINE_KERNEL)
    parser.add_argument("--runtime-kernel", default=RUNTIME_KERNEL)
    parser.add_argument("--fresh-runtime", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_bundle(
            args.username, args.output_dir, slug=args.slug, title=args.title,
            baseline_kernel=args.baseline_kernel,
            runtime_kernel=None if args.fresh_runtime else args.runtime_kernel,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
