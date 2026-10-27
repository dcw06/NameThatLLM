#!/usr/bin/env python3
"""Prepare the offline CPU-Qwen blend submission and its two input notebooks; never push or submit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SLUG = "name-that-llm-qwen-model-input"
DEPENDENCY_SLUG = "name-that-llm-neural-wheels"
SUBMISSION_SLUG = "name-that-llm-qwen-blend-submission"
REHEARSAL_ROWS = 4000
REHEARSAL_BELOW_ROWS = 1000


GPU_PROBE = '''\
gpu_probe = """
import torch
import bitsandbytes as bnb
assert torch.cuda.is_available(), 'Enable a Kaggle T4 GPU'
print('GPU:', torch.cuda.get_device_name(0))
layer = bnb.nn.Linear4bit(16, 8, bias=False, compute_dtype=torch.float16,
                         quant_type='nf4').to('cuda:0')
result = layer(torch.randn(2, 16, device='cuda:0', dtype=torch.float16))
assert torch.isfinite(result).all(), 'NF4 smoke probe returned nonfinite values'
print('Tiny NF4 forward probe passed:', tuple(result.shape))
"""
subprocess.run([RUNTIME_PYTHON, "-c", gpu_probe], check=True, env=RUNTIME_ENV)
'''


FIT_AND_PREDICT = '''\
import time
FIT_DIR = "/kaggle/working/blend-model"
started = time.perf_counter()
# Extract every training passage and refit both classifiers on all official rows.
run_runtime("ntl.ensemble", ["fit", "--train", TRAIN, "--model-path", MODEL_DIR,
                             "--recipe", RECIPE_PATH, "--output-dir", FIT_DIR, "--device", "cuda:0"])
print(f"Fit finished after {time.perf_counter() - started:.0f} s")
run_runtime("ntl.ensemble", ["predict", "--model-dir", FIT_DIR, "--test", TEST,
                             "--model-path", MODEL_DIR, "--output", SUBMISSION, "--device", "cuda:0"])
print(f"Current test scored after {time.perf_counter() - started:.0f} s")
'''


REHEARSAL = f'''\
# The visible test has a few example rows; the hidden rerun has about 4,000.
# Only for the small visible file, time the identical predict path on a
# {REHEARSAL_ROWS:,}-row workload of training passages. The scored rerun skips this.
import random
REHEARSAL = {{"ran": False, "current_test_rows": len(test_rows)}}
if len(test_rows) < {REHEARSAL_BELOW_ROWS}:
    sampler = random.Random(2026)
    rehearsal_path = Path("/tmp/rehearsal_test.csv")
    with rehearsal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text"])
        writer.writeheader()
        for index in range({REHEARSAL_ROWS}):
            writer.writerow({{"id": f"rehearsal-{{index:05d}}", "text": sampler.choice(train_rows)["text"]}})
    rehearsal_started = time.perf_counter()
    run_runtime("ntl.ensemble", ["predict", "--model-dir", FIT_DIR, "--test", str(rehearsal_path),
                                 "--model-path", MODEL_DIR, "--output", "/tmp/rehearsal_submission.parquet",
                                 "--device", "cuda:0"])
    REHEARSAL.update(ran=True, rows={REHEARSAL_ROWS},
                     predict_seconds=time.perf_counter() - rehearsal_started,
                     run_seconds_including_fit=time.perf_counter() - started)
    subprocess.run([RUNTIME_PYTHON, "-c", validation_code, "/tmp/rehearsal_submission.parquet",
                    str(rehearsal_path), json.dumps(sorted(VALID_LABELS))], check=True, env=RUNTIME_ENV)
Path("/kaggle/working/rehearsal.json").write_text(json.dumps(REHEARSAL, indent=2) + chr(10))
print(json.dumps(REHEARSAL, indent=2))
'''


def _pinned(value: str, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[a-z0-9-]+/[1-9][0-9]*", value):
        raise ValueError(f"{name} must have an explicit owner/slug/version")
    return value


def _metadata(username: str, slug: str, title: str, *, gpu: bool, internet: bool,
              kernel_sources: list[str]) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", username):
        raise ValueError("Provide a Kaggle username without a slash or URL")
    if len(title) > 50:
        raise ValueError("Notebook title must contain at most 50 characters")
    return {"id": f"{username}/{slug}", "title": title, "code_file": "main.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": internet, "competition_sources": ["name-that-llm"],
            "kernel_sources": kernel_sources, "dataset_sources": [], "model_sources": []}


def _write(output: Path, cells: list[dict], metadata: dict, exporter: dict, extra: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    exporter["write_notebook"](output / "main.ipynb", cells, gpu=metadata["enable_gpu"],
                               internet=metadata["enable_internet"])
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {"schema_version": 1, "kernel_id": metadata["id"], "submitted": False, **extra,
                "files": [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in [output / "main.ipynb", output / "kernel-metadata.json"]]}
    (output / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output_dir": str(output), "kernel_id": metadata["id"]}


def build_inputs(username: str, output_root: str | Path) -> list[dict]:
    """CPU notebooks with Internet on: the pinned Qwen snapshot and the hash-locked wheels."""
    exporter = runpy.run_path(str(ROOT / "scripts" / "build_kaggle_notebooks.py"))
    root = Path(output_root).expanduser().absolute()
    lock_hash = hashlib.sha256((ROOT / "requirements-kaggle.lock").read_bytes()).hexdigest()
    return [
        _write(root / "model", exporter["make_preparation"](),
               _metadata(username, MODEL_SLUG, "Name That LLM Qwen model input", gpu=False, internet=True,
                         kernel_sources=[]), exporter, {"purpose": "pinned Qwen3-8B-Base snapshot"}),
        _write(root / "dependencies", exporter["make_dependency_preparation"](),
               _metadata(username, DEPENDENCY_SLUG, "Name That LLM neural wheels", gpu=False, internet=True,
                         kernel_sources=[]), exporter,
               {"purpose": "hash-locked offline wheels", "requirements_sha256": lock_hash}),
    ]


def build_submission(username: str, output_dir: str | Path, recipe_path: str | Path,
                     model_kernel: str, dependency_kernel: str) -> dict:
    """GPU notebook with Internet off: offline install, full refit, blend, validated submission."""
    _pinned(model_kernel, "Model kernel")
    _pinned(dependency_kernel, "Dependency kernel")
    exporter = runpy.run_path(str(ROOT / "scripts" / "build_kaggle_notebooks.py"))
    sources = exporter["package_sources"]()
    if "ntl/ensemble.py" not in sources:
        raise ValueError("The ensemble module must exist before export")
    recipe_text = Path(recipe_path).read_text(encoding="utf-8")
    recipe = json.loads(recipe_text)
    if recipe.get("kind") != "cpu-qwen-blend-recipe" or not 0 <= recipe.get("weight", -1) <= 1:
        raise ValueError("Expected a frozen cpu-qwen-blend-recipe with a weight in [0, 1]")
    dependencies = exporter["dependency_cell"](neural=True)
    original = 'RUNTIME_DIR = Path("/kaggle/working/ntl_env")'
    if original not in dependencies:
        raise ValueError("Shared dependency cell changed; review the runtime location")
    # Keep the multi-GB environment out of the saved notebook output.
    dependencies = dependencies.replace(original, 'RUNTIME_DIR = Path("/tmp/ntl_env")')
    check = exporter["SUBMISSION_CHECK"]
    cell = exporter["cell"]
    cells = [
        cell("markdown", f'''\
        # Name That LLM: CPU + frozen Qwen3-8B-Base blend

        Offline submission notebook. Attach the official competition data and the
        two pinned input notebooks (Qwen snapshot and neural wheels). Select
        **GPU T4 x2** and **Internet off**. The notebook installs the hash-locked
        runtime, extracts frozen NF4 features for every training passage, refits
        the CPU classifier and the 18-class Qwen head on all official rows, and
        scores the current `test.csv` with `(1 - w) * CPU + w * Qwen`, `w = {recipe["weight"]}`.
        The weight was frozen beforehand by inner prompt folds over all training rows.
        '''),
        cell("code", exporter["bootstrap"](sources)),
        cell("code", dependencies),
        cell("code", exporter["DATA_DISCOVERY"]),
        cell("code", exporter["MODEL_DISCOVERY"]),
        cell("code", GPU_PROBE),
        cell("code", (
            f"RECIPE_JSON = {recipe_text!r}\n"
            f"assert hashlib.sha256(RECIPE_JSON.encode()).hexdigest() == "
            f"{hashlib.sha256(recipe_text.encode()).hexdigest()!r}\n"
            "RECIPE_PATH = '/kaggle/working/recipe.json'\n"
            "Path(RECIPE_PATH).write_text(RECIPE_JSON, encoding='utf-8')\n"
            "print('Frozen blend weight:', json.loads(RECIPE_JSON)['weight'])\n"
        )),
        cell("code", FIT_AND_PREDICT),
        cell("code", check),
        cell("code", REHEARSAL),
        cell("markdown", '''\
        Submit this saved version's `submission.parquet`. `rehearsal.json` records the
        4,000-row predict time on this hardware; `blend-model/fit.json` records the fit.
        '''),
    ]
    metadata = _metadata(username, SUBMISSION_SLUG, "Name That LLM Qwen blend submission", gpu=True,
                         internet=False, kernel_sources=[model_kernel, dependency_kernel])
    return _write(Path(output_dir).expanduser().absolute(), cells, metadata, exporter, {
        "recipe_sha256": hashlib.sha256(recipe_text.encode()).hexdigest(), "weight": recipe["weight"],
        "model_kernel": model_kernel, "dependency_kernel": dependency_kernel,
        "rehearsal_rows": REHEARSAL_ROWS, "rehearsal_below_rows": REHEARSAL_BELOW_ROWS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("inputs")
    inputs.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "blend_inputs")
    submission = commands.add_parser("submission")
    submission.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "blend_submission")
    submission.add_argument("--recipe", type=Path, default=ROOT / "artifacts" / "qwen-blend" / "recipe.json")
    submission.add_argument("--model-kernel", required=True, help="owner/slug/version")
    submission.add_argument("--dependency-kernel", required=True, help="owner/slug/version")
    args = parser.parse_args(argv)
    try:
        if args.command == "inputs":
            result = build_inputs(args.username, args.output_root)
        else:
            result = build_submission(args.username, args.output_dir, args.recipe,
                                      args.model_kernel, args.dependency_kernel)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
