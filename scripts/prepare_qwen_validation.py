#!/usr/bin/env python3
"""Prepare the private 18-class frozen Qwen3-8B-Base validation notebook; never launch or submit it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_KERNEL = "daichongwei06/name-that-llm-cpu-first-submission/1"
DEFAULT_SLUG = "name-that-llm-qwen-frozen-validation"
DEFAULT_TITLE = "Name That LLM Qwen frozen validation"
MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"


GPU_PROBE = '''\
# Stop before the 16 GB download if NF4 kernels do not run on this GPU.
gpu_probe = """
import torch
import bitsandbytes as bnb
assert torch.cuda.is_available(), 'Enable a free Kaggle T4 GPU'
print('GPU:', torch.cuda.get_device_name(0))
print('Free / total GPU bytes:', torch.cuda.mem_get_info(0))
layer = bnb.nn.Linear4bit(16, 8, bias=False, compute_dtype=torch.float16,
                         quant_type='nf4').to('cuda:0')
inputs = torch.randn(2, 16, device='cuda:0', dtype=torch.float16)
result = layer(inputs)
assert torch.isfinite(result).all(), 'NF4 smoke probe returned nonfinite values'
print('Tiny NF4 forward probe passed:', tuple(result.shape))
"""
subprocess.run([RUNTIME_PYTHON, *RUNTIME_ARGS, "-c", gpu_probe], check=True, env=RUNTIME_ENV)
setup_log("NF4 probe passed")
'''


PREFLIGHT = '''\
validation_code = """
import json
from pathlib import Path
import sys
from ntl.data import fingerprint_frame, read_train
from ntl.folds import load_fold_manifest
train = read_train(sys.argv[1])
manifest, metadata = load_fold_manifest(train, sys.argv[2])
assert len(train) == 2699 and metadata['n_splits'] == 5, 'Expected the original five folds over 2,699 rows'
report = {'rows': len(train), 'prompts': int(train['prompt_id'].nunique()),
          'train_fingerprint': fingerprint_frame(train), 'fold_metadata': metadata}
Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + chr(10))
print(json.dumps(report, indent=2))
"""
subprocess.run([RUNTIME_PYTHON, *RUNTIME_ARGS, "-c", validation_code, TRAIN_PATH, FOLD_MANIFEST,
                str(WORKING / "input_protocol.json")], check=True, env=RUNTIME_ENV)
'''


EXPERIMENT = '''\
# Embed all official rows once (label-free), then fit a fresh scaler and head per outer fold.
FEATURES = WORKING / 'features' / 'qwen3-8b-base-train.npz'
FEATURES.parent.mkdir(parents=True, exist_ok=True)
setup_log('Extracting frozen NF4 features for every training passage')
subprocess.run([
    RUNTIME_PYTHON, *RUNTIME_ARGS, '-m', 'ntl.ensemble', 'extract', '--input', TRAIN_PATH,
    '--model-path', MODEL_PATH, '--output', str(FEATURES), '--device', 'cuda:0',
], check=True, env=RUNTIME_ENV)
setup_log('Fitting fold-local 18-class Qwen heads')
subprocess.run([
    RUNTIME_PYTHON, *RUNTIME_ARGS, '-m', 'ntl.ensemble', 'qwen-oof', '--train', TRAIN_PATH,
    '--fold-manifest', FOLD_MANIFEST, '--features', str(FEATURES),
    '--output-dir', str(WORKING / 'qwen-oof'),
], check=True, env=RUNTIME_ENV)
setup_log('Completed frozen Qwen validation')
'''


def model_cell() -> str:
    helper_source = (ROOT / "scripts" / "prepare_model.py").read_text()
    return f'''\
# Pretrained weights stay in temporary storage; only features and reports are saved.
import shutil
PREPARATION_SOURCE = {helper_source!r}
assert hashlib.sha256(PREPARATION_SOURCE.encode()).hexdigest() == {hashlib.sha256(helper_source.encode()).hexdigest()!r}
MODEL_HELPER = Path("/tmp/ntl_prepare_qwen_model.py")
MODEL_HELPER.write_text(PREPARATION_SOURCE, encoding="utf-8")
MODEL_PATH = "/tmp/ntl_qwen_model/qwen3-8b-base"
setup_log("Free temporary disk bytes: " + str(shutil.disk_usage("/tmp").free))
prepare_code = """
import runpy
import sys
namespace = runpy.run_path(sys.argv[1], run_name='ntl_prepare_qwen_snapshot')
result = namespace['main'](['--repo-id', sys.argv[2], '--revision', sys.argv[3], '--output-dir', sys.argv[4]])
if result not in (None, 0):
    raise RuntimeError('Pinned Qwen3-8B-Base preparation failed')
"""
download_env = dict(RUNTIME_ENV)
download_env['HF_HUB_OFFLINE'] = '0'
download_env['TRANSFORMERS_OFFLINE'] = '0'
subprocess.run([RUNTIME_PYTHON, *RUNTIME_ARGS, '-c', prepare_code, str(MODEL_HELPER),
                {MODEL_ID!r}, {MODEL_REVISION!r}, MODEL_PATH], check=True, env=download_env)
RUNTIME_ENV['HF_HUB_OFFLINE'] = '1'
RUNTIME_ENV['TRANSFORMERS_OFFLINE'] = '1'
setup_log("Pinned snapshot ready at " + MODEL_PATH)
'''


def build_bundle(
    username: str, output_dir: str | Path, *, slug: str = DEFAULT_SLUG, title: str = DEFAULT_TITLE,
    baseline_kernel: str = BASELINE_KERNEL,
) -> dict:
    frozen = runpy.run_path(str(ROOT / "scripts" / "prepare_frozen_gemini_experiment.py"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", username):
        raise ValueError("Provide a Kaggle username without a slash or URL")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise ValueError("Notebook slug must use lowercase letters, digits, and hyphens")
    if not title.strip() or len(title) > 50:
        raise ValueError("Notebook title must contain 1 to 50 characters")
    frozen["pinned_kernel"](baseline_kernel, "Baseline kernel")
    exporter = runpy.run_path(str(ROOT / "scripts" / "build_kaggle_notebooks.py"))
    shared = runpy.run_path(str(ROOT / "scripts" / "prepare_gemini_experiment.py"))
    if runpy.run_path(str(ROOT / "scripts" / "prepare_model.py"))["MODEL_SPECS"][MODEL_ID]["revision"] != MODEL_REVISION:
        raise ValueError("Snapshot helper must contain the reviewed Qwen3-8B-Base pin")
    sources = exporter["package_sources"]()
    if not {"ntl/ensemble.py", "ntl/neural.py"}.issubset(sources):
        raise ValueError("Frozen Qwen and ensemble modules must exist before export")
    lock = (ROOT / "requirements-kaggle.lock").read_text()
    lock_hash = hashlib.sha256(lock.encode()).hexdigest()
    runtime = frozen["scratch_runtime"](shared["RUNTIME"])
    runtime = runtime.replace("/tmp/ntl_frozen_gemini_env", "/tmp/ntl_qwen_env")
    runtime = runtime.replace("/tmp/ntl_frozen_pip_cache", "/tmp/ntl_qwen_pip_cache")
    cell = exporter["cell"]
    protocol = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "encoder_training": False,
        "retained_layers": 24, "pool_layers": [7, 22], "quantization": "nf4", "compute_dtype": "float16",
        "max_length": 512, "batch_size": 2, "head": "StandardScaler + LogisticRegression(C=0.005)",
        "outer_folds": "Original five prompt-grouped CPU folds", "rows": 2699,
        "head_fitting": "Fresh scaler and 18-class head on each outer training partition only",
        "seed": 2026, "competition_submission": False,
    }
    cells = [
        cell("markdown", '''\
        # Frozen Qwen3-8B-Base: 18-class prompt-grouped validation

        Private validation run for the existing frozen recipe: the first 24 decoder
        blocks of pinned `Qwen/Qwen3-8B-Base` in NF4, masked mean pooling of blocks
        7 and 22, and a fresh scaled logistic head fitted on each of the original
        five prompt folds' training partitions. Every official passage receives one
        out-of-fold 18-class score vector; the cached features support a nested
        CPU-Qwen blend evaluated separately.

        Select **T4 x2**, **Internet on**, and **Private**. One GPU is used. The
        pinned environment and weights stay under `/tmp`; output contains source,
        features, OOF scores and reports. No competition submission is created.
        '''),
        cell("code", exporter["bootstrap"](sources)),
        cell("code", f"BASELINE_KERNEL_REFERENCE = {baseline_kernel!r}\nRUNTIME_KERNEL_REFERENCE = None\n"
             + frozen["LOCATORS"]),
        cell("code", (
            f"REQUIREMENTS_LOCK = {lock!r}\nREQUIREMENTS_SHA256 = {lock_hash!r}\n"
            f"EXPECTED_RUNTIME_PACKAGES = {exporter['requirements'](neural=True)!r}\n"
            "RUNTIME_REUSE_ENABLED = False\n" + runtime
        )),
        cell("code", GPU_PROBE),
        cell("code", frozen["INPUT_LOCATORS"] + PREFLIGHT),
        cell("code", f"PROTOCOL = {protocol!r}\n"
             "(WORKING / 'qwen_protocol.json').write_text(json.dumps(PROTOCOL, indent=2) + chr(10))\n"),
        cell("code", model_cell()),
        cell("code", EXPERIMENT),
        cell("markdown", '''\
        Review `qwen-oof/metrics.json` (overall and per-fold balanced accuracy, per-class
        recall) and `features/qwen3-8b-base-train.json` (runtime, peak GPU memory,
        truncation). Release the GPU runtime when the run finishes.
        '''),
    ]
    output = Path(output_dir).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    exporter["write_notebook"](output / "main.ipynb", cells, gpu=True, internet=True)
    metadata = {
        "id": f"{username}/{slug}", "title": title, "code_file": "main.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "competition_sources": ["name-that-llm"], "kernel_sources": [baseline_kernel],
        "dataset_sources": [], "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "kernel_id": metadata["id"], "protocol": protocol,
        "baseline_kernel": baseline_kernel, "requirements_sha256": lock_hash, "submitted": False,
        "files": [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in [output / "main.ipynb", output / "kernel-metadata.json"]],
    }
    (output / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output_dir": str(output), "kernel_id": metadata["id"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "qwen_validation")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--baseline-kernel", default=BASELINE_KERNEL)
    args = parser.parse_args(argv)
    try:
        result = build_bundle(args.username, args.output_dir, slug=args.slug, title=args.title,
                              baseline_kernel=args.baseline_kernel)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
