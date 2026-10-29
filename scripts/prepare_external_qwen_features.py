#!/usr/bin/env python3
"""Prepare the private Kaggle notebook that extracts frozen NF4 Qwen3-8B-Base features for the 316
matched external GLM passages (docs/glm-pair-qwen.md); never launch or submit it.

The notebook reuses the frozen Qwen validation kernel's pinned runtime, NF4 probe and model
snapshot cells unchanged. It attaches the public CC BY 4.0 dataset, checks the audited parquet's
SHA-256, selects exactly the matched passages by row_id, and runs `ntl.ensemble extract` on them.
It also re-extracts the first 16 official passages of train.csv, in their original batch pairs,
so the run's environment can be compared with the cached official features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "name-that-llm-external-glm-qwen-features"
DEFAULT_TITLE = "Name That LLM external GLM Qwen features"
DATASET = "maniceet/name-that-llm-openrouter-openai-generations"
PARQUET = "ntl_openrouter_openai_generations.parquet"
REFERENCE_ROWS = 16


INPUTS = '''\
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
EXTERNAL_PATH = None
owner, slug = EXTERNAL_DATASET.split('/')
for candidate in [
    Path('/kaggle/input/datasets') / owner / slug / EXTERNAL_FILE,
    Path('/kaggle/input') / slug / EXTERNAL_FILE,
    Path('/kaggle/input') / owner / slug / EXTERNAL_FILE,
]:
    if candidate.is_file():
        EXTERNAL_PATH = str(candidate)
        break
if EXTERNAL_PATH is None:
    raise RuntimeError('Attach the external dataset ' + EXTERNAL_DATASET)
digest = hashlib.sha256(Path(EXTERNAL_PATH).read_bytes()).hexdigest()
if digest != EXTERNAL_SHA256:
    raise RuntimeError('The attached external parquet is not the audited release: ' + digest)
setup_log('Official train: ' + TRAIN_PATH)
setup_log('Audited external parquet: ' + EXTERNAL_PATH)
INPUT_DIR = WORKING / 'inputs'
INPUT_DIR.mkdir(parents=True, exist_ok=True)
(INPUT_DIR / 'matched_ids.json').write_text(json.dumps(MATCHED_IDS))
prepare_code = """
import json
import sys
from pathlib import Path
import pandas as pd
from ntl.data import read_train
external = pd.read_parquet(sys.argv[1]).set_index('row_id')
ids = json.loads(Path(sys.argv[2]).read_text())
matched = external.loc[ids]
counts = matched['model'].value_counts().to_dict()
assert len(matched) == len(set(ids)) == int(sys.argv[5]), 'Unexpected matched passage count'
assert counts == json.loads(sys.argv[6]), counts
matched.reset_index().rename(columns={'row_id': 'id'})[['id', 'text']].to_csv(sys.argv[3], index=False)
train = read_train(sys.argv[4])
train.head(int(sys.argv[7]))[['id', 'text']].to_csv(sys.argv[8], index=False)
print(json.dumps({'external_rows': len(matched), 'by_version': counts, 'reference_rows': int(sys.argv[7])}))
"""
EXTERNAL_CSV = str(INPUT_DIR / 'external-glm.csv')
REFERENCE_CSV = str(INPUT_DIR / 'official-reference.csv')
subprocess.run([RUNTIME_PYTHON, *RUNTIME_ARGS, '-c', prepare_code, EXTERNAL_PATH, str(INPUT_DIR / 'matched_ids.json'),
                EXTERNAL_CSV, TRAIN_PATH, str(len(MATCHED_IDS)), json.dumps(MATCHED_COUNTS), str(REFERENCE_ROWS),
                REFERENCE_CSV], check=True, env=RUNTIME_ENV)
'''


EXTRACT = '''\
# The same frozen recipe as the official passages: NF4, 24 blocks, blocks 7 and 22, batch size 2.
FEATURES = WORKING / 'features'
FEATURES.mkdir(parents=True, exist_ok=True)
for name, source in (('external-glm', EXTERNAL_CSV), ('official-reference', REFERENCE_CSV)):
    setup_log('Extracting frozen NF4 features: ' + name)
    subprocess.run([
        RUNTIME_PYTHON, *RUNTIME_ARGS, '-m', 'ntl.ensemble', 'extract', '--input', source,
        '--model-path', MODEL_PATH, '--output', str(FEATURES / f'qwen3-8b-base-{name}.npz'), '--device', 'cuda:0',
    ], check=True, env=RUNTIME_ENV)
setup_log('Completed external feature extraction')
'''


def matched_ids() -> tuple[list[str], dict[str, int]]:
    """The 316 matched passages, exactly as ntl.glm_pair selects them."""
    sys.path.insert(0, str(ROOT / "src"))
    import pandas as pd

    from ntl import external_data, glm_pair

    pool = pd.read_parquet(ROOT / "artifacts" / "external-data-v2" / "pool.parquet").set_index("id")
    matched = glm_pair.matched_external(
        external_data.load_external(ROOT / "data" / "external" / "ntl-openrouter-openai-v2" / PARQUET),
        pool["link"].fillna(""))
    return matched["row_id"].tolist(), {key: int(value) for key, value in matched["model"].value_counts().items()}


def build_bundle(username: str, output_dir: str | Path, *, slug: str = DEFAULT_SLUG,
                 title: str = DEFAULT_TITLE) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", username):
        raise ValueError("Provide a Kaggle username without a slash or URL")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise ValueError("Notebook slug must use lowercase letters, digits, and hyphens")
    if not title.strip() or len(title) > 50:
        raise ValueError("Notebook title must contain 1 to 50 characters")
    validation = runpy.run_path(str(ROOT / "scripts" / "prepare_qwen_validation.py"))
    frozen = runpy.run_path(str(ROOT / "scripts" / "prepare_frozen_gemini_experiment.py"))
    exporter = runpy.run_path(str(ROOT / "scripts" / "build_kaggle_notebooks.py"))
    shared = runpy.run_path(str(ROOT / "scripts" / "prepare_gemini_experiment.py"))
    ids, counts = matched_ids()
    from ntl.external_data import SOURCE_SHA256 as external_sha256
    if len(ids) != 316 or counts != {"glm-5.3": 158, "glm-5.3-flash": 158}:
        raise ValueError(f"Expected the 316 matched passages, found {len(ids)} ({counts})")
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
        "model_id": validation["MODEL_ID"], "model_revision": validation["MODEL_REVISION"], "encoder_training": False,
        "retained_layers": 24, "pool_layers": [7, 22], "quantization": "nf4", "compute_dtype": "float16",
        "max_length": 512, "batch_size": 2, "external_dataset": DATASET, "external_sha256": external_sha256,
        "external_rows": len(ids), "external_by_version": counts, "reference_rows": REFERENCE_ROWS,
        "competition_submission": False,
    }
    cells = [
        cell("markdown", '''\
        # Frozen Qwen3-8B-Base features for matched external GLM passages

        Private feature-extraction run: the first 24 decoder blocks of pinned
        `Qwen/Qwen3-8B-Base` in NF4, masked mean pooling of blocks 7 and 22, exactly as
        the official passages' cached features. Inputs are the 316 matched GLM 5.3 /
        GLM 5.3 Flash passages of the attached CC BY 4.0 dataset (Maniceet Sahay, "Name
        That LLM: OpenRouter + OpenAI Generations", 2026), checked against the audited
        file hash, and the first 16 official training passages for an environment check.

        Select **T4 x2**, **Internet on**, and **Private**. Output contains only features
        and their reports. No competition submission is created.
        '''),
        cell("code", exporter["bootstrap"](sources)),
        cell("code", "BASELINE_KERNEL_REFERENCE = None\nRUNTIME_KERNEL_REFERENCE = None\n" + frozen["LOCATORS"]),
        cell("code", (
            f"REQUIREMENTS_LOCK = {lock!r}\nREQUIREMENTS_SHA256 = {lock_hash!r}\n"
            f"EXPECTED_RUNTIME_PACKAGES = {exporter['requirements'](neural=True)!r}\n"
            "RUNTIME_REUSE_ENABLED = False\n" + runtime
        )),
        cell("code", validation["GPU_PROBE"]),
        cell("code", (
            f"EXTERNAL_DATASET = {DATASET!r}\nEXTERNAL_FILE = {PARQUET!r}\nEXTERNAL_SHA256 = {external_sha256!r}\n"
            f"MATCHED_IDS = {ids!r}\nMATCHED_COUNTS = {counts!r}\nREFERENCE_ROWS = {REFERENCE_ROWS}\n" + INPUTS
        )),
        cell("code", f"PROTOCOL = {protocol!r}\n"
             "(WORKING / 'external_qwen_protocol.json').write_text(json.dumps(PROTOCOL, indent=2) + chr(10))\n"),
        cell("code", validation["model_cell"]()),
        cell("code", EXTRACT),
        cell("markdown", '''\
        Download `features/` (both archives and their `.json` reports). Release the GPU
        runtime when the run finishes.
        '''),
    ]
    output = Path(output_dir).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    exporter["write_notebook"](output / "main.ipynb", cells, gpu=True, internet=True)
    metadata = {
        "id": f"{username}/{slug}", "title": title, "code_file": "main.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "competition_sources": ["name-that-llm"], "kernel_sources": [],
        "dataset_sources": [DATASET], "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "kernel_id": metadata["id"], "protocol": protocol,
        "requirements_sha256": lock_hash, "submitted": False,
        "files": [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in [output / "main.ipynb", output / "kernel-metadata.json"]],
    }
    (output / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output_dir": str(output), "kernel_id": metadata["id"], "external_rows": len(ids)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "external_qwen_features")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    args = parser.parse_args(argv)
    try:
        result = build_bundle(args.username, args.output_dir, slug=args.slug, title=args.title)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
