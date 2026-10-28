#!/usr/bin/env python3
"""Prepare the offline Claude-GLM specialist submission and its StyleDistance input notebook; never push or submit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_KERNEL = "daichongwei06/name-that-llm-qwen-model-input/1"
DEPENDENCY_KERNEL = "daichongwei06/name-that-llm-neural-wheels/1"
STYLEDISTANCE_SLUG = "name-that-llm-styledistance-model-input"
SUBMISSION_SLUG = "name-that-llm-claude-glm-specialist"
STYLEDISTANCE_REPO = "StyleDistance/styledistance"
STYLEDISTANCE_REVISION = "b7df5f0b0480773c097ba3121d83ca32b71015ca"
REHEARSAL_ROWS = 4000
REHEARSAL_BELOW_ROWS = 1000


STYLEDISTANCE_DISCOVERY = f'''\
# Attach the StyleDistance snapshot prepared by the StyleDistance model-input notebook.
# Extraction validates every file against its manifest before loading.
STYLEDISTANCE_DIR = None  # e.g. "/kaggle/input/your-styledistance-snapshot"
if STYLEDISTANCE_DIR is None:
    candidates = []
    for manifest_path in Path("/kaggle/input").rglob("snapshot_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if (manifest.get("repo_id") == "{STYLEDISTANCE_REPO}"
                and manifest.get("revision") == "{STYLEDISTANCE_REVISION}"
                and manifest.get("includes_weights") is True):
            candidates.append(manifest_path.parent)
    if len(candidates) != 1:
        raise RuntimeError("Attach one pinned StyleDistance snapshot; found " + str(candidates))
    STYLEDISTANCE_DIR = str(candidates[0])
print("StyleDistance snapshot:", STYLEDISTANCE_DIR)
'''


FIT_AND_PREDICT = '''\
import time
FIT_DIR = "/kaggle/working/blend-model"
SPECIALIST_DIR = "/kaggle/working/specialist-model"
started = time.perf_counter()
# The current system: CPU classifier and frozen-Qwen head refit on every official row.
run_runtime("ntl.ensemble", ["fit", "--train", TRAIN, "--model-path", MODEL_DIR,
                             "--recipe", RECIPE_PATH, "--output-dir", FIT_DIR, "--device", "cuda:0"])
print(f"Current system fit after {time.perf_counter() - started:.0f} s")
# The specialist: the blend fit's own Qwen features plus StyleDistance, Claude and GLM rows.
run_runtime("ntl.claude_glm", ["fit", "--train", TRAIN, "--blend-dir", FIT_DIR,
                               "--styledistance-path", STYLEDISTANCE_DIR, "--recipe", SPECIALIST_RECIPE_PATH,
                               "--output-dir", SPECIALIST_DIR, "--device", "cuda:0"])
print(f"Specialist fit after {time.perf_counter() - started:.0f} s")
run_runtime("ntl.claude_glm", ["predict", "--specialist-dir", SPECIALIST_DIR, "--blend-dir", FIT_DIR,
                               "--test", TEST, "--model-path", MODEL_DIR,
                               "--styledistance-path", STYLEDISTANCE_DIR, "--output", SUBMISSION,
                               "--device", "cuda:0"])
print(f"Current test scored after {time.perf_counter() - started:.0f} s")
print(Path("/kaggle/working/submission_profile.json").read_text())
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
    run_runtime("ntl.claude_glm", ["predict", "--specialist-dir", SPECIALIST_DIR, "--blend-dir", FIT_DIR,
                                   "--test", str(rehearsal_path), "--model-path", MODEL_DIR,
                                   "--styledistance-path", STYLEDISTANCE_DIR,
                                   "--output", "/tmp/rehearsal_submission.parquet", "--device", "cuda:0"])
    REHEARSAL.update(ran=True, rows={REHEARSAL_ROWS},
                     predict_seconds=time.perf_counter() - rehearsal_started,
                     run_seconds_including_fit=time.perf_counter() - started)
    subprocess.run([RUNTIME_PYTHON, "-c", validation_code, "/tmp/rehearsal_submission.parquet",
                    str(rehearsal_path), json.dumps(sorted(VALID_LABELS))], check=True, env=RUNTIME_ENV)
Path("/kaggle/working/rehearsal.json").write_text(json.dumps(REHEARSAL, indent=2) + chr(10))
print(json.dumps(REHEARSAL, indent=2))
'''


def _helpers() -> tuple[dict, dict]:
    return (runpy.run_path(str(ROOT / "scripts" / "build_kaggle_notebooks.py")),
            runpy.run_path(str(ROOT / "scripts" / "prepare_blend_submission.py")))


def build_styledistance_input(username: str, output_dir: str | Path) -> dict:
    """CPU notebook with Internet on: the pinned StyleDistance snapshot, with its manifest."""
    exporter, helper = _helpers()
    source = (ROOT / "scripts" / "prepare_model.py").read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()
    cell = exporter["cell"]
    cells = [
        cell("markdown", '''\
        # Prepare the pinned StyleDistance model input

        Run once in a **CPU notebook with Internet on**. It downloads the public
        StyleDistance encoder at the project's exact revision and writes its snapshot
        manifest under `/kaggle/working/styledistance`. The offline specialist
        submission attaches this notebook's output with **Internet off**.
        '''),
        cell("code", '''\
        import importlib.util
        import subprocess
        import sys
        if importlib.util.find_spec("huggingface_hub") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from importlib.metadata import version
        print("huggingface_hub", version("huggingface_hub"))
        '''),
        cell("code", f'''\
import hashlib
from pathlib import Path
import runpy

PREPARATION_SOURCE = {source!r}
assert hashlib.sha256(PREPARATION_SOURCE.encode()).hexdigest() == {digest!r}
helper_path = Path("/kaggle/working/prepare_model.py")
helper_path.write_text(PREPARATION_SOURCE, encoding="utf-8")
helper_namespace = runpy.run_path(str(helper_path), run_name="ntl_prepare_model")
preparation_result = helper_namespace["main"](["--output-dir", "/kaggle/working/styledistance",
                                               "--repo-id", "{STYLEDISTANCE_REPO}",
                                               "--revision", "{STYLEDISTANCE_REVISION}"])
if preparation_result not in (None, 0):
    raise RuntimeError(f"Model preparation failed with exit code {{preparation_result}}")
helper_path.unlink()
'''),
        cell("code", f'''\
        import json
        snapshot = Path("/kaggle/working/styledistance")
        manifest = json.loads((snapshot / "snapshot_manifest.json").read_text())
        assert manifest["repo_id"] == "{STYLEDISTANCE_REPO}"
        assert manifest["revision"] == "{STYLEDISTANCE_REVISION}"
        assert manifest["includes_weights"] is True
        print("Files:", len(manifest["files"]))
        '''),
    ]
    metadata = helper["_metadata"](username, STYLEDISTANCE_SLUG, "Name That LLM StyleDistance model input",
                                   gpu=False, internet=True, kernel_sources=[])
    return helper["_write"](Path(output_dir).expanduser().absolute(), cells, metadata, exporter,
                            {"purpose": "pinned StyleDistance snapshot", "prepare_model_sha256": digest})


def build_submission(username: str, output_dir: str | Path, current_recipe: str | Path,
                     specialist_recipe: str | Path, styledistance_kernel: str, *, slug: str = SUBMISSION_SLUG,
                     title: str = "Name That LLM Claude-GLM specialist") -> dict:
    """GPU notebook with Internet off: refit the current system, fit the specialist, score test.csv."""
    exporter, helper = _helpers()
    helper["_pinned"](styledistance_kernel, "StyleDistance kernel")
    sources = exporter["package_sources"]()
    if "ntl/claude_glm.py" not in sources:
        raise ValueError("The specialist module must exist before export")
    current_text = Path(current_recipe).read_text(encoding="utf-8")
    specialist_text = Path(specialist_recipe).read_text(encoding="utf-8")
    current, specialist = json.loads(current_text), json.loads(specialist_text)
    if current.get("kind") != "cpu-qwen-blend-recipe" or current.get("weight") != 0.25:
        raise ValueError("Expected the validated current recipe (weight 0.25)")
    if specialist.get("kind") != "claude-glm-specialist-recipe":
        raise ValueError("Expected a frozen Claude-GLM specialist recipe")
    current_sha = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
    if specialist.get("current_recipe_sha256") != current_sha:
        raise ValueError("The specialist recipe was frozen for a different current system")
    dependencies = exporter["dependency_cell"](neural=True)
    original = 'RUNTIME_DIR = Path("/kaggle/working/ntl_env")'
    if original not in dependencies:
        raise ValueError("Shared dependency cell changed; review the runtime location")
    # Keep the multi-GB environment out of the saved notebook output.
    dependencies = dependencies.replace(original, 'RUNTIME_DIR = Path("/tmp/ntl_env")')
    cell = exporter["cell"]
    cells = [
        cell("markdown", f'''\
        # Name That LLM: current system + Claude-GLM specialist

        Offline submission notebook. Attach the official competition data and the three
        pinned input notebooks (Qwen snapshot, StyleDistance snapshot, neural wheels).
        Select **GPU T4 x2** and **Internet off**. The notebook refits the CPU + frozen
        Qwen3-8B-Base blend on every official row, fits a seven-class Claude-GLM head on
        the blend fit's Qwen features plus frozen StyleDistance features, and scores the
        current `test.csv`: on the five Claude and two GLM classes, `p' = (1 - s) p + s M q`,
        `s = {specialist["weight"]}`, where M is the blend's mass on those classes and q the
        specialist's distribution. Both weights were frozen beforehand by inner prompt folds.
        '''),
        cell("code", exporter["bootstrap"](sources)),
        cell("code", dependencies),
        cell("code", exporter["DATA_DISCOVERY"]),
        cell("code", exporter["MODEL_DISCOVERY"]),
        cell("code", STYLEDISTANCE_DISCOVERY),
        cell("code", helper["GPU_PROBE"]),
        cell("code", (
            f"RECIPE_JSON = {current_text!r}\n"
            f"assert hashlib.sha256(RECIPE_JSON.encode()).hexdigest() == "
            f"{hashlib.sha256(current_text.encode()).hexdigest()!r}\n"
            "RECIPE_PATH = '/kaggle/working/recipe.json'\n"
            "Path(RECIPE_PATH).write_text(RECIPE_JSON, encoding='utf-8')\n"
            f"SPECIALIST_RECIPE_JSON = {specialist_text!r}\n"
            f"assert hashlib.sha256(SPECIALIST_RECIPE_JSON.encode()).hexdigest() == "
            f"{hashlib.sha256(specialist_text.encode()).hexdigest()!r}\n"
            "SPECIALIST_RECIPE_PATH = '/kaggle/working/specialist-recipe.json'\n"
            "Path(SPECIALIST_RECIPE_PATH).write_text(SPECIALIST_RECIPE_JSON, encoding='utf-8')\n"
            "print('Blend weight:', json.loads(RECIPE_JSON)['weight'],\n"
            "      '| specialist weight:', json.loads(SPECIALIST_RECIPE_JSON)['weight'])\n"
        )),
        cell("code", FIT_AND_PREDICT),
        cell("code", exporter["SUBMISSION_CHECK"]),
        cell("code", REHEARSAL),
        cell("markdown", '''\
        Submit this saved version's `submission.parquet`. `submission_profile.json` records how
        many rows the specialist changed; `rehearsal.json` records the 4,000-row predict time.
        '''),
    ]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise ValueError("Notebook slug must use lowercase letters, digits, and hyphens")
    metadata = helper["_metadata"](username, slug, title, gpu=True, internet=False,
                                   kernel_sources=[MODEL_KERNEL, DEPENDENCY_KERNEL, styledistance_kernel])
    return helper["_write"](Path(output_dir).expanduser().absolute(), cells, metadata, exporter, {
        "current_recipe_sha256": hashlib.sha256(current_text.encode()).hexdigest(),
        "specialist_recipe_sha256": hashlib.sha256(specialist_text.encode()).hexdigest(),
        "specialist_weight": specialist["weight"], "styledistance_kernel": styledistance_kernel,
        "rehearsal_rows": REHEARSAL_ROWS, "rehearsal_below_rows": REHEARSAL_BELOW_ROWS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("styledistance-input")
    model.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "styledistance_input")
    submission = commands.add_parser("submission")
    submission.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "specialist_submission")
    submission.add_argument("--current-recipe", type=Path, default=ROOT / "artifacts" / "qwen-blend" / "recipe.json")
    submission.add_argument("--specialist-recipe", type=Path,
                            default=ROOT / "artifacts" / "claude-glm" / "recipe.json")
    submission.add_argument("--styledistance-kernel", required=True, help="owner/slug/version")
    args = parser.parse_args(argv)
    if args.command == "styledistance-input":
        result = build_styledistance_input(args.username, args.output_dir)
    else:
        result = build_submission(args.username, args.output_dir, args.current_recipe, args.specialist_recipe,
                                  args.styledistance_kernel)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
