#!/usr/bin/env python3
"""Prepare offline QLoRA probe or prompt-fold notebooks; never push or submit them."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_KERNEL = "daichongwei06/name-that-llm-qwen-model-input/1"
DEPENDENCY_KERNEL = "daichongwei06/name-that-llm-neural-wheels/1"
FEATURE_KERNEL = "daichongwei06/name-that-llm-qwen-frozen-validation/1"


FEATURES = '''\
# Cached frozen features of every training passage (validated run); the head starts from them.
feature_candidates = sorted(Path("/kaggle/input").rglob("qwen3-8b-base-train.npz"))
if len(feature_candidates) != 1:
    raise RuntimeError("Attach the frozen-validation notebook output; found " + str(feature_candidates))
FEATURES = str(feature_candidates[0])
print("Cached frozen features:", FEATURES)
'''


RUN = '''\
import subprocess
import time
OUTPUT = Path("/kaggle/working/lora")
OUTPUT.mkdir(parents=True, exist_ok=True)
gpus = int(subprocess.run([RUNTIME_PYTHON, "-c", "import torch; print(torch.cuda.device_count())"],
                          check=True, env=RUNTIME_ENV, capture_output=True, text=True).stdout.strip())
print("GPUs:", gpus, "| mode:", MODE, "| folds:", FOLDS)
common = ["--train", TRAIN, "--plan", PLAN_PATH, "--features", FEATURES, "--model-path", MODEL_DIR,
          "--output-dir", str(OUTPUT), "--device", "cuda:0"]
if MODE == "probe":
    jobs = [["probe", *common]]
else:
    # One process per GPU; each process loads the backbone once and runs its folds in order.
    slots = [FOLDS[index::gpus] for index in range(min(gpus, len(FOLDS)))]
    jobs = [["fold", *common, "--outer-fold", *[str(fold) for fold in slot]] for slot in slots]
started = time.perf_counter()
processes = []
for gpu, job in enumerate(jobs):
    environment = dict(RUNTIME_ENV, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    log = open(OUTPUT / f"gpu{gpu}.log", "w")
    processes.append((subprocess.Popen([RUNTIME_PYTHON, "-m", "ntl.lora", *job], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT), log, gpu))
failed = []
for process, log, gpu in processes:
    code = process.wait()
    log.close()
    print(f"--- GPU {gpu} exited {code} after {time.perf_counter() - started:.0f} s ---")
    print((OUTPUT / f"gpu{gpu}.log").read_text()[-6000:])
    if code != 0:
        failed.append(gpu)
if failed:
    raise RuntimeError(f"LoRA job failed on GPU(s) {failed}; see lora/gpu*.log")
'''


FINAL = '''\
import subprocess
import time
OUTPUT = Path("/kaggle/working/lora_final")
OUTPUT.mkdir(parents=True, exist_ok=True)
common = ["--train", TRAIN, "--plan", PLAN_PATH, "--features", FEATURES, "--output-dir", str(OUTPUT)]
started = time.perf_counter()
# 1. The two full-data inner folds in parallel, one per GPU.
processes = []
for inner_fold in (0, 1):
    environment = dict(RUNTIME_ENV, CUDA_VISIBLE_DEVICES=str(inner_fold), PYTHONUNBUFFERED="1")
    log = open(OUTPUT / f"inner{inner_fold}.log", "w")
    processes.append((subprocess.Popen(
        [RUNTIME_PYTHON, "-m", "ntl.lora_final", "inner", *common, "--model-path", MODEL_DIR,
         "--device", "cuda:0", "--inner-fold", str(inner_fold)],
        env=environment, stdout=log, stderr=subprocess.STDOUT), log, inner_fold))
for process, log, inner_fold in processes:
    code = process.wait()
    log.close()
    print((OUTPUT / f"inner{inner_fold}.log").read_text()[-3000:])
    if code != 0:
        raise RuntimeError(f"Final inner fold {inner_fold} failed")
print(f"Inner folds finished after {time.perf_counter() - started:.0f} s")
# 2. Epochs, then the blend weight, from the inner rows only (CPU).
run_runtime("ntl.lora_final", ["choose", *common])
# 3. Adapters and head on every official row.
subprocess.run([RUNTIME_PYTHON, "-m", "ntl.lora_final", "fit", *common, "--model-path", MODEL_DIR,
                "--device", "cuda:0", "--recipe", str(OUTPUT / "recipe.json")],
               env=dict(RUNTIME_ENV, CUDA_VISIBLE_DEVICES="0"), check=True)
print(f"Final adapters saved after {time.perf_counter() - started:.0f} s:", OUTPUT / "final")
'''


SUBMISSION = '''\
import time
final_candidates = [path.parent for path in Path("/kaggle/input").rglob("head.npz")
                    if (path.parent / "recipe.json").is_file() and (path.parent / "adapter").is_dir()]
if len(final_candidates) != 1:
    raise RuntimeError("Attach exactly one fitted adapter output; found " + str(final_candidates))
FINAL_DIR = str(final_candidates[0])
print("Fitted adapter:", FINAL_DIR, json.loads((Path(FINAL_DIR) / "recipe.json").read_text())["adapted_weight"])
FIT_DIR = "/kaggle/working/blend-model"
started = time.perf_counter()
# The current system: CPU classifier and frozen-Qwen head refit on every official row.
run_runtime("ntl.ensemble", ["fit", "--train", TRAIN, "--model-path", MODEL_DIR,
                             "--recipe", RECIPE_PATH, "--output-dir", FIT_DIR, "--device", "cuda:0"])
run_runtime("ntl.lora_final", ["predict", "--final-dir", FINAL_DIR, "--blend-dir", FIT_DIR, "--test", TEST,
                               "--model-path", MODEL_DIR, "--output", SUBMISSION, "--device", "cuda:0"])
print(f"Current test scored after {time.perf_counter() - started:.0f} s")
'''


REHEARSAL = '''\
# Visible test only: time the identical predict path on 4,000 training passages.
import random
REHEARSAL = {"ran": False, "current_test_rows": len(test_rows)}
if len(test_rows) < 1000:
    sampler = random.Random(2026)
    rehearsal_path = Path("/tmp/rehearsal_test.csv")
    with rehearsal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text"])
        writer.writeheader()
        for index in range(4000):
            writer.writerow({"id": f"rehearsal-{index:05d}", "text": sampler.choice(train_rows)["text"]})
    rehearsal_started = time.perf_counter()
    run_runtime("ntl.lora_final", ["predict", "--final-dir", FINAL_DIR, "--blend-dir", FIT_DIR,
                                   "--test", str(rehearsal_path), "--model-path", MODEL_DIR,
                                   "--output", "/tmp/rehearsal_submission.parquet", "--device", "cuda:0"])
    REHEARSAL.update(ran=True, rows=4000, predict_seconds=time.perf_counter() - rehearsal_started,
                     run_seconds_including_fit=time.perf_counter() - started)
    subprocess.run([RUNTIME_PYTHON, "-c", validation_code, "/tmp/rehearsal_submission.parquet",
                    str(rehearsal_path), json.dumps(sorted(VALID_LABELS))], check=True, env=RUNTIME_ENV)
Path("/kaggle/working/rehearsal.json").write_text(json.dumps(REHEARSAL, indent=2) + chr(10))
print(json.dumps(REHEARSAL, indent=2))
'''


def build_submission(username: str, output_dir: str | Path, current_recipe: str | Path, final_kernel: str,
                     slug: str, title: str) -> dict:
    """Offline T4 submission: refit the current system, then blend with the fitted adapters."""
    helper = runpy.run_path(str(ROOT / "scripts" / "prepare_blend_submission.py"))
    helper["_pinned"](final_kernel, "Fitted adapter kernel")
    exporter = runpy.run_path(str(ROOT / "scripts" / "build_kaggle_notebooks.py"))
    sources = exporter["package_sources"]()
    recipe_text = Path(current_recipe).read_text(encoding="utf-8")
    if json.loads(recipe_text).get("weight") != 0.25:
        raise ValueError("The current system's validated recipe has weight 0.25")
    dependencies = exporter["dependency_cell"](neural=True).replace(
        'RUNTIME_DIR = Path("/kaggle/working/ntl_env")', 'RUNTIME_DIR = Path("/tmp/ntl_env")')
    cell = exporter["cell"]
    cells = [
        cell("markdown", '''\
        # Name That LLM: current system + QLoRA-adapted Qwen3-8B-Base

        Offline T4 submission. Refits the current CPU + frozen-Qwen system on every official
        row, loads the backbone once with the fitted adapters, and scores the current
        `test.csv` with `(1 - v) * current + v * adapted`, using the frozen recipe's v.
        '''),
        cell("code", exporter["bootstrap"](sources)),
        cell("code", dependencies),
        cell("code", exporter["DATA_DISCOVERY"]),
        cell("code", exporter["MODEL_DISCOVERY"]),
        cell("code", helper["GPU_PROBE"]),
        cell("code", (
            f"RECIPE_JSON = {recipe_text!r}\n"
            f"assert hashlib.sha256(RECIPE_JSON.encode()).hexdigest() == "
            f"{hashlib.sha256(recipe_text.encode()).hexdigest()!r}\n"
            "RECIPE_PATH = '/kaggle/working/recipe.json'\n"
            "Path(RECIPE_PATH).write_text(RECIPE_JSON, encoding='utf-8')\n"
        )),
        cell("code", SUBMISSION),
        cell("code", exporter["SUBMISSION_CHECK"]),
        cell("code", REHEARSAL),
    ]
    metadata = helper["_metadata"](username, slug, title, gpu=True, internet=False,
                                   kernel_sources=[MODEL_KERNEL, DEPENDENCY_KERNEL, final_kernel])
    return helper["_write"](Path(output_dir).expanduser().absolute(), cells, metadata, exporter,
                            {"final_kernel": final_kernel,
                             "current_recipe_sha256": hashlib.sha256(recipe_text.encode()).hexdigest()})


def build(username: str, output_dir: str | Path, plan_path: str | Path, mode: str, folds: list[int],
          slug: str, title: str) -> dict:
    if mode not in {"probe", "folds", "final"} or (mode == "folds" and (not folds or set(folds) - set(range(5)))):
        raise ValueError("Use mode probe or final, or mode folds with outer folds from 0 to 4")
    helper = runpy.run_path(str(ROOT / "scripts" / "prepare_blend_submission.py"))
    exporter = runpy.run_path(str(ROOT / "scripts" / "build_kaggle_notebooks.py"))
    sources = exporter["package_sources"]()
    if "ntl/lora.py" not in sources:
        raise ValueError("The LoRA module must exist before export")
    plan_text = Path(plan_path).read_text(encoding="utf-8")
    plan_hash = hashlib.sha256(plan_text.encode()).hexdigest()
    dependencies = exporter["dependency_cell"](neural=True)
    original = 'RUNTIME_DIR = Path("/kaggle/working/ntl_env")'
    if original not in dependencies:
        raise ValueError("Shared dependency cell changed; review the runtime location")
    dependencies = dependencies.replace(original, 'RUNTIME_DIR = Path("/tmp/ntl_env")')
    cell = exporter["cell"]
    cells = [
        cell("markdown", f'''\
        # Name That LLM: QLoRA adaptation of frozen Qwen3-8B-Base ({mode})

        Offline T4 notebook. The NF4 backbone, tokenizer, 512-token limit, 24 retained blocks
        and pooling of blocks 7 and 22 are the frozen recipe. Rank-16 adapters on blocks 1-22
        train jointly with the frozen recipe's own head (initialised from its C = 0.005
        logistic fit). Inner prompt folds of each outer training partition choose the learning
        rate and duration (0 = frozen head); outer folds are scored once. Attach the competition data,
        the Qwen snapshot, the neural wheels and the frozen-validation features.
        '''),
        cell("code", exporter["bootstrap"](sources)),
        cell("code", dependencies),
        cell("code", exporter["DATA_DISCOVERY"]),
        cell("code", exporter["MODEL_DISCOVERY"]),
        cell("code", helper["GPU_PROBE"]),
        cell("code", FEATURES),
        cell("code", (
            f"PLAN_CSV = {plan_text!r}\n"
            f"assert hashlib.sha256(PLAN_CSV.encode()).hexdigest() == {plan_hash!r}\n"
            "PLAN_PATH = '/kaggle/working/lora_plan.csv'\n"
            "Path(PLAN_PATH).write_text(PLAN_CSV, encoding='utf-8')\n"
            f"MODE = {mode!r}\nFOLDS = {list(folds)!r}\n"
        )),
        cell("code", FINAL if mode == "final" else RUN),
    ]
    metadata = helper["_metadata"](username, slug, title, gpu=True, internet=False,
                                   kernel_sources=[MODEL_KERNEL, DEPENDENCY_KERNEL, FEATURE_KERNEL])
    return helper["_write"](Path(output_dir).expanduser().absolute(), cells, metadata, exporter,
                            {"mode": mode, "folds": list(folds), "plan_sha256": plan_hash,
                             "kernel_sources": [MODEL_KERNEL, DEPENDENCY_KERNEL, FEATURE_KERNEL]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "artifacts" / "lora" / "plan.csv")
    parser.add_argument("--mode", choices=("probe", "folds", "final", "submission"), required=True)
    parser.add_argument("--folds", type=int, nargs="*", default=[])
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-recipe", type=Path, default=ROOT / "artifacts" / "qwen-blend" / "recipe.json")
    parser.add_argument("--final-kernel", help="owner/slug/version of the fitted-adapter notebook")
    args = parser.parse_args(argv)
    try:
        if args.mode == "submission":
            result = build_submission(args.username, args.output_dir, args.current_recipe, args.final_kernel or "",
                                      args.slug, args.title)
        else:
            result = build(args.username, args.output_dir, args.plan, args.mode, args.folds, args.slug, args.title)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
