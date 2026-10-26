"""The frozen-encoder notebook preserves the evaluation protocol and temporary storage."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

SPEC = importlib.util.spec_from_file_location(
    "ntl_prepare_frozen_gemini_experiment",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_frozen_gemini_experiment.py",
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def test_frozen_bundle_pins_models_inputs_and_nested_evaluation(tmp_path):
    helper.build_bundle("example_user", tmp_path)
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["id"] == "example_user/name-that-llm-frozen-gemini-embeddings"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["competition_sources"] == ["name-that-llm"]
    assert metadata["kernel_sources"] == [helper.BASELINE_KERNEL, helper.RUNTIME_KERNEL]
    manifest = json.loads((tmp_path / "bundle-manifest.json").read_text())
    assert manifest["submitted"] is False
    protocol = manifest["protocol"]
    assert protocol["encoder_training"] is False
    assert protocol["instruction_prefix"] is None
    assert protocol["l2_normalize_both_encoders"] is True
    assert protocol["dtype"] == "float32"
    assert protocol["C_grid"] == [0.001, 0.01, 0.1, 1, 10]
    assert protocol["inner_folds"] == 3
    assert protocol["models"][0]["revision"] == "b7df5f0b0480773c097ba3121d83ca32b71015ca"
    assert protocol["models"][1]["revision"] == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    for artifact in manifest["files"]:
        assert hashlib.sha256((tmp_path / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    notebook = nbformat.read(tmp_path / "main.ipynb", as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'RUNTIME_DIR = Path("/tmp/ntl_frozen_gemini_env")' in source
    assert "Path('/tmp/ntl_frozen_models')" in source
    assert '"/tmp/ntl_prepare_frozen_models.py"' in source
    assert '"--require-hashes"' in source
    assert "'-m', 'ntl.encoders'" in source
    assert "'-m', 'ntl.frozen_gemini'" in source
    assert '"--test"' not in source
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source, filename=f"frozen-cell-{index}")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and any(isinstance(t, ast.Name) and t.id in {
                    "probe_code", "preflight_code", "prepare_code"
                } for t in node.targets)
            ):
                ast.parse(node.value.value, filename=f"frozen-subprocess-{index}")


def test_extraction_and_evaluation_run_both_models_with_runtime_isolation(tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "ntl.frozen_gemini" in command:
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "metrics.json").write_text(json.dumps({"synthetic_fixture": True}))
        return SimpleNamespace(returncode=0)

    namespace = {
        "Path": Path, "json": json, "WORKING": tmp_path,
        "MODELS": helper.MODELS,
        "MODEL_PATHS": {item["key"]: "/tmp/model/" + item["key"] for item in helper.MODELS},
        "RUNTIME_PYTHON": "/python", "RUNTIME_ARGS": ["-S"],
        "RUNTIME_ENV": {"HF_HUB_OFFLINE": "1"},
        "TRAIN_PATH": "/official/train.csv", "BASELINE_OOF": "/baseline/oof.parquet",
        "FOLD_MANIFEST": "/baseline/folds.csv", "subprocess": SimpleNamespace(run=fake_run),
    }
    exec(helper.EXPERIMENT, namespace)
    assert len(calls) == 4
    for command, kwargs in calls:
        assert command[:2] == ["/python", "-S"]
        assert command[command.index("--baseline-oof") + 1] == "/baseline/oof.parquet"
        assert command[command.index("--fold-manifest") + 1] == "/baseline/folds.csv"
        assert "--test" not in command
        assert kwargs["check"] is True
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert [calls[index][0][calls[index][0].index("--model-key") + 1] for index in (0, 2)] == [
        "styledistance", "qwen3-embedding-0.6b"
    ]
    comparison = json.loads((tmp_path / "frozen-gemini" / "comparison.json").read_text())
    assert set(comparison) == {item["key"] for item in helper.MODELS}


@pytest.mark.parametrize("reference", ["owner/slug", "owner/slug/latest", "../slug/1"])
def test_reused_runtime_requires_saved_version(tmp_path, reference):
    with pytest.raises(ValueError, match="explicit owner/slug/version"):
        helper.build_bundle("example_user", tmp_path, runtime_kernel=reference)


def test_scratch_runtime_change_cannot_silently_save_binaries_in_output():
    with pytest.raises(ValueError, match="template changed"):
        helper.scratch_runtime("unexpected future runtime template")


@pytest.mark.parametrize("modern_layout", [True, False])
def test_known_input_roots_are_found_without_entering_environment_tree(tmp_path, monkeypatch, modern_layout):
    inputs = tmp_path / 'inputs'
    roots = inputs / 'notebooks' / 'example_user' if modern_layout else inputs
    runtime = roots / 'runtime-notebook'
    baseline = roots / 'baseline-notebook' / 'cpu-cv'
    baseline.mkdir(parents=True)
    for filename in ['oof.parquet', 'folds.csv', 'folds.json']:
        (baseline / filename).write_text('discovery fixture only')
    (runtime / 'gemini_env' / 'lib' / 'python3.13' / 'site-packages' / 'do-not-enter').mkdir(parents=True)
    official = inputs / 'competitions' / 'name-that-llm'
    official.mkdir(parents=True)
    (official / 'train.csv').write_text('id,text,model,prompt_id\n')

    def disallowed_walk(*args, **kwargs):
        raise AssertionError('Input discovery must not enumerate attached environment contents')

    monkeypatch.setattr(Path, 'rglob', disallowed_walk)
    monkeypatch.setattr(Path, 'iterdir', disallowed_walk)
    namespace = {
        'Path': Path,
        'BASELINE_KERNEL_REFERENCE': 'example_user/baseline-notebook/1',
        'RUNTIME_KERNEL_REFERENCE': 'example_user/runtime-notebook/1',
    }
    exec(helper.LOCATORS.replace('/kaggle/input', str(inputs)), namespace)
    assert namespace['runtime_candidates']() == [runtime / 'gemini_env']
    exec(helper.INPUT_LOCATORS.replace('/kaggle/input', str(inputs)), namespace)
    assert namespace['BASELINE_OOF'] == str(baseline / 'oof.parquet')
    assert namespace['TRAIN_PATH'] == str(official / 'train.csv')


def test_saved_runtime_probe_has_progress_and_a_finite_timeout():
    shared = helper.runpy.run_path(str(helper.ROOT / 'scripts' / 'prepare_gemini_experiment.py'))
    runtime = helper.scratch_runtime(shared['RUNTIME'])
    tree = ast.parse(runtime)
    assert not any(isinstance(node, ast.Attribute) and node.attr == 'rglob' for node in ast.walk(tree))
    probe = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'probe_runtime')
    timed_calls = [
        node for node in ast.walk(probe) if isinstance(node, ast.Call)
        and any(keyword.arg == 'timeout' and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == 180 for keyword in node.keywords)
    ]
    assert len(timed_calls) == 1
    assert 'Runtime probe: importing torch' in runtime
    assert 'Runtime probe: importing Transformers' in runtime
    assert 'Runtime probe: checking CUDA' in runtime
    assert '*arguments, "-u", "-c", probe_code' in runtime
