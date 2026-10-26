"""The Gemini diagnostic keeps its source, environment and baseline version explicit."""

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

SPEC = importlib.util.spec_from_file_location(
    "ntl_prepare_gemini_experiment",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_gemini_experiment.py",
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def test_diagnostic_bundle_is_private_gpu_and_pins_upstream_inputs(tmp_path):
    result = helper.build_bundle("example_user", tmp_path)
    assert result["model_revision"] == "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is True
    assert metadata["competition_sources"] == ["name-that-llm"]
    assert metadata["kernel_sources"] == ["daichongwei06/name-that-llm-cpu-first-submission/1"]
    manifest = json.loads((tmp_path / "bundle-manifest.json").read_text())
    assert manifest["model_id"] == "microsoft/deberta-v3-base"
    assert manifest["submitted"] is False
    for artifact in manifest["files"]:
        assert hashlib.sha256((tmp_path / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    notebook = nbformat.read(tmp_path / "main.ipynb", as_version=4)
    nbformat.validate(notebook)
    sources = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'system_site_packages=False' in sources
    assert 'venv.EnvBuilder(with_pip=False,' in sources
    assert '[sys.executable, "-m", "pip", "--python", RUNTIME_PYTHON,' in sources
    assert 'venv.EnvBuilder(with_pip=True,' not in sources
    assert '"--require-hashes"' in sources
    assert '"--baseline-oof", BASELINE_OOF' in sources
    assert '"--fold-manifest", FOLD_MANIFEST' in sources
    assert '"--epochs", "3"' in sources
    assert "ntl.gemini" in sources
    assert "competition_submit" not in sources
    assert '"--test"' not in sources

    # Both notebook cell code and code executed in subprocesses must compile.
    nested_scripts = {"probe_code", "download_code", "preflight_code"}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source, filename=f"gemini-cell-{index}")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and any(isinstance(t, ast.Name) and t.id in nested_scripts for t in node.targets)
            ):
                ast.parse(node.value.value, filename=f"gemini-nested-{index}")


def test_original_offline_exports_use_existing_pip_for_pipless_venv(tmp_path):
    exporter = helper.runpy.run_path(str(helper.ROOT / "scripts" / "build_kaggle_notebooks.py"))
    sources = exporter["package_sources"]()
    for cells in [exporter["make_cpu"](sources), exporter["make_gpu"](sources)]:
        source = "\n".join(item["source"] for item in cells if item["cell_type"] == "code")
        assert 'venv.EnvBuilder(with_pip=False,' in source
        assert '[sys.executable, "-m", "pip", "--python", RUNTIME_PYTHON,' in source
        assert '"--no-index"' in source
        assert '"--require-hashes"' in source
        assert 'venv.EnvBuilder(with_pip=True,' not in source


def test_saved_runtime_uses_current_python_without_global_site_packages(tmp_path, monkeypatch):
    import venv

    input_root = tmp_path / "inputs"
    saved = input_root / "prior-output"
    environment = saved / "gemini_env"
    site_packages = environment / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    (site_packages / "torch").mkdir(parents=True)
    (site_packages / "nvidia").mkdir()
    lock = "synthetic lock fixture"
    (saved / "gemini-requirements.lock").write_text(lock)
    (saved / "environment.json").write_text(json.dumps({"python": sys.version}))
    source_root = tmp_path / "source"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    def unexpected_builder(**kwargs):
        raise AssertionError("A valid saved runtime must not create or install another environment")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(venv, "EnvBuilder", unexpected_builder)
    code = helper.RUNTIME.replace("/kaggle/working", str(tmp_path / "working"))
    code = code.replace("/kaggle/input", str(input_root))
    namespace = {
        "Path": Path, "os": os, "sys": sys, "json": json, "hashlib": hashlib,
        "SOURCE_ROOT": source_root, "REQUIREMENTS_LOCK": lock,
        "REQUIREMENTS_SHA256": hashlib.sha256(lock.encode()).hexdigest(),
        "EXPECTED_RUNTIME_PACKAGES": [], "RUNTIME_REUSE_ENABLED": True,
    }
    exec(code, namespace)
    assert namespace["RUNTIME_PYTHON"] == sys.executable
    assert namespace["RUNTIME_ARGS"] == ["-S"]
    assert calls[0][0][:3] == [sys.executable, "-S", "-c"]
    assert calls[0][1]["env"]["PYTHONPATH"] == os.pathsep.join([str(source_root), str(site_packages)])


def test_runtime_reference_requires_new_slug_and_fixed_version(tmp_path):
    with pytest.raises(ValueError, match="explicit owner/slug/version"):
        helper.build_bundle("example_user", tmp_path, runtime_kernel="owner/notebook")
    with pytest.raises(ValueError, match="new notebook slug"):
        helper.build_bundle(
            "example_user", tmp_path,
            runtime_kernel=f"example_user/{helper.DEFAULT_SLUG}/2",
        )
    helper.build_bundle(
        "example_user", tmp_path, slug="gemini-run",
        runtime_kernel=f"example_user/{helper.DEFAULT_SLUG}/2",
    )
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["kernel_sources"][-1] == f"example_user/{helper.DEFAULT_SLUG}/2"


@pytest.mark.parametrize("baseline", ["owner/notebook", "owner/notebook/latest", "../other/1"])
def test_baseline_reference_requires_an_immutable_version(tmp_path, baseline):
    with pytest.raises(ValueError, match="explicit owner/slug/version"):
        helper.build_bundle("example_user", tmp_path, baseline_kernel=baseline)
    assert not list(tmp_path.iterdir())


def test_input_discovery_rejects_ambiguous_and_missing_oof(tmp_path):
    official = tmp_path / "official"
    official.mkdir()
    (official / "train.csv").write_text("id,text,model,prompt_id\n")
    source = helper.INPUTS.replace(
        "/kaggle/input/competitions/name-that-llm/train.csv", str(official / "train.csv")
    ).replace("/kaggle/input/name-that-llm/train.csv", str(tmp_path / "absent.csv"))
    source = source.replace("/kaggle/input", str(tmp_path / "inputs"))
    namespace = {"Path": Path}
    with pytest.raises(RuntimeError, match="pinned CPU"):
        exec(source, namespace)
    for folder in ["first", "second"]:
        baseline = tmp_path / "inputs" / folder / "cpu-cv"
        baseline.mkdir(parents=True)
        for filename in ["oof.parquet", "folds.csv", "folds.json"]:
            (baseline / filename).write_text("discovery fixture only")
    with pytest.raises(RuntimeError, match="pinned CPU"):
        exec(source, {"Path": Path})
