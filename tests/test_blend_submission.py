"""Offline blend submission bundles pin inputs, recipe and runtime; nothing is pushed or submitted."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import nbformat
import pytest

SPEC = importlib.util.spec_from_file_location(
    "ntl_prepare_blend_submission",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_blend_submission.py",
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def code(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{path}:cell{index}")
    return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")


def test_input_notebooks_are_private_cpu_and_online(tmp_path):
    results = helper.build_inputs("example_user", tmp_path)
    assert [item["kernel_id"] for item in results] == [
        "example_user/name-that-llm-qwen-model-input", "example_user/name-that-llm-neural-wheels"]
    for folder in ("model", "dependencies"):
        metadata = json.loads((tmp_path / folder / "kernel-metadata.json").read_text())
        assert metadata["is_private"] and metadata["enable_internet"] and not metadata["enable_gpu"]
    assert "/kaggle/working/qwen3-8b-base" in code(tmp_path / "model" / "main.ipynb")
    assert "neural-wheelhouse" in code(tmp_path / "dependencies" / "main.ipynb") or \
        "-wheelhouse" in code(tmp_path / "dependencies" / "main.ipynb")


def test_submission_is_offline_pins_inputs_recipe_and_rehearsal(tmp_path):
    recipe = {"kind": "cpu-qwen-blend-recipe", "weight": 0.25, "schema_version": 1}
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    helper.build_submission("example_user", tmp_path / "bundle", recipe_path,
                            "example_user/name-that-llm-qwen-model-input/1",
                            "example_user/name-that-llm-neural-wheels/2")
    metadata = json.loads((tmp_path / "bundle" / "kernel-metadata.json").read_text())
    assert metadata["enable_internet"] is False and metadata["enable_gpu"] is True and metadata["is_private"]
    assert metadata["kernel_sources"] == ["example_user/name-that-llm-qwen-model-input/1",
                                          "example_user/name-that-llm-neural-wheels/2"]
    manifest = json.loads((tmp_path / "bundle" / "bundle-manifest.json").read_text())
    assert manifest["submitted"] is False and manifest["weight"] == 0.25
    assert manifest["recipe_sha256"] == hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    source = code(tmp_path / "bundle" / "main.ipynb")
    assert 'RUNTIME_DIR = Path("/tmp/ntl_env")' in source and '"--no-index"' in source
    assert '"ntl.ensemble", ["fit"' in source and '"ntl.ensemble", ["predict"' in source
    assert "if len(test_rows) < 1000:" in source and "range(4000)" in source
    assert "Path(SUBMISSION).unlink(missing_ok=True)" in source
    with pytest.raises(ValueError, match="owner/slug/version"):
        helper.build_submission("example_user", tmp_path / "other", recipe_path,
                                "example_user/name-that-llm-qwen-model-input", "x/y/1")
