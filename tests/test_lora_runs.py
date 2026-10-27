"""LoRA notebooks are offline, pin their inputs and fold plan, and spread folds over the GPUs."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import nbformat
import pytest

SPEC = importlib.util.spec_from_file_location(
    "ntl_prepare_lora_runs", Path(__file__).resolve().parents[1] / "scripts" / "prepare_lora_runs.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def test_fold_notebook_is_offline_and_pins_plan_and_inputs(tmp_path):
    plan = tmp_path / "plan.csv"
    plan.write_text("id,prompt_id,outer_fold,inner_0\nx,p,0,-1\n")
    helper.build("example_user", tmp_path / "bundle", plan, "folds", [0, 1], "lora-folds", "LoRA folds")
    metadata = json.loads((tmp_path / "bundle" / "kernel-metadata.json").read_text())
    assert metadata["enable_internet"] is False and metadata["enable_gpu"] is True and metadata["is_private"]
    assert metadata["kernel_sources"] == [helper.MODEL_KERNEL, helper.DEPENDENCY_KERNEL, helper.FEATURE_KERNEL]
    manifest = json.loads((tmp_path / "bundle" / "bundle-manifest.json").read_text())
    assert manifest["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert manifest["submitted"] is False and manifest["folds"] == [0, 1]
    notebook = nbformat.read(tmp_path / "bundle" / "main.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"cell{index}")
    assert "MODE = 'folds'" in source and "FOLDS = [0, 1]" in source
    assert 'RUNTIME_DIR = Path("/tmp/ntl_env")' in source and "CUDA_VISIBLE_DEVICES" in source
    with pytest.raises(ValueError):
        helper.build("example_user", tmp_path / "other", plan, "folds", [5], "lora-folds", "LoRA folds")


def test_final_and_submission_notebooks_chain_selection_fit_and_offline_scoring(tmp_path):
    plan = tmp_path / "final_plan.csv"
    plan.write_text("id,prompt_id,inner\nx,p,0\n")
    helper.build("example_user", tmp_path / "final", plan, "final", [], "lora-final", "LoRA final")
    source = "\n".join(cell.source for cell in nbformat.read(tmp_path / "final" / "main.ipynb", as_version=4).cells
                       if cell.cell_type == "code")
    assert '"ntl.lora_final", "inner"' in source and '["choose", *common]' in source and '"fit", *common' in source
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"weight": 0.25}))
    helper.build_submission("example_user", tmp_path / "submission", recipe, "example_user/lora-final/3",
                            "lora-submission", "LoRA submission")
    metadata = json.loads((tmp_path / "submission" / "kernel-metadata.json").read_text())
    assert metadata["enable_internet"] is False and metadata["kernel_sources"][-1] == "example_user/lora-final/3"
    notebook = nbformat.read(tmp_path / "submission" / "main.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"cell{index}")
    assert '"ntl.ensemble", ["fit"' in source and '"ntl.lora_final", ["predict"' in source
    assert "if len(test_rows) < 1000:" in source
    recipe.write_text(json.dumps({"weight": 1.0}))
    with pytest.raises(ValueError, match="0.25"):
        helper.build_submission("example_user", tmp_path / "other", recipe, "example_user/lora-final/3", "s", "t")
