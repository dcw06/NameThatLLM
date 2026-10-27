"""The Qwen validation notebook pins the recipe, folds and temporary storage; it never submits."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import nbformat

SPEC = importlib.util.spec_from_file_location(
    "ntl_prepare_qwen_validation",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_qwen_validation.py",
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def test_bundle_pins_recipe_inputs_and_fold_local_heads(tmp_path):
    helper.build_bundle("example_user", tmp_path)
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["id"] == "example_user/name-that-llm-qwen-frozen-validation"
    assert metadata["is_private"] is True and metadata["enable_gpu"] is True
    assert metadata["competition_sources"] == ["name-that-llm"]
    assert metadata["kernel_sources"] == [helper.BASELINE_KERNEL]
    manifest = json.loads((tmp_path / "bundle-manifest.json").read_text())
    assert manifest["submitted"] is False
    protocol = manifest["protocol"]
    assert protocol["model_revision"] == "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
    assert (protocol["retained_layers"], protocol["pool_layers"], protocol["quantization"]) == (24, [7, 22], "nf4")
    assert protocol["competition_submission"] is False
    for artifact in manifest["files"]:
        assert hashlib.sha256((tmp_path / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    notebook = nbformat.read(tmp_path / "main.ipynb", as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'RUNTIME_DIR = Path("/tmp/ntl_qwen_env")' in source
    assert 'MODEL_PATH = "/tmp/ntl_qwen_model/qwen3-8b-base"' in source
    assert "RUNTIME_REUSE_ENABLED = False" in source
    assert "'-m', 'ntl.ensemble', 'extract'" in source and "'qwen-oof'" in source
    assert "load_fold_manifest" in source and '"--require-hashes"' in source
    assert '"--test"' not in source and "'predict'" not in source
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"qwen-validation-cell-{index}")
