"""Specialist notebooks: a pinned StyleDistance input, and an offline submission tied to both recipes."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import nbformat
import pytest

SPEC = importlib.util.spec_from_file_location(
    "ntl_prepare_specialist_submission",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_specialist_submission.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def _recipes(tmp_path, current_weight=0.25):
    current = {"kind": "cpu-qwen-blend-recipe", "weight": current_weight}
    specialist = {"kind": "claude-glm-specialist-recipe", "weight": 0.25,
                  "current_recipe_sha256": hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()}
    (tmp_path / "current.json").write_text(json.dumps(current))
    (tmp_path / "specialist.json").write_text(json.dumps(specialist))
    return tmp_path / "current.json", tmp_path / "specialist.json"


def _code(bundle: Path) -> list[str]:
    notebook = nbformat.read(bundle / "main.ipynb", as_version=4)
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for index, source in enumerate(cells):
        ast.parse(source, filename=f"cell{index}")
    return cells


def test_styledistance_input_downloads_the_pinned_revision_with_internet(tmp_path):
    helper.build_styledistance_input("example_user", tmp_path / "input")
    metadata = json.loads((tmp_path / "input" / "kernel-metadata.json").read_text())
    assert metadata["enable_internet"] is True and metadata["enable_gpu"] is False and metadata["is_private"]
    source = "\n".join(_code(tmp_path / "input"))
    assert helper.STYLEDISTANCE_REPO in source and helper.STYLEDISTANCE_REVISION in source
    assert '"/kaggle/working/styledistance"' in source


def test_submission_is_offline_pins_inputs_and_embeds_both_recipes(tmp_path):
    current, specialist = _recipes(tmp_path)
    kernel = "example_user/name-that-llm-styledistance-model-input/1"
    helper.build_submission("example_user", tmp_path / "bundle", current, specialist, kernel)
    metadata = json.loads((tmp_path / "bundle" / "kernel-metadata.json").read_text())
    assert metadata["enable_internet"] is False and metadata["enable_gpu"] is True and metadata["is_private"]
    assert metadata["kernel_sources"] == [helper.MODEL_KERNEL, helper.DEPENDENCY_KERNEL, kernel]
    manifest = json.loads((tmp_path / "bundle" / "bundle-manifest.json").read_text())
    assert manifest["submitted"] is False
    assert manifest["specialist_recipe_sha256"] == hashlib.sha256(specialist.read_bytes()).hexdigest()
    cells = _code(tmp_path / "bundle")
    source = "\n".join(cells)
    assert "ntl/claude_glm.py" in cells[0]
    assert 'RUNTIME_DIR = Path("/tmp/ntl_env")' in source
    order = [source.index(marker) for marker in ('"ntl.ensemble", ["fit"', '"ntl.claude_glm", ["fit"',
                                                 '"ntl.claude_glm", ["predict"')]
    assert order == sorted(order)
    assert hashlib.sha256(specialist.read_bytes()).hexdigest() in source


def test_submission_refuses_unpinned_inputs_and_mismatched_recipes(tmp_path):
    current, specialist = _recipes(tmp_path)
    with pytest.raises(ValueError, match="owner/slug/version"):
        helper.build_submission("example_user", tmp_path / "a", current, specialist,
                                "example_user/name-that-llm-styledistance-model-input")
    (tmp_path / "other").mkdir()
    other, _ = _recipes(tmp_path / "other", 0.5)
    with pytest.raises(ValueError, match="weight 0.25"):
        helper.build_submission("example_user", tmp_path / "b", other, specialist,
                                "example_user/name-that-llm-styledistance-model-input/1")
    current_text = json.loads(current.read_text())
    current.write_text(json.dumps(dict(current_text, extra=1)))
    with pytest.raises(ValueError, match="different current system"):
        helper.build_submission("example_user", tmp_path / "c", current, specialist,
                                "example_user/name-that-llm-styledistance-model-input/1")
