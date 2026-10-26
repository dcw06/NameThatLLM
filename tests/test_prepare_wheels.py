"""Offline dependency bundles must retain exact content and matching Python/platform."""

import importlib.util
import json
import platform
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ntl_prepare_wheels", Path(__file__).resolve().parents[1] / "scripts" / "prepare_wheels.py"
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def bundle(directory):
    (directory / "requirements.lock").write_text("synthetic test fixture\n")
    (directory / "test.whl").write_bytes(b"synthetic test fixture, not a real wheel")
    manifest = {
        "schema_version": 1, "profile": "cpu",
        "python": list(sys.version_info[:2]),
        "system": platform.system(), "machine": platform.machine(),
        "files": [
            {"path": path.name, "size": path.stat().st_size, "sha256": helper.sha256(path)}
            for path in sorted(directory.iterdir())
        ],
    }
    (directory / "wheel_manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_verify_detects_same_size_corruption_without_network(tmp_path):
    bundle(tmp_path)
    helper.verify(tmp_path)
    path = tmp_path / "test.whl"
    contents = path.read_bytes()
    path.write_bytes(b"X" + contents[1:])
    with pytest.raises(ValueError, match="changed"):
        helper.verify(tmp_path)


def test_verify_rejects_wrong_python_and_unexpected_files(tmp_path):
    manifest = bundle(tmp_path)
    manifest["python"] = [0, 0]
    (tmp_path / "wheel_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Python version"):
        helper.verify(tmp_path)
    manifest["python"] = list(sys.version_info[:2])
    (tmp_path / "wheel_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "unexpected.whl").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="inventory"):
        helper.verify(tmp_path)


def test_verify_rejects_manifest_path_escape(tmp_path):
    manifest = bundle(tmp_path)
    manifest["files"][0]["path"] = "../escape"
    (tmp_path / "wheel_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path"):
        helper.verify(tmp_path)
