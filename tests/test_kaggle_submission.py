"""The first submission must remain private, offline, and independent of sample size."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from ntl.smoke import synthetic_frames

SPEC = importlib.util.spec_from_file_location(
    "ntl_prepare_kaggle_submission",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_kaggle_submission.py",
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def test_private_cpu_metadata_and_current_source_are_packaged(tmp_path):
    result = helper.build_bundle("example_user", tmp_path)
    assert result["kernel_id"] == "example_user/name-that-llm-cpu-first-submission"
    assert result["run_grouped_cv"] is True
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["competition_sources"] == ["name-that-llm"]
    assert metadata["dataset_sources"] == []
    notebook = nbformat.read(tmp_path / metadata["code_file"], as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata.kaggle.isInternetEnabled is False
    assert notebook.metadata.kaggle.isGpuEnabled is False
    assert all(not cell.get("outputs") for cell in notebook.cells)
    manifest = json.loads((tmp_path / "bundle-manifest.json").read_text())
    for artifact in manifest["files"]:
        assert hashlib.sha256((tmp_path / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    assert manifest["requires_offline_wheel_bundle"] is False
    assert manifest["submitted"] is False


@pytest.mark.parametrize("username", ["https://kaggle.com/someone", "owner/notebook", "../escape"])
def test_username_cannot_change_notebook_paths(tmp_path, username):
    with pytest.raises(ValueError, match="username"):
        helper.build_bundle(username, tmp_path)
    assert not list(tmp_path.iterdir())


def test_exported_notebook_runs_cv_and_preserves_arbitrary_test_ids(tmp_path, monkeypatch):
    """Run the actual embedded package through fresh subprocesses, without Kaggle."""
    output = tmp_path / "bundle"
    working = tmp_path / "working"
    data = tmp_path / "official-data"
    data.mkdir()
    train, original_test = synthetic_frames()
    train.to_csv(data / "train.csv", index=False)
    original_test.to_csv(data / "test.csv", index=False)
    helper.build_bundle("example_user", output)
    notebook = nbformat.read(output / "main.ipynb", as_version=4)
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {}
    code = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source.replace("/kaggle/working", str(working))
        source = source.replace("/kaggle/input/competitions/name-that-llm", str(data))
        source = source.replace("/kaggle/input/name-that-llm", str(tmp_path / "absent"))
        exec(compile(source, f"first-submission-cell-{index}", "exec"), namespace)
        code.append(source)

    environment = json.loads((working / "first_submission_environment.json").read_text())
    assert environment["fresh_model_fit_in_same_environment"] is True
    assert environment["uses_kaggle_installed_cpu_packages"] is True
    assert environment["source_sha256"] == namespace["SOURCE_SHA256"]
    assert environment["versions"]["scikit-learn"]
    cv_report = json.loads((working / "cpu-cv" / "metrics.json").read_text())
    assert cv_report["n_splits"] == 5
    assert cv_report["n_rows"] == len(train)
    assert cv_report["synthetic_data"] is True

    # Keep this one meaningful CV run, then exercise prediction on sample-sized
    # and larger files without fitting again or making accuracy claims.
    for count in (1, 23):
        test = original_test.iloc[[i % len(original_test) for i in range(count)]].copy()
        test["id"] = [f"synthetic-sized-{i:04d}" for i in range(count)]
        test.to_csv(data / "test.csv", index=False)
        namespace["run_runtime"]("ntl", [
            "predict", "--model", str(working / "cpu-model" / "model.joblib"),
            "--test", str(data / "test.csv"), "--output", str(working / "submission.parquet"),
        ])
        exec(compile(code[-1], "first-submission-output-check", "exec"), namespace)
        submission = pd.read_parquet(working / "submission.parquet")
        assert list(submission.columns) == ["id", "model"]
        assert submission["id"].tolist() == test["id"].tolist()

    # Ensure the notebook's final check rejects output from the wrong test run.
    submission.loc[0, "id"] = "synthetic-wrong-id"
    submission.to_parquet(working / "submission.parquet", index=False)
    with pytest.raises(subprocess.CalledProcessError):
        exec(compile(code[-1], "first-submission-corrupt-output", "exec"), namespace)
