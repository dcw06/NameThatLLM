"""Gemini cohort, fold and paired-report contracts without downloading a model."""

import json

import numpy as np
import pandas as pd
import pytest

from ntl import gemini
from ntl.folds import build_fold_manifest, save_fold_manifest
from ntl.smoke import synthetic_frames


@pytest.fixture
def cohort(tmp_path):
    train, _ = synthetic_frames()
    manifest, metadata = build_fold_manifest(train)
    path = save_fold_manifest(manifest, metadata, tmp_path / "folds")
    baseline = train[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    baseline["predicted_model"] = baseline["true_model"]
    baseline["fold"] = baseline["id"].map(manifest.set_index("id")["fold"])
    indices = baseline.index[baseline["true_model"].isin(gemini.GEMINI_LABELS)]
    baseline.loc[indices[0], "predicted_model"] = "gpt-5.5"
    baseline.loc[indices[1], "predicted_model"] = gemini.GEMINI_LABELS[0]
    return train, baseline, path


def test_cohort_is_id_aligned_and_keeps_cpu_family_false_negatives_for_training(cohort):
    train, baseline, path = cohort
    frame, protocol = gemini.prepare_gemini_evaluation(train, baseline.sample(frac=1, random_state=4), path)
    assert len(frame) == 24
    assert frame["selected"].sum() == 23
    assert (~frame["selected"]).sum() == 1
    assert protocol["counts"]["baseline_selected_correct"] == 22
    assert set(frame["model"]) == set(gemini.GEMINI_LABELS)
    assert frame.groupby("prompt_id")["fold"].nunique().max() == 1


@pytest.mark.parametrize("change,match", [
    ("truth", "true model"), ("prompt", "prompt ids"), ("fold", "folds differ"),
    ("missing", "every official"), ("duplicate", "duplicate"),
])
def test_wrong_baseline_artifact_is_rejected(cohort, change, match):
    train, baseline, path = cohort
    broken = baseline.copy()
    if change == "truth":
        broken.loc[0, "true_model"] = "gpt-6-astra"
    elif change == "prompt":
        broken.loc[0, "prompt_id"] = "wrong-prompt"
    elif change == "fold":
        broken.loc[0, "fold"] = (int(broken.loc[0, "fold"]) + 1) % 5
    elif change == "missing":
        broken = broken.iloc[:-1]
    else:
        broken = pd.concat([broken, broken.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match=match):
        gemini.prepare_gemini_evaluation(train, broken, path)


def test_fixed_official_cohort_counts_fail_closed(cohort):
    train, baseline, path = cohort
    with pytest.raises(ValueError, match="cohort mismatch"):
        gemini.prepare_gemini_evaluation(train, baseline, path, gemini.OFFICIAL_BASELINE_COUNTS)


def test_run_uses_all_training_gemini_and_saves_complete_paired_reports(cohort, tmp_path, monkeypatch):
    train, baseline, path = cohort
    monkeypatch.setattr(gemini, "validate_model_snapshot", lambda directory: {"revision": gemini.MODEL_REVISION})
    calls = []

    def fake_fold(fitting, validation, model_path, config, fold):
        assert not set(fitting["prompt_id"]) & set(validation["prompt_id"])
        calls.append({"fold": fold, "ids": set(fitting["id"]), "selected_only": fitting["selected"].all()})
        values = np.eye(4)[validation["target"].to_numpy()]
        return values, {"fold": fold, "train_rows": len(fitting), "validation_rows": len(validation)}

    monkeypatch.setattr(gemini, "_fit_predict_fold", fake_fold)
    output = tmp_path / "experiment"
    metrics = gemini.run_experiment(train, baseline, path, "unused", output,
                                    expected_counts={"gemini_rows": 24, "selected_rows": 23})
    assert len(calls) == 5
    assert sum(not call["selected_only"] for call in calls) == 4
    assert metrics["correctly_routed_gemini"]["baseline_correct"] == 22
    assert metrics["correctly_routed_gemini"]["specialist_correct"] == 23
    assert metrics["correctly_routed_gemini"]["paired_transitions"] == {
        "both_correct": 22, "baseline_wrong_specialist_correct": 1,
        "baseline_correct_specialist_wrong": 0, "both_wrong": 0,
    }
    oof = pd.read_parquet(output / "oof.parquet")
    assert len(oof) == 24 and oof["id"].is_unique
    assert len(pd.read_parquet(output / "selected_oof.parquet")) == 23
    matrix = pd.read_csv(output / "confusion_matrix.csv", index_col=0)
    assert matrix.shape == (4, 4) and np.trace(matrix.to_numpy()) == 24
    saved = json.loads((output / "metadata.json").read_text())
    assert saved["model_revision"] == gemini.MODEL_REVISION
    assert saved["configuration"]["epochs"] == 3
    assert not list(output.glob("*.bin")) and not list(output.glob("*.safetensors"))


def test_model_snapshot_rejects_missing_weights_and_wrong_revision(tmp_path):
    config = {"model_type": "deberta-v2", "hidden_size": 768, "num_hidden_layers": 12}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "spm.model").write_bytes(b"fixture")
    (tmp_path / "pytorch_model.bin").write_bytes(b"fixture")
    manifest = {"schema_version": 1, "repo_id": gemini.MODEL_ID, "revision": gemini.MODEL_REVISION,
                "includes_weights": True,
                "files": [{"path": p.name, "size": p.stat().st_size} for p in tmp_path.iterdir()]}
    manifest_path = tmp_path / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    gemini.validate_model_snapshot(tmp_path)
    (tmp_path / "pytorch_model.bin").write_bytes(b"truncated fixture")
    with pytest.raises(ValueError, match="changed"):
        gemini.validate_model_snapshot(tmp_path)
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned"):
        gemini.validate_model_snapshot(tmp_path)


def test_tiny_real_deberta_performs_training_updates_and_predicts(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    architecture = transformers.DebertaV2Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=24, max_position_embeddings=32, num_labels=4,
        pooler_hidden_size=16, relative_attention=False,
    )
    tiny = transformers.DebertaV2ForSequenceClassification(architecture)
    tiny.save_pretrained(tmp_path / "tiny", safe_serialization=False)

    class TinyTokenizer:
        def __call__(self, texts, **kwargs):
            ids = [[1, 2, 3 + index % 4, 2] for index, _ in enumerate(texts)]
            return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

        def pad(self, rows, **kwargs):
            return {key: torch.tensor([row[key] for row in rows], dtype=torch.long) for key in rows[0]}

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    fitting = pd.DataFrame({"text": ["alpha", "beta", "gamma", "delta", "tail"],
                            "target": [0, 1, 2, 3, 0],
                            "prompt_id": ["train"] * 5, "selected": [True, False, True, True, True]})
    validation = pd.DataFrame({"text": ["held out alpha", "held out beta"], "target": [0, 1],
                              "prompt_id": ["validation"] * 2, "selected": [True, True]})
    config = gemini.GeminiConfig(epochs=1, batch_size=2, gradient_accumulation=2,
                                max_length=16, device="cpu")
    probabilities, log = gemini._fit_predict_fold(fitting, validation, tmp_path / "tiny", config, 0)
    assert probabilities.shape == (2, 4)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    # Three microbatches: one full accumulation group and one single-row tail.
    assert log["optimizer_updates"] == 2
    assert log["skipped_amp_updates"] == 0
    assert len(log["epochs"]) == 1
    assert np.isfinite(log["epochs"][0]["mean_training_loss"])


def test_cuda_allocator_is_initialized_before_resetting_peak_stats():
    calls = []

    class Transient:
        def __del__(self):
            calls.append("release_transient")

    class FakeCuda:
        def init(self):
            calls.append("init")

        def set_device(self, device):
            assert device == "cuda:0"
            calls.append("set_device")

        def empty_cache(self):
            calls.append("empty_cache")

        def reset_peak_memory_stats(self, device):
            assert device == "cuda:0"
            assert calls == ["init", "set_device", "allocate", "release_transient", "empty_cache"]
            calls.append("reset_peak")

    class FakeTorch:
        cuda = FakeCuda()

        def empty(self, size, *, device):
            assert size == 1 and device == "cuda:0"
            calls.append("allocate")
            return Transient()

    gemini._initialize_cuda_memory_stats(FakeTorch(), "cuda:0")
    assert calls[-1] == "reset_peak"
