"""Network-free checks for immutable model snapshot preparation."""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    "ntl_prepare_model", Path(__file__).resolve().parents[1] / "scripts" / "prepare_model.py"
)
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)


class PrepareModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.destination = Path(self.temporary.name) / "snapshot"
        self.calls = []

    def downloader(self, **kwargs):
        self.calls.append(kwargs)
        directory = Path(kwargs["local_dir"])
        spec = prep.MODEL_SPECS[kwargs["repo_id"]]
        config = {key: spec[key] for key in
                  ("model_type", "hidden_size", "num_hidden_layers")}
        if spec.get("nested_text"):
            config = {"model_type": spec["model_type"], "text_config": config}
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "README.md").write_text("fixture model card\n")
        # Hub tracking files are not deployment artifacts.
        cache = directory / ".cache" / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "tracking.metadata").write_text("tracking")
        if "*.safetensors" in kwargs["allow_patterns"]:
            (directory / "model.safetensors").write_bytes(b"fixture weights")
        if "pytorch_model.bin" in kwargs["allow_patterns"]:
            (directory / "pytorch_model.bin").write_bytes(b"fixture bin weights")

    def prepare(self, **kwargs):
        return prep.prepare_snapshot(self.destination, downloader=self.downloader, **kwargs)

    def test_metadata_is_pinned_anonymous_and_contains_no_weight_pattern(self):
        manifest = self.prepare(metadata_only=True)
        self.assertFalse(manifest["includes_weights"])
        self.assertEqual(manifest["revision"], prep.MODEL_SPECS[prep.DEFAULT_REPO]["revision"])
        self.assertFalse(self.calls[0]["token"])
        self.assertNotIn("*.safetensors", self.calls[0]["allow_patterns"])
        self.assertNotIn("pytorch_model.bin", self.calls[0]["allow_patterns"])
        self.assertEqual({row["path"] for row in manifest["files"]}, {"config.json", "README.md"})
        self.assertTrue(manifest["downloaded_at_utc"].endswith("Z"))
        self.assertEqual(prep.verify_snapshot(self.destination), manifest)

    def test_full_default_and_metadata_upgrade_preserve_manifest_identity(self):
        self.prepare(metadata_only=True)
        manifest = self.prepare()
        self.assertTrue(manifest["includes_weights"])
        self.assertEqual(len(self.calls), 2)
        self.assertIn("*.safetensors", self.calls[-1]["allow_patterns"])
        self.assertEqual(prep.verify_snapshot(self.destination), manifest)
        # A later metadata refresh does not incorrectly claim existing weights vanished.
        self.assertTrue(self.prepare(metadata_only=True)["includes_weights"])

    def test_nonempty_unmanaged_destination_is_refused_before_download(self):
        self.destination.mkdir()
        (self.destination / "notes.txt").write_text("do not overwrite")
        with self.assertRaisesRegex(prep.SnapshotError, "Nonempty destination"):
            self.prepare()
        self.assertEqual(self.calls, [])
        self.assertEqual((self.destination / "notes.txt").read_text(), "do not overwrite")

    def test_existing_other_model_and_unreviewed_revision_are_refused(self):
        self.prepare(metadata_only=True)
        before = (self.destination / prep.MANIFEST_NAME).read_bytes()
        with self.assertRaisesRegex(prep.SnapshotError, "different model"):
            self.prepare(repo_id="Qwen/Qwen3-14B-Base")
        with self.assertRaisesRegex(prep.SnapshotError, "reviewed pin"):
            self.prepare(revision="a" * 40)
        with self.assertRaisesRegex(prep.SnapshotError, "immutable"):
            self.prepare(revision="main")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual((self.destination / prep.MANIFEST_NAME).read_bytes(), before)

    def test_config_mismatch_never_publishes_manifest(self):
        def wrong_config(**kwargs):
            self.downloader(**kwargs)
            path = Path(kwargs["local_dir"]) / "config.json"
            config = json.loads(path.read_text())
            config["num_hidden_layers"] = 32
            path.write_text(json.dumps(config))
        with self.assertRaisesRegex(prep.SnapshotError, "num_hidden_layers"):
            prep.prepare_snapshot(self.destination, downloader=wrong_config)
        self.assertFalse((self.destination / prep.MANIFEST_NAME).exists())

    def test_offline_verification_detects_byte_changes_and_extra_files(self):
        self.prepare()
        weights = self.destination / "model.safetensors"
        original = weights.read_bytes()
        weights.write_bytes(original + b"modified")
        with self.assertRaisesRegex(prep.SnapshotError, "byte sizes"):
            prep.verify_snapshot(self.destination)
        weights.write_bytes(original)
        (self.destination / "unexpected.txt").write_text("extra")
        with self.assertRaisesRegex(prep.SnapshotError, "file count or paths"):
            prep.verify_snapshot(self.destination)

    def test_verify_cli_infers_model_without_importing_or_calling_downloader(self):
        self.prepare(repo_id="Qwen/Qwen3-4B-Base", metadata_only=True)
        with patch.object(prep, "_download_snapshot", side_effect=AssertionError("network")):
            with redirect_stdout(io.StringIO()) as captured:
                result = prep.main(["--output-dir", str(self.destination), "--verify"])
        self.assertEqual(result, 0)
        self.assertTrue(json.loads(captured.getvalue())["verified_offline"])

    def test_missing_shard_is_rejected_before_manifest_commit(self):
        def incomplete(**kwargs):
            self.downloader(**kwargs)
            (Path(kwargs["local_dir"]) / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"tensor": "missing-00001.safetensors"}})
            )
        with self.assertRaisesRegex(prep.SnapshotError, "missing weight shards"):
            prep.prepare_snapshot(self.destination, downloader=incomplete)
        self.assertFalse((self.destination / prep.MANIFEST_NAME).exists())

    def test_traversal_manifest_path_and_symlinks_are_rejected(self):
        self.prepare(metadata_only=True)
        manifest_path = self.destination / prep.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][0]["path"] = "../outside.json"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(prep.SnapshotError, "normalized relative"):
            prep.verify_snapshot(self.destination)
        self.assertEqual(len(self.calls), 1)

    def test_download_error_does_not_echo_authentication_or_commit_manifest(self):
        def broken(**kwargs):
            raise RuntimeError("token=secret-credential")
        with patch.object(prep, "_download_snapshot", broken):
            with redirect_stderr(io.StringIO()) as captured:
                result = prep.main(["--output-dir", str(self.destination)])
        self.assertEqual(result, 1)
        self.assertNotIn("secret-credential", captured.getvalue())
        self.assertFalse((self.destination / prep.MANIFEST_NAME).exists())

    def test_deberta_uses_only_discriminator_bin_and_qwen35_nested_config(self):
        manifest = self.prepare(repo_id="microsoft/deberta-v3-large")
        self.assertTrue(manifest["includes_weights"])
        self.assertIn("pytorch_model.bin", self.calls[0]["allow_patterns"])
        self.assertNotIn("*.bin", self.calls[0]["allow_patterns"])
        self.assertEqual(prep.verify_snapshot(self.destination)["repo_id"],
                         "microsoft/deberta-v3-large")
        qwen35 = self.destination.parent / "qwen35"
        prep.prepare_snapshot(qwen35, repo_id="Qwen/Qwen3.5-9B",
                              metadata_only=True, downloader=self.downloader)
        self.assertEqual(prep.verify_snapshot(qwen35)["repo_id"], "Qwen/Qwen3.5-9B")


if __name__ == "__main__":
    unittest.main()
