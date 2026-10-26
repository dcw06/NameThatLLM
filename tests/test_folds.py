import tempfile
import unittest

import pandas as pd

from ntl.folds import (
    build_fold_manifest,
    load_fold_manifest,
    save_fold_manifest,
    validate_fold_manifest,
)
from ntl.smoke import synthetic_frames


class FoldTests(unittest.TestCase):
    def setUp(self):
        self.train, _ = synthetic_frames()

    def test_deterministic_complete_prompt_isolation_and_roundtrip(self):
        manifest, metadata = build_fold_manifest(self.train)
        shuffled, _ = build_fold_manifest(self.train.sample(frac=1, random_state=9))
        pd.testing.assert_frame_equal(manifest, shuffled)
        self.assertEqual(len(manifest), len(self.train))
        self.assertTrue(manifest["id"].is_unique)
        self.assertTrue(manifest.groupby("prompt_id")["fold"].nunique().eq(1).all())
        for fold in range(5):
            fit = set(manifest.loc[manifest["fold"] != fold, "prompt_id"])
            valid = set(manifest.loc[manifest["fold"] == fold, "prompt_id"])
            self.assertFalse(fit & valid)
        with tempfile.TemporaryDirectory() as directory:
            path = save_fold_manifest(manifest, metadata, directory)
            loaded, loaded_metadata = load_fold_manifest(self.train, path)
            pd.testing.assert_frame_equal(manifest, loaded)
            self.assertEqual(metadata, loaded_metadata)

    def test_changed_training_data_and_leaking_manifest_rejected(self):
        manifest, metadata = build_fold_manifest(self.train)
        changed = self.train.copy()
        changed.loc[0, "text"] += "changed"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_fold_manifest(changed, manifest, metadata)
        broken = manifest.copy()
        broken.loc[0, "fold"] = (int(broken.loc[0, "fold"]) + 1) % 5
        with self.assertRaisesRegex(ValueError, "Prompt leakage"):
            validate_fold_manifest(self.train, broken, metadata)


if __name__ == "__main__":
    unittest.main()
