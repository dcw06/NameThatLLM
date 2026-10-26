import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ntl.constants import LABELS
from ntl.data import validate_submission
from ntl.modeling import (
    BaselineConfig,
    aligned_probabilities,
    build_pipeline,
    predict_baseline,
    run_cross_validation,
    train_baseline,
)
from ntl.smoke import synthetic_frames


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = synthetic_frames()
        cls.config = BaselineConfig(char_max_features=1500, word_max_features=500, max_iter=300)

    def test_oof_completeness_probabilities_and_per_class_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_cross_validation(self.train, directory, config=self.config)
            oof = pd.read_parquet(Path(directory) / "oof.parquet")
            self.assertEqual(len(oof), len(self.train))
            self.assertEqual(set(oof["id"]), set(self.train["id"]))
            self.assertTrue(oof["id"].is_unique)
            probabilities = oof[[f"probability::{label}" for label in LABELS]].to_numpy()
            self.assertTrue(np.isfinite(probabilities).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
            class_metrics = pd.read_csv(Path(directory) / "class_metrics.csv")
            self.assertEqual(class_metrics["model"].tolist(), list(LABELS))
            self.assertAlmostEqual(metrics["balanced_accuracy"], class_metrics["recall"].mean())

    def test_vocabulary_is_fit_only_on_fitting_rows(self):
        pipeline = build_pipeline(self.config)
        training = self.train.loc[self.train["prompt_id"] != "synthetic-prompt-000"]
        pipeline.fit(training["text"], training["model"])
        held_out = self.test["text"].copy() + " HeldoutOnlyTokenZzz"
        pipeline.predict(held_out)
        vocabulary = dict(pipeline.named_steps["features"].transformer_list)["word"].vocabulary_
        self.assertNotIn("HeldoutOnlyTokenZzz", vocabulary)

    def test_artifact_roundtrip_arbitrary_test_sizes_and_label_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            model = train_baseline(self.train, Path(directory) / "model", self.config)
            for row_count in (1, 3, 7):
                test = pd.DataFrame({"id": [f"test-{index}" for index in range(row_count)],
                                     "text": [self.test.iloc[index % 3]["text"] for index in range(row_count)]})
                submission = predict_baseline(model, test, Path(directory) / f"submission-{row_count}.parquet")
                validate_submission(submission, test)
                self.assertEqual(len(submission), row_count)
        # Deliberately reverse an estimator's class order. Alignment must recover
        # canonical LABELS instead of assuming predict_proba columns match it.
        class FakeClassifier:
            classes_ = np.asarray(list(reversed(LABELS)))
        class FakePipeline:
            named_steps = {"classifier": FakeClassifier()}
            def predict_proba(self, texts):
                values = np.arange(1, 19, dtype=float)
                return np.tile(values / values.sum(), (len(texts), 1))
        aligned = aligned_probabilities(FakePipeline(), ["text"])
        self.assertGreater(aligned[0, 0], aligned[0, -1])


if __name__ == "__main__":
    unittest.main()
