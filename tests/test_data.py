import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ntl.constants import LABELS
from ntl.data import (
    data_quality_report,
    fingerprint_frame,
    read_test,
    validate_submission,
    validate_test,
    validate_train,
)
from ntl.smoke import synthetic_frames


class DataValidationTests(unittest.TestCase):
    def setUp(self):
        self.train, self.test = synthetic_frames()

    def test_text_is_preserved_exactly_and_fingerprint_is_order_invariant(self):
        checked = validate_train(self.train)
        self.assertEqual(checked["text"].tolist(), self.train["text"].tolist())
        self.assertEqual(fingerprint_frame(self.train), fingerprint_frame(self.train.sample(frac=1, random_state=4)))
        changed = self.train.copy()
        changed.loc[0, "text"] += "!"
        self.assertNotEqual(fingerprint_frame(self.train), fingerprint_frame(changed))

    def test_null_duplicate_unknown_and_nonstring_rejected(self):
        for column, value in (("id", None), ("model", "not-a-model"), ("prompt_id", ""), ("text", 42)):
            frame = self.train.copy()
            frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_train(frame)
        duplicate = pd.concat([self.train, self.train.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_train(duplicate)

    def test_training_only_columns_are_rejected_in_test(self):
        for column in ("model", "prompt_id"):
            frame = self.test.assign(**{column: "accidental-leak"})
            with self.assertRaises(ValueError):
                validate_test(frame)

    def test_csv_reader_preserves_leading_zero_ids_and_literal_na_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            path.write_text('id,text\n00001,NA\n00002,"Case—and ‘quotes’"\n')
            frame = read_test(path)
            self.assertEqual(frame["id"].tolist(), ["00001", "00002"])
            self.assertEqual(frame["text"].tolist(), ["NA", "Case—and ‘quotes’"])

    def test_empty_test_rejected_and_cross_prompt_cross_label_duplicates_reported(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_test(self.test.iloc[:0])
        modified = self.train.copy()
        # A duplicate spans two prompts and two author labels. Keep the rows,
        # but expose both ambiguity and the validation risk.
        modified.loc[19, "text"] = modified.loc[0, "text"]
        checked = validate_train(modified)
        report = data_quality_report(checked)
        self.assertEqual(len(checked), len(self.train))
        self.assertEqual(report["exact_duplicate_text_groups"], 1)
        self.assertEqual(report["exact_duplicate_text_rows"], 2)
        self.assertEqual(report["cross_label_identical_text_groups"], 1)
        self.assertEqual(report["cross_prompt_identical_text_groups"], 1)
        self.assertEqual(len(report["warnings"]), 2)

    def test_submission_coverage_and_labels_are_validated_without_assuming_order(self):
        submission = pd.DataFrame({"id": self.test["id"], "model": list(reversed(LABELS))[:len(self.test)]})
        validate_submission(submission.iloc[::-1], self.test)
        broken = submission.copy()
        broken.loc[0, "model"] = "gpt-unknown"
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_submission(broken, self.test)
        for broken in (submission.iloc[:-1], pd.concat([submission, submission.iloc[:1]]),
                       submission.assign(extra=1), submission.assign(model=None)):
            with self.assertRaises(ValueError):
                validate_submission(broken, self.test)


if __name__ == "__main__":
    unittest.main()
