"""Error inspection: ranks, gaps, swaps, the in-sample boost and example choice on hand-made probabilities."""

import numpy as np
import pandas as pd
import pytest

from ntl import error_inspection
from ntl.constants import LABELS

FABLE, FLASH, GLM, OPUS = (LABELS.index(name) for name in ("claude-fable-5", "glm-5.3-flash", "glm-5.3",
                                                             "claude-opus-5"))


def _case():
    truth = [FABLE, FLASH, FABLE, GLM, FABLE]
    probabilities = np.full((5, len(LABELS)), 0.01)
    probabilities[0, [FLASH, FABLE]] = [0.40, 0.30]  # Fable called GLM Flash, Fable second
    probabilities[1, [FABLE, OPUS, FLASH]] = [0.50, 0.20, 0.10]  # GLM Flash called Fable, true third
    probabilities[2, [FABLE, FLASH]] = [0.45, 0.43]  # correct, narrowly
    probabilities[3, [OPUS, GLM]] = [0.30, 0.29]  # GLM called Opus 5, narrowly
    probabilities[4, [FLASH, FABLE]] = [0.60, 0.10]  # a confident error
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    train = pd.DataFrame({"id": [f"r{i}" for i in range(5)], "prompt_id": ["a", "a", "b", "c", "d"],
                          "model": [LABELS[label] for label in truth],
                          "text": ["one two three", "four five", "six", "seven eight nine ten", "x y"]})
    return train, probabilities, np.array(truth)


def test_ranks_gaps_and_lengths():
    train, probabilities, truth = _case()
    table = error_inspection.passage_table(train, probabilities)
    assert table["true_rank"].tolist() == [2, 3, 1, 2, 2]
    assert table["predicted"].tolist() == ["glm-5.3-flash", "claude-fable-5", "claude-fable-5", "claude-opus-5",
                                           "glm-5.3-flash"]
    np.testing.assert_allclose(table["gap"], probabilities.max(axis=1) - probabilities[np.arange(5), truth])
    assert table["gap"][2] == 0 and table["words"].tolist() == [3, 2, 1, 4, 2]
    summary = error_inspection.direction_summary(table, "claude-fable-5", "glm-5.3-flash")
    assert summary["errors"] == 2 and summary["true_rank"] == {"2": 2, "3": 0, "4+": 0}
    assert summary["correct"] == 1 and summary["gap_below_0.05"] == 0
    swaps = error_inspection.mutual_swaps(table, "claude-fable-5", "glm-5.3-flash")
    assert swaps == {"prompts_both_ways": 1, "prompts": ["a"]}


def test_the_in_sample_boost_counts_every_class_it_moves():
    train, probabilities, truth = _case()
    result = error_inspection.boost(truth, probabilities, "claude-fable-5", 1.5)
    # Row 0 flips to Fable (gained); row 1 stays Fable (still wrong); row 4 stays GLM Flash.
    assert result["gained"] == 1 and result["lost"] == 0
    assert result["by_class"] == {"claude-fable-5": {"gained": 1, "lost": 0}}
    harmful = error_inspection.boost(truth, probabilities, "glm-5.3-flash", 1.1)
    assert harmful["gained"] == 0 and harmful["lost"] == 1  # row 2's narrow Fable call flips to GLM Flash
    assert harmful["by_class"] == {"claude-fable-5": {"gained": 0, "lost": 1}}
    assert harmful["balanced_accuracy_change"] < 0


def test_examples_pick_the_narrowest_median_and_most_confident_errors():
    train, probabilities, _ = _case()
    table = error_inspection.passage_table(train, probabilities)
    picked = error_inspection.examples(table, "claude-fable-5", "glm-5.3-flash")
    assert picked["id"].tolist() == ["r0", "r4"] and picked["kind"].tolist() == ["narrowest", "most confident"]
    assert error_inspection.examples(table, "kimi-k3", "gpt-5.5").empty
    with pytest.raises(KeyError):
        error_inspection.direction_summary(table.drop(columns="margin"), "claude-fable-5", "glm-5.3-flash")
