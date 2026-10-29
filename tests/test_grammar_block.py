"""Grammatical block: feature arithmetic on hand-made token records, the archive's checks, and the
incumbent check. The parser itself is exercised only where the optional spaCy model is installed."""

import json

import numpy as np
import pandas as pd
import pytest

from ntl import grammar_block
from ntl.constants import LABELS
from ntl.grammar_block import Token


def _passage():
    """'The ball was kicked.' and 'It may not rain, perhaps.'"""
    first = [Token("The", "DET", "DT", "det"), Token("ball", "NOUN", "NN", "nsubjpass"),
             Token("was", "AUX", "VBD", "auxpass"), Token("kicked", "VERB", "VBN", "ROOT"),
             Token(".", "PUNCT", ".", "punct", is_punct=True)]
    second = [Token("It", "PRON", "PRP", "nsubj"), Token("may", "AUX", "MD", "aux"), Token("not", "PART", "RB", "neg"),
              Token("rain", "VERB", "VB", "ROOT"), Token(",", "PUNCT", ",", "punct", is_punct=True),
              Token("perhaps", "ADV", "RB", "advmod"), Token(".", "PUNCT", ".", "punct", is_punct=True),
              Token("\n", "SPACE", "_SP", "dep", is_space=True)]
    return [first, second]


def test_features_count_parts_of_speech_voice_and_modality():
    result = grammar_block.features(_passage())
    assert list(result) == list(grammar_block.FEATURES) and len(result) == 30
    # 12 non-space tokens, 9 words, 2 sentences; the space token counts nowhere.
    assert result["pos_aux"] == pytest.approx(2 / 12) and result["pos_punct"] == pytest.approx(3 / 12)
    assert result["pos_verb"] == pytest.approx(2 / 12) and sum(result[f"pos_{t.lower()}"] for t in grammar_block.UPOS) == pytest.approx(1)
    assert result["words_per_sentence"] == 4.5
    assert result["passive_per_sentence"] == 0.5 and result["passive_per_verb"] == 0.5
    for name in ("modal", "negation", "advmod", "hedge"):
        assert result[f"{name}_per_100_words"] == pytest.approx(100 / 9)
    assert all(result[f"{name}_per_sentence"] == 0 for name in grammar_block.CLAUSES)


def test_clauses_are_counted_per_sentence_and_empty_passages_are_refused():
    sentence = [Token("who", "PRON", "WP", "nsubj"), Token("came", "VERB", "VBD", "relcl"),
                Token("left", "VERB", "VBD", "ROOT"), Token("because", "SCONJ", "IN", "mark"),
                Token("tired", "ADJ", "JJ", "advcl"), Token("being", "AUX", "VBG", "csubjpass")]
    result = grammar_block.features([sentence, [Token("Yes", "INTJ", "UH", "ROOT")]])
    assert result["relcl_per_sentence"] == 0.5 and result["advcl_per_sentence"] == 0.5
    assert result["csubj_per_sentence"] == 0.5 and result["passive_per_verb"] == 0
    with pytest.raises(ValueError, match="at least one word"):
        grammar_block.features([[Token(".", "PUNCT", ".", "punct", is_punct=True)]])


def test_the_archive_refuses_other_rows_features_or_parsers(tmp_path):
    train = pd.DataFrame({"id": ["a", "b"], "prompt_id": ["p", "p"], "model": [LABELS[0], LABELS[1]],
                          "text": ["one", "two"]})
    values = np.arange(60, dtype=float).reshape(2, 30)
    metadata = {"spacy": grammar_block.SPACY_VERSION, "model": grammar_block.MODEL,
                "model_version": grammar_block.MODEL_VERSION}
    grammar_block.save_archive(tmp_path / "ok.npz", train, values, metadata)
    assert np.array_equal(grammar_block.load_archive(tmp_path / "ok.npz", train), values)
    with pytest.raises(ValueError, match="pinned parser"):
        grammar_block.load_archive(tmp_path / "ok.npz", train.iloc[::-1].reset_index(drop=True))
    grammar_block.save_archive(tmp_path / "old.npz", train, values, {**metadata, "model_version": "3.7.1"})
    with pytest.raises(ValueError, match="pinned parser"):
        grammar_block.load_archive(tmp_path / "old.npz", train)
    with pytest.raises(ValueError, match="One row"):
        grammar_block.save_archive(tmp_path / "bad.npz", train, values[:1], metadata)


def test_the_incumbent_check_requires_identical_predictions(tmp_path):
    truth = [0, 1, 2, 0]
    train = pd.DataFrame({"id": list("abcd"), "prompt_id": list("ppqq"), "model": [LABELS[i] for i in truth],
                          "text": ["x"] * 4})
    locked = np.eye(len(LABELS))[[0, 1, 0, 0]]
    frame = pd.DataFrame({"id": list("abcd"), "prediction::incumbent": [LABELS[i] for i in (0, 1, 0, 0)]})
    frame.to_parquet(tmp_path / "predictions.parquet", index=False)
    score = (1 + 1 + 0) / 3
    assert grammar_block.check_incumbent(tmp_path, train, locked, {"balanced_accuracy": score})["predictions_identical"]
    with pytest.raises(ValueError, match="does not reproduce"):
        grammar_block.check_incumbent(tmp_path, train, np.eye(len(LABELS))[[0, 1, 2, 0]], {"balanced_accuracy": score})


def test_the_pinned_parser_reads_passive_voice_modality_and_relative_clauses():
    pytest.importorskip("spacy")
    pytest.importorskip(grammar_block.MODEL)
    parsed = grammar_block.parse(["The ball was kicked by the boy who laughed. It may not rain."],
                                 grammar_block.load_parser())
    result = grammar_block.features(parsed[0])
    assert result["passive_per_sentence"] > 0 and result["modal_per_100_words"] > 0
    assert result["relcl_per_sentence"] > 0 and result["negation_per_100_words"] > 0
    assert json.dumps(result)  # finite, serializable
