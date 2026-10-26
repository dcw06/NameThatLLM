"""Case-sensitive lexical features and transparent punctuation/style statistics."""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


class StylometryTransformer(TransformerMixin, BaseEstimator):
    """Small numeric feature block; preserve the original punctuation/case."""

    FEATURE_NAMES = (
        "log_char_count", "log_word_count", "mean_word_length", "mean_sentence_words",
        "uppercase_ratio", "digit_ratio", "whitespace_ratio", "newline_ratio",
        "comma_ratio", "semicolon_ratio", "colon_ratio", "exclamation_ratio",
        "question_ratio", "dash_ratio", "em_dash_ratio", "curly_quote_ratio",
        "straight_quote_ratio", "parenthesis_ratio", "markdown_ratio", "unique_word_ratio",
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
            chars = max(len(text), 1)
            count = max(len(words), 1)
            sentences = max(len(re.findall(r"[.!?]+(?:\s|$)", text)), 1)
            rows.append([
                np.log1p(len(text)), np.log1p(len(words)),
                sum(map(len, words)) / count, len(words) / sentences,
                sum(char.isupper() for char in text) / chars,
                sum(char.isdigit() for char in text) / chars,
                sum(char.isspace() for char in text) / chars,
                text.count("\n") / chars, text.count(",") / chars,
                text.count(";") / chars, text.count(":") / chars,
                text.count("!") / chars, text.count("?") / chars,
                text.count("-") / chars, text.count("—") / chars,
                sum(text.count(char) for char in "‘’“”") / chars,
                sum(text.count(char) for char in "\"'") / chars,
                sum(text.count(char) for char in "()[]{}") / chars,
                sum(text.count(char) for char in "*#`_") / chars,
                len(set(words)) / count,
            ])
        return np.asarray(rows, dtype=np.float64).reshape(-1, len(self.FEATURE_NAMES))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.FEATURE_NAMES, dtype=object)


def build_features(char_max_features: int = 50000, word_max_features: int = 30000) -> FeatureUnion:
    return FeatureUnion(
        [
            ("char", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), lowercase=False, strip_accents=None,
                max_features=char_max_features, min_df=2, sublinear_tf=True, dtype=np.float64,
            )),
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), lowercase=False, strip_accents=None,
                token_pattern=r"(?u)\b\w+\b", max_features=word_max_features,
                min_df=2, sublinear_tf=True, dtype=np.float64,
            )),
            ("style", Pipeline([
                ("extract", StylometryTransformer()),
                ("scale", StandardScaler(with_mean=False)),
            ])),
        ],
        transformer_weights={"char": 1.0, "word": 1.0, "style": 0.1},
        n_jobs=1,
    )
