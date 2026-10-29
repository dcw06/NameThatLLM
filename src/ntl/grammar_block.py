"""Explicit grammatical features for the Claude-GLM specialist (docs/grammar-block.md).

Thirty proportions and rates per passage from a pinned spaCy parse: Universal POS shares,
clauses per sentence, passive voice, and modality and qualification. They are a fixed function
of a fixed pretrained parser, so nothing is fitted to this data and no label is read.
`ntl.specialist_mix` then compares Qwen + StyleDistance with Qwen + StyleDistance + this block
inside the Claude-GLM specialist, at the incumbent's own inner-chosen s.

The feature arithmetic works on plain token records, so it is independent of the parser; spaCy
is imported only to parse (the optional `nlp` dependencies).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from . import claude_glm, ensemble, specialist_mix
from .constants import LABELS
from .data import fingerprint_frame, read_train

SPACY_VERSION = "3.8.16"
MODEL = "en_core_web_sm"
MODEL_VERSION = "3.8.0"
EXCLUDED = ("ner", "lemmatizer")
UPOS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
        "SCONJ", "SYM", "VERB", "X")
CLAUSES = {"relcl": ("relcl",), "advcl": ("advcl",), "ccomp": ("ccomp",), "acl": ("acl",), "xcomp": ("xcomp",),
           "csubj": ("csubj", "csubjpass")}
# Generic hedging adverbs, fixed before any feature was computed and not drawn from these passages.
HEDGES = frozenset({"perhaps", "possibly", "probably", "likely", "arguably", "apparently", "seemingly", "somewhat",
                    "relatively", "generally", "typically", "often", "usually", "largely", "mostly", "roughly",
                    "approximately", "presumably", "conceivably"})
FEATURES = (*(f"pos_{tag.lower()}" for tag in UPOS), *(f"{name}_per_sentence" for name in CLAUSES),
            "words_per_sentence", "passive_per_sentence", "passive_per_verb", "modal_per_100_words",
            "negation_per_100_words", "advmod_per_100_words", "hedge_per_100_words")
BLOCK = "grammar"


@dataclass(frozen=True)
class Token:
    text: str
    pos: str
    tag: str
    dep: str
    is_punct: bool = False
    is_space: bool = False


def features(sentences: list[list[Token]]) -> dict[str, float]:
    """The thirty features of one passage, from its sentences of token records."""
    sentences = [sentence for sentence in sentences if any(not token.is_space for token in sentence)]
    tokens = [token for sentence in sentences for token in sentence if not token.is_space]
    words = sum(not token.is_punct for token in tokens)
    if not sentences or words == 0:
        raise ValueError("A passage needs at least one word")
    count = len(sentences)
    deps = [token.dep for token in tokens]
    passive = deps.count("auxpass")
    verbs = sum(token.pos == "VERB" for token in tokens)
    result = {f"pos_{tag.lower()}": sum(token.pos == tag for token in tokens) / len(tokens) for tag in UPOS}
    result.update({f"{name}_per_sentence": sum(dep in labels for dep in deps) / count for name, labels in CLAUSES.items()})
    result.update({
        "words_per_sentence": words / count,
        "passive_per_sentence": passive / count,
        "passive_per_verb": passive / verbs if verbs else 0.0,
        "modal_per_100_words": 100 * sum(token.tag == "MD" for token in tokens) / words,
        "negation_per_100_words": 100 * deps.count("neg") / words,
        "advmod_per_100_words": 100 * deps.count("advmod") / words,
        "hedge_per_100_words": 100 * sum(token.text.lower() in HEDGES for token in tokens) / words,
    })
    return result


def load_parser():
    """The pinned spaCy pipeline, without named entities or lemmas."""
    import spacy

    nlp = spacy.load(MODEL, exclude=list(EXCLUDED))
    if spacy.__version__ != SPACY_VERSION or nlp.meta.get("version") != MODEL_VERSION:
        raise ValueError(f"Expected spaCy {SPACY_VERSION} with {MODEL} {MODEL_VERSION}")
    return nlp


def parse(texts, nlp, batch_size: int = 64) -> list[list[list[Token]]]:
    """Each passage's sentences as token records."""
    return [[[Token(token.text, token.pos_, token.tag_, token.dep_, token.is_punct, token.is_space)
              for token in sentence] for sentence in doc.sents]
            for doc in nlp.pipe(texts, batch_size=batch_size)]


def block(parsed: list[list[list[Token]]]) -> np.ndarray:
    values = np.array([[row[name] for name in FEATURES] for row in map(features, parsed)], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Grammatical features must be finite")
    return values


def save_archive(path: str | Path, train: pd.DataFrame, values: np.ndarray, metadata: dict) -> None:
    if values.shape != (len(train), len(FEATURES)):
        raise ValueError("One row of features per training passage")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=train["id"].to_numpy().astype(str), features=values, names=np.asarray(FEATURES),
             metadata=json.dumps({**metadata, "train_fingerprint": fingerprint_frame(train)}))


def load_archive(path: str | Path, train: pd.DataFrame) -> np.ndarray:
    """The block, checked against these rows, this feature list and the pinned parser."""
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata"]))
    if (not np.array_equal(archive["ids"], train["id"].to_numpy().astype(str))
            or tuple(archive["names"]) != FEATURES or metadata.get("train_fingerprint") != fingerprint_frame(train)
            or (metadata.get("spacy"), metadata.get("model"), metadata.get("model_version"))
            != (SPACY_VERSION, MODEL, MODEL_VERSION)):
        raise ValueError(f"{path} was not built from these rows with the pinned parser and features")
    return archive["features"]


def check_incumbent(output_dir: str | Path, train: pd.DataFrame, locked: np.ndarray, reference: dict) -> dict:
    """The evaluation's incumbent must reproduce the locked reference's predictions and score."""
    frame = pd.read_parquet(Path(output_dir) / "predictions.parquet").set_index("id").loc[train["id"]]
    predicted = frame["prediction::incumbent"].to_numpy()
    expected = np.asarray(LABELS)[locked.argmax(axis=1)]
    score = float(balanced_accuracy_score(ensemble._targets(train), [LABELS.index(name) for name in predicted]))
    if (predicted != expected).any() or score != reference["balanced_accuracy"]:
        raise ValueError("The evaluation's incumbent does not reproduce the locked reference")
    return {"predictions_identical": True, "balanced_accuracy": score}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("features", help="Parse every training passage and save the block")
    first.add_argument("--train", type=Path, required=True)
    first.add_argument("--output", type=Path, required=True)
    second = commands.add_parser("evaluate", help="Qwen + StyleDistance against the same plus the block")
    for name in ("train", "plan", "qwen-features", "styledistance-features", "grammar-features", "current-oof",
                 "reference", "output-dir"):
        second.add_argument(f"--{name}", type=Path, required=True)
    second.add_argument("--current-inner", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    train = read_train(args.train)
    if args.command == "features":
        if args.output.exists():
            raise ValueError("The block already exists; it must not be overwritten")
        values = block(parse(train["text"].tolist(), load_parser()))
        save_archive(args.output, train, values, {"spacy": SPACY_VERSION, "model": MODEL,
                                                  "model_version": MODEL_VERSION, "excluded": list(EXCLUDED),
                                                  "features": list(FEATURES)})
        print(json.dumps({"rows": int(values.shape[0]), "features": int(values.shape[1])}))
        return 0
    blocks = claude_glm._load_blocks(train, args.qwen_features, args.styledistance_features)
    blocks[BLOCK] = load_archive(args.grammar_features, train)
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    reference = json.loads(args.reference.read_text())
    locked = np.load(args.reference.with_suffix(".npz"), allow_pickle=False)
    if not np.array_equal(locked["ids"], train["id"].to_numpy().astype(str)):
        raise ValueError("The locked reference is for other rows")
    cache = pd.concat([pd.read_parquet(path) for path in args.current_inner], ignore_index=True)
    inputs = [args.plan, args.qwen_features, args.styledistance_features, args.grammar_features, args.current_oof,
              args.reference, *args.current_inner]
    result = specialist_mix.evaluate(train, blocks, (BLOCK,), plan, pd.read_parquet(args.current_oof), cache,
                                     args.output_dir,
                                     provenance={"input_sha256": {str(path): claude_glm._sha256(path) for path in inputs}})
    check = check_incumbent(args.output_dir, train, locked["probabilities"], reference)
    paired = result["paired"]
    print(json.dumps({
        "balanced_accuracy": {name: values["balanced_accuracy"] for name, values in result["systems"].items()},
        "delta": paired["balanced_accuracy_delta"], "delta_95_interval": paired["delta_95_interval"],
        "passes": paired["balanced_accuracy_delta"] >= 0.005 and paired["delta_95_interval"][0] > 0,
        "selected": [item["selected"] for item in result["selection"]], "s": [item["s"] for item in result["selection"]],
        "changes": result["changes"], "incumbent_check": check,
        "option_balanced_accuracy": result["option_balanced_accuracy"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.grammar_block import main as imported_main

    raise SystemExit(imported_main())
