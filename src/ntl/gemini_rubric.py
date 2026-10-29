"""Gemini 3.7 / 3.8 Flash stylistic rubric, as features for the pair specialist.

Fifteen fixed measures of clausal depth, kinetic and copular verbs, systemic and tangible
vocabulary, hedges and absolutes, and opening and closing frames. Each is a rate or a ratio,
so none is raw length; `rubric_length` adds log word and sentence counts. They feed
`ntl.gemini_pair.evaluate` unchanged: a binary specialist on the true-pair rows of each
training partition, integrated into the complete incumbent, with the same nested selection.

Diagnostics, outside the decision: paired sign tests of each measure over the 150 prompts,
binary accuracy of fixed configurations and a length-only control, and what known prompt
groups would be worth (measures centred on their prompt, and the incumbent's probabilities
assigned to distinct models within each prompt). The test file has no prompt identifier, so
the last two are not deployable.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from . import claude_glm, ensemble, gemini_pair, neural
from .data import read_train
from .lora import AdapterConfig, validate_plan

CONFIGURATIONS = {"rubric": (0.01, 0.1, 1.0), "rubric_length": (0.01, 0.1, 1.0),
                  "qwen+rubric": (0.005, 0.05, 0.5)}
DIAGNOSTIC_C = (0.01, 0.1, 1.0)

_WORD = re.compile(r"[a-z']+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_PARTICIPLE = re.compile(r"^[A-Z][a-z]+(?:ing|ed)\b|^(?:Known|Woven|Driven|Given|Taken|Born|Hidden|Frozen|Broken|"
                         r"Chosen|Written)\b")
_SUBORDINATOR = re.compile(r"\b(?:which|whose|whom|wherein|whereby|while|although|whereas|because|since)\b")
_LEXICONS = {  # per 100 words
    "connectives": r"\b(?:wherein|whereby|thereby|thus|consequently|therefore|hence|yielding|moreover|furthermore)\b",
    "kinetic_verbs": r"\b(?:ignit|collaps|bloom|demand|shatter|surg|burst|explod|crash|fractur|erupt|plung|forg)\w*",
    "copulas": r"\b(?:is|are|was|were|be|been|being)\b",
    "systemic": r"\b(?:paradigm|architect|syntax|subroutine|equilibri|framework|continuum|mechanism|system|structur|"
                r"evolution)\w*",
    "tangible_analogy": r"\b(?:imagine|picture|like a|as if|as though)\b",
    "hedges": r"\b(?:merely|not simply|not merely|not just|partially|insofar|not necessarily|rather than|somewhat)\b",
    "absolutes": r"\b(?:forever|complete|completely|total|totally|entire|entirely|utter|utterly|absolute|absolutely|"
                 r"shatter\w*|abyss|decisive|decisively)\b",
}
_OPENS_ANALOGY = re.compile(r"^(?:imagine|picture)\b")
_OPENS_DEFINITION = re.compile(r"\b(?:is a|is the|are the|refers to|known as|defined as)\b")
_COLLECTIVE = re.compile(r"\b(?:we|our|us|humanity|human|collective)\b")
RUBRIC = ("words_per_sentence", "commas_per_sentence", "participial_openings", "subordinators_per_sentence",
          *_LEXICONS, "opens_with_analogy", "opens_with_definition", "collective_conclusion", "final_cadence")
LENGTH = ("log_words", "log_sentences")


def _sentences(text: str) -> list[str]:
    return [sentence for sentence in _SENTENCE.split(text.strip()) if sentence]


def measures(text: str) -> dict[str, float]:
    """The rubric's 15 measures and the two length measures for one passage."""
    sentences = _sentences(text) or [text]
    lower = text.lower()
    words = len(_WORD.findall(lower))
    per_sentence = [len(_WORD.findall(sentence.lower())) for sentence in sentences]
    count = len(sentences)
    result = {
        "words_per_sentence": words / count,
        "commas_per_sentence": text.count(",") / count,
        "participial_openings": sum(bool(_PARTICIPLE.match(sentence)) for sentence in sentences) / count,
        "subordinators_per_sentence": len(_SUBORDINATOR.findall(lower)) / count,
        **{name: 100 * len(re.findall(pattern, lower)) / max(words, 1) for name, pattern in _LEXICONS.items()},
        "opens_with_analogy": float(bool(_OPENS_ANALOGY.match(sentences[0].lower()))),
        "opens_with_definition": float(bool(_OPENS_DEFINITION.search(sentences[0].lower()))),
        "collective_conclusion": float(bool(_COLLECTIVE.search(sentences[-1].lower()))),
        "final_cadence": per_sentence[-1] / max(np.mean(per_sentence), 1),
        "log_words": float(np.log1p(words)),
        "log_sentences": float(np.log(count)),
    }
    return result


def feature_blocks(texts) -> dict[str, np.ndarray]:
    """`rubric` (15 rates and ratios), `length` (log words, log sentences) and `rubric_length` (both)."""
    table = pd.DataFrame([measures(text) for text in texts])
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("Rubric measures must be finite")
    return {"rubric": table[list(RUBRIC)].to_numpy(float), "length": table[list(LENGTH)].to_numpy(float),
            "rubric_length": table[list(RUBRIC + LENGTH)].to_numpy(float)}


def claims(train: pd.DataFrame, blocks: dict[str, np.ndarray]) -> dict:
    """Each measure, 3.7 against 3.8 Flash within the same prompt: means, wins and a two-sided sign test."""
    truth = ensemble._targets(train)
    table = pd.DataFrame(blocks["rubric_length"], columns=list(RUBRIC + LENGTH))
    table["prompt_id"], table["truth"] = train["prompt_id"].to_numpy(), truth
    seven, eight = (table[table["truth"] == index].set_index("prompt_id") for index in gemini_pair.PAIR_INDEX)
    prompts = seven.index.intersection(eight.index).sort_values()
    if len(prompts) != len(seven) or len(prompts) != len(eight):
        raise ValueError("Every prompt needs exactly one passage of each version")
    result = {"prompts": int(len(prompts))}
    for name in RUBRIC + LENGTH:
        first, second = seven.loc[prompts, name].to_numpy(), eight.loc[prompts, name].to_numpy()
        higher_first, higher_second = int((first > second).sum()), int((second > first).sum())
        result[name] = {"mean_3.7": float(first.mean()), "mean_3.8": float(second.mean()),
                        "3.7_higher": higher_first, "3.8_higher": higher_second,
                        "sign_test_p": float(binomtest(higher_first, higher_first + higher_second).pvalue)
                        if higher_first + higher_second else 1.0}
    return result


def prompt_centred(train: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """Each measure minus its mean over the passages of the same prompt (all 18 models)."""
    frame = pd.DataFrame(values)
    return (frame - frame.groupby(train["prompt_id"].to_numpy()).transform("mean")).to_numpy()


def binary_diagnostics(train: pd.DataFrame, plan: pd.DataFrame, blocks: dict[str, np.ndarray],
                       seed: int) -> dict:
    """Binary balanced accuracy on the true-pair rows: each block at each C, fit on the true-pair rows
    of each outer training partition and scored on its outer fold. Not a selection."""
    truth = ensemble._targets(train)
    folds = plan["outer_fold"].to_numpy()
    pair = np.isin(truth, gemini_pair.PAIR_INDEX)
    target = (truth == gemini_pair.PAIR_INDEX[1]).astype(int)
    result = {}
    for name, values in blocks.items():
        for C in DIAGNOSTIC_C:
            scores = np.empty(len(train))
            for fold in range(5):
                fit, rows = np.flatnonzero(pair & (folds != fold)), np.flatnonzero(pair & (folds == fold))
                head = claude_glm.specialist_head((values.shape[1],), neural.FrozenConfig(C=C, seed=seed))
                with threadpool_limits(limits=2):
                    head.fit(values[fit], target[fit])
                    scores[rows] = head.predict_proba(values[rows])[:, 1]
            result[f"{name}@C={C:g}"] = float(balanced_accuracy_score(target[pair], scores[pair] > 0.5))
    return result


def within_prompt_ranking(train: pd.DataFrame, third_eight: np.ndarray) -> float:
    """Share of prompts in which the true 3.8 passage gets the higher 3.8 share than the true 3.7 one."""
    truth = ensemble._targets(train)
    frame = pd.DataFrame({"prompt_id": train["prompt_id"].to_numpy(), "truth": truth, "score": third_eight})
    seven, eight = (frame[frame["truth"] == index].set_index("prompt_id")["score"]
                    for index in gemini_pair.PAIR_INDEX)
    return float((eight.loc[seven.index] > seven).mean())


def prompt_assignment(train: pd.DataFrame, probabilities: np.ndarray) -> np.ndarray:
    """Within each prompt, the assignment of passages to distinct labels that maximizes the summed
    log-probability. Needs prompt groups, which the test file does not have."""
    assigned = np.empty(len(train), dtype=int)
    cost = -np.log(np.clip(probabilities, 1e-300, None))
    for _, rows in train.groupby("prompt_id").indices.items():
        passages, labels = linear_sum_assignment(cost[rows])
        assigned[rows[passages]] = labels
    return assigned


def known_prompts(train: pd.DataFrame, plan: pd.DataFrame, blocks: dict[str, np.ndarray], incumbent: np.ndarray,
                  seed: int) -> dict:
    """What prompt groups would be worth: not deployable on the test file."""
    truth = ensemble._targets(train)
    folds = plan["outer_fold"].to_numpy()
    assigned = prompt_assignment(train, incumbent)
    argmax = incumbent.argmax(axis=1)
    systems = {name: ensemble._system_metrics(truth, values, folds) for name, values in
               (("incumbent", argmax), ("incumbent_assigned", assigned))}
    share = incumbent[:, gemini_pair.PAIR_INDEX[1]] / incumbent[:, gemini_pair.PAIR_INDEX].sum(axis=1)
    centred = {f"centred_{name}": prompt_centred(train, blocks[name]) for name in ("length", "rubric_length")}
    return {
        "assignment": {
            "balanced_accuracy": {name: values["balanced_accuracy"] for name, values in systems.items()},
            "fold_balanced_accuracy": {name: values["fold_balanced_accuracy"] for name, values in systems.items()},
            "pair_recall": {name: {label: systems[name]["recall"][label] for label in gemini_pair.PAIR}
                            for name in systems},
            "swaps": gemini_pair._swaps(truth, argmax, assigned),
        },
        "incumbent_within_prompt_ranking": within_prompt_ranking(train, share),
        "centred_binary": binary_diagnostics(train, plan, centred, seed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--qwen-features", type=Path, required=True)
    parser.add_argument("--styledistance-features", type=Path, required=True)
    parser.add_argument("--current-oof", type=Path, required=True)
    parser.add_argument("--current-inner", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="the locked incumbent (ntl.style_adapt lock), for the known-prompt diagnostics")
    parser.add_argument("--groups", type=Path, help="CSV of id, prompt_id regrouping rows (e.g. by topic)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    train = read_train(args.train)
    blocks = claude_glm._load_blocks(train, args.qwen_features, args.styledistance_features)
    plan = pd.read_csv(args.plan, dtype={"id": str, "prompt_id": str}, keep_default_na=False)
    rubric = feature_blocks(train["text"])
    original = train
    train, plan = claude_glm._grouped(train, plan, args.groups)
    plan = validate_plan(train, plan, AdapterConfig())
    cache = pd.concat([pd.read_parquet(path) for path in args.current_inner], ignore_index=True)
    inputs = [args.plan, args.qwen_features, args.styledistance_features, args.current_oof, args.reference,
              *args.current_inner, *([args.groups] if args.groups else [])]
    seed = neural.FrozenConfig().seed
    result = gemini_pair.evaluate(train, {**blocks, **rubric}, plan, pd.read_parquet(args.current_oof), cache,
                                  args.output_dir, configurations=CONFIGURATIONS,
                                  provenance={"input_sha256": {str(path): claude_glm._sha256(path)
                                                               for path in inputs}})
    diagnostics = {"claims": claims(original, rubric),
                   "binary": binary_diagnostics(train, plan, rubric, seed)}
    if args.groups is None:  # prompt groups, as the training data records them
        reference = json.loads(args.reference.read_text())
        locked = np.load(reference["probabilities_file"], allow_pickle=False)
        if not np.array_equal(locked["ids"], train["id"].to_numpy().astype(str)):
            raise ValueError("The locked reference is for other rows")
        diagnostics["known_prompts"] = known_prompts(train, plan, rubric, locked["probabilities"], seed)
    claude_glm._json(Path(args.output_dir) / "diagnostics.json", diagnostics)
    paired = result["paired"]
    print(json.dumps({
        "balanced_accuracy": {name: round(values["balanced_accuracy"], 4) for name, values in result["systems"].items()},
        "delta": round(paired["balanced_accuracy_delta"], 4),
        "delta_95_interval": [round(bound, 4) for bound in paired["delta_95_interval"]],
        "s": [item["s"] for item in result["selection"]],
        "selected": [item["selected"] for item in result["selection"]],
        "binary": {key: value for key, value in result["binary"].items() if key != "fixed_specialists"},
        "fixed_specialists": {key: round(value, 4) for key, value in result["binary"]["fixed_specialists"].items()},
        "diagnostic_binary": {key: round(value, 4) for key, value in diagnostics["binary"].items()},
        "known_prompts": {key: value for key, value in diagnostics.get("known_prompts", {}).items()
                          if key != "assignment"},
        "assignment": diagnostics.get("known_prompts", {}).get("assignment", {}).get("balanced_accuracy"),
        "swaps": result["swaps"],
        "changes": {key: value for key, value in result["changes"].items() if key != "by_class"},
    }, indent=2))
    return 0


if __name__ == "__main__":
    from ntl.gemini_rubric import main as imported_main

    raise SystemExit(imported_main())
