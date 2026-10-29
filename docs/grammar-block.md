# Grammatical block for the Claude–GLM specialist

**Question.** Does a small block of explicit grammatical features, added to the existing
seven-class Claude–GLM specialist, improve the complete incumbent?

**Why this experiment.** The [Fable / GLM error inspection](fable-glm-errors.md) found that
most of these errors are not narrow decisions that a score adjustment could recover. This
experiment therefore tests a different representation rather than another class weight, and
does so without another large encoder search.

**References to beat:**
- 47.01% grouped CV (the complete incumbent: blend plus Claude–GLM specialist);
- 42.30% on the eight-topic split;
- 0.429 on the public leaderboard.

**What is not used.** The inspection noted that short GLM Flash passages lean towards Fable.
That observation came from looking at every label, so it is not used here, as a rule or as a
feature.

## Design

This section was written before any parser was installed or any feature computed.

**Parser.** spaCy 3.8.16 with `en_core_web_sm` 3.8.0, both pinned in the `nlp` optional
dependencies. Named-entity recognition and the lemmatizer are disabled. The parser is a fixed
pretrained model: nothing is fitted to this data and no label is read.

**The structural block.** Thirty features per passage, each a proportion or rate, so none is
raw length. N is the passage's non-punctuation, non-space tokens; S is its sentences, as the
parser splits them.

| Group | Features |
| --- | --- |
| Part of speech (17) | Share of non-space tokens with each Universal POS tag: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X |
| Clauses (7) | Per sentence: relative clauses (`relcl`), adverbial clauses (`advcl`), clausal complements (`ccomp`), clausal noun modifiers (`acl`), open complements (`xcomp`), clausal subjects (`csubj`, `csubjpass`); and N / S |
| Voice (2) | Passive auxiliaries (`auxpass`) per sentence; passive auxiliaries per VERB token (0 when there is none) |
| Modality and qualification (4) | Per 100 N: modal verbs (tag `MD`), negations (`neg`), adverbial modifiers (`advmod`), and hedging adverbs from a fixed generic list |

**The hedging list** is generic, fixed before any feature was computed, and not drawn from
these passages: *perhaps, possibly, probably, likely, arguably, apparently, seemingly,
somewhat, relatively, generally, typically, often, usually, largely, mostly, roughly,
approximately, presumably, conceivably*.

**Specialist and integration.** These reuse `ntl.specialist_mix` unchanged, as in the LUAR-MUD
and LUSR experiment, with the grammatical block as the added block.
- **The augmented specialist** is the existing recipe: StandardScaler, logistic regression
  with `C = 0.005`, every block after Qwen scaled to Qwen's total variance. It is fit on Qwen
  + StyleDistance + the grammatical block.
- **Mixing.** The existing and augmented seven-class distributions are mixed,
  `q = (1 − a) q_existing + a q_augmented`. At `a = 1` this is exactly Qwen + StyleDistance
  + the block in place of Qwen + StyleDistance. The mix feeds the incumbent's integration at
  its own inner-chosen `s`, so `a = 0` is the incumbent exactly.
- **Selection.** Within the original five prompt folds and their two inner folds
  (`artifacts/lora/plan.csv`):
  1. The inner folds choose `s` first, as the incumbent does.
  2. They then choose `a ∈ {0, ¼, ½, 1}` by mean inner 18-class balanced accuracy.
  3. Ties keep the incumbent, then the smaller `a`.
- **Check.** The evaluation's incumbent must reproduce the locked reference's predictions
  and balanced accuracy exactly.

**Promotion rule,** unchanged:
- an unrounded gain of at least +0.5 points;
- a 95% paired prompt-bootstrap interval above zero (10,000 resamples, seed 2026).

Only a passing candidate repeats the procedure on the eight-topic split. Only after both
would anything be packaged; deployment would also need the parser in the offline Kaggle
environment.

**Reported:**
- overall balanced accuracy and the paired interval;
- the seven classes' recall;
- the targeted Fable / GLM confusions;
- gains and losses inside and outside the specialist's classes;
- the selected `a` and `s` per fold;
- as a hindsight diagnostic, each `a` applied in every fold.

## Result: original prompt folds (October 28, 2026)

```sh
PYTHONPATH=src .venv/bin/python -m ntl.grammar_block features --train data/raw/train.csv \
  --output artifacts/grammar-block/grammar-train.npz
PYTHONPATH=src .venv/bin/python -m ntl.grammar_block evaluate --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv \
  --qwen-features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --styledistance-features artifacts/claude-glm/styledistance-train.npz \
  --grammar-features artifacts/grammar-block/grammar-train.npz \
  --current-oof artifacts/qwen-blend/oof.parquet \
  --current-inner artifacts/claude-glm/current-inner/fold{0,1,2,3,4}.parquet \
  --reference artifacts/styledistance-adapt/reference-prompts.json \
  --output-dir artifacts/grammar-block/prompts
```

**The block:**
- The pinned parser read all 2,699 passages in about 11 seconds.
- Parsing the same 200 passages twice gave identical features.
- Every feature varies across passages.
- A few long passages without sentence breaks give extreme values (up to 226 words per
  sentence). The specification was kept as frozen.

**Check.** The evaluation's incumbent reproduces the locked reference's predictions exactly
(balanced accuracy 0.4700745712155108), with `s` = ¼, ¼, ½, ¼, ¼.

| System | Grouped BA | Change | 95% interval | Gained / lost |
| --- | ---: | ---: | --- | --- |
| Incumbent | 0.4701 | — | — | — |
| With the grammatical block | 0.4697 | −0.04 | [−0.41, +0.33] | 14 / 15 |

**Not promoted.** The change is −0.04 points, and 58% of resamples are at or below zero. As
declared, the eight-topic check was not run and nothing was packaged.

**Selection:**

| Fold | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Selected | Incumbent | Incumbent | `a` = 1 | Incumbent (tie) | `a` = ¼ |
| Best mix's inner margin (points) | −0.09 | −0.14 | +0.002 | 0.00 | +0.19 |
| Held-out change (points) | 0.00 | 0.00 | 0.00 | 0.00 | −0.19 |

In fold 2 the block's 50 changed passages cancel exactly.

**In hindsight.** Applied in every fold, `a = ¼` would gain +0.22 points, `a = ½` would lose
0.11, and `a = 1` would lose 0.67. At `a = 1`, Qwen + StyleDistance + the block replaces
Qwen + StyleDistance.

**Specialist classes** (recall, incumbent → candidate):
- Fable: 20.1% → 22.1%;
- GLM 5.3: 26.7% → 27.3%;
- GLM Flash: 32.0% → 32.0%;
- Opus 4.6: 65.3% → 64.0%;
- Opus 4.8: 39.3% → 38.7%;
- Opus 5: 76.7% → 76.0%;
- Sonnet 5: 51.3% → 51.3%.

Outside the seven, one GPT-5.5 passage is lost.

**Targeted confusions** (incumbent → candidate):
- Fable → GLM: 19 → 17;
- Fable → GLM Flash: 28 → 28;
- GLM → Fable: 21 → 19;
- GLM Flash → Fable: 16 → 17;
- GLM → GLM Flash: 28 → 30;
- GLM Flash → GLM: 30 → 29.

**Diagnostic, not part of the design.** This was added after the result. Each specialist was
fit on the outer training partition with the existing recipe, and scored on the held-out true
Claude and GLM passages:

| Features | Seven-class BA | Fable / GLM Flash passages called correctly | GLM / GLM Flash passages called correctly |
| --- | ---: | ---: | ---: |
| Grammatical block alone | 25.0% | 7.4% | 9.7% |
| StyleDistance alone | 33.6% | 22.7% | 19.0% |
| Qwen + StyleDistance (existing) | 46.2% | 30.8% | 32.7% |
| Qwen + StyleDistance + block | 44.4% | 26.8% | 27.3% |

**What this answers:**
- **Weak signal.** The grammatical rates carry some seven-class signal (25% against 14%
  chance), much less than StyleDistance.
- **They dilute the specialist.** At the recipe's block weight, equal total variance to Qwen,
  adding them lowers the specialist's separation overall and on both targeted pairs.
- **The inner folds did not find the small benefit.** Only a light mix helps, and only in
  hindsight.
- **Conclusion.** Explicit part-of-speech, clause, voice and modality rates at this
  granularity add no Fable / GLM information beyond Qwen + StyleDistance.
- **Not tested.** A smaller weight for the block was not tested; doing so now would be a new,
  post-hoc experiment.

**Dependencies.** The parser is in the new `nlp` optional dependencies: spaCy 3.8.16, with
`en_core_web_sm` 3.8.0 mapped to its release wheel in `[tool.uv.sources]`. The default
install is unchanged. The Kaggle notebooks do not need it unless a parser-based candidate is
promoted.
