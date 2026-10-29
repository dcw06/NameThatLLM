# Gemini 3.7 / 3.8 Flash stylistic rubric

**Status: superseded before classifier evaluation.** The
[seven-marker study](gemini-marker-study.md) replaces this proposal. This 15-feature
classifier study will not run in addition to its successor. Preserve this document
and its code as historical work. The recorded findings below are descriptive claim
checks; no fitted rubric cross-validation results are recorded. The remaining design
describes the superseded proposal, not current execution instructions.

**Question.** Do hand-written stylistic measures, meant to separate Gemini 3.7 from 3.8
Flash independently of raw length, recover the complete incumbent's errors between the two?

The rubric was proposed after inspecting the 300 true 3.7 and 3.8 Flash training passages
(`artifacts/datasets/gemini-3.7-3.8-flash.csv`). It names four kinds of cue:
- **Clausal depth.** 3.7 is said to chain subordinate and participial clauses; 3.8 to
  favour short, active, kinetic predicates.
- **Metaphor.** 3.7 is said to prefer systemic, meta-scientific vocabulary; 3.8 tangible
  "imagine an X" analogies.
- **Epistemic calibration.** 3.7 is said to hedge ("merely", "not simply"); 3.8 to use
  absolutes ("forever", "completely").
- **Paired-prompt calibration.** Length would be judged relative to the other passages
  of the same prompt.

**References to beat:**
- 47.01% grouped CV (the complete incumbent: blend plus Claude–GLM specialist);
- 42.30% on the eight-topic split;
- 0.429 on the public leaderboard.

## What can be deployed

The test file has only `id` and `text`, with no prompt identifier. So:
- **Prompt-relative features cannot be deployed.** They are measured only as a
  diagnostic of what known prompt groups would be worth.
- **Restricting the output to 3.7 or 3.8 does not apply.** The test is 18-class. The
  quoted 59.7% is the incumbent's forced choice on passages already known to be 3.7 or
  3.8.

## Checking the claims first

Within each of the 150 prompts there is one 3.7 and one 3.8 passage, so every claim is
tested as a paired comparison with a sign test (`scratchpad` analysis, reproduced in
the results).

| Claim | Finding |
| --- | --- |
| 3.7 is longer | Yes: 118 against 101 words; longer in 89 of 150 prompts (p = 0.02) |
| The incumbent leans on length | Yes: within each version, its 3.7-versus-3.8 log-odds have Spearman −0.45 with word count |
| 3.7 has deeper clauses | Barely: 24.0 against 22.8 words per sentence (p = 0.005). Commas, relative clauses and connectives do not differ; participial openings lean the other way |
| 3.8 has about double the kinetic verbs | No: 0.85 against 0.63 per 100 words (p = 0.09) |
| 3.7 hedges more | Marginal: 0.24 against 0.19 per 100 words (p = 0.07) |
| 3.8 uses more absolutes | No difference |
| Systemic against tangible vocabulary | No difference |

The rubric's cue lists were chosen by reading all 300 passages, so its cross-validated
scores are optimistic: the classifier's weights are fit inside each fold, but the
feature definitions were chosen with all folds in view.

## Design

This section was written before any evaluation ran. It reuses the
[Gemini pair specialist](gemini-pair.md) protocol unchanged, with new features.

**Features** (`ntl.gemini_rubric`), fixed functions of the text with nothing fitted:

| Measure | Definition |
| --- | --- |
| Words per sentence | Words over sentences (split at `.`, `!`, `?`) |
| Commas per sentence | |
| Participial openings | Share of sentences opening with a capitalized `-ing` or `-ed` word, or *known, woven, driven, given, taken, born, hidden, frozen, broken, chosen, written* |
| Subordinators per sentence | *which, whose, whom, wherein, whereby, while, although, whereas, because, since* |
| Connectives | *wherein, whereby, thereby, thus, consequently, therefore, hence, yielding, moreover, furthermore*, per 100 words |
| Kinetic verbs | Stems *ignit, collaps, bloom, demand, shatter, surg, burst, explod, crash, fractur, erupt, plung, forg*, per 100 words |
| Copulas | *is, are, was, were, be, been, being*, per 100 words |
| Systemic vocabulary | Stems *paradigm, architect, syntax, subroutine, equilibri, framework, continuum, mechanism, system, structur, evolution*, per 100 words |
| Tangible analogy | *imagine, picture, like a, as if, as though*, per 100 words |
| Hedges | *merely, not simply, not merely, not just, partially, insofar, not necessarily, rather than, somewhat*, per 100 words |
| Absolutes | *forever, complete(ly), total(ly), entire(ly), utter(ly), absolute(ly), shatter-, abyss, decisive(ly)*, per 100 words |
| Opens with an analogy | First sentence starts with *Imagine* or *Picture* |
| Opens with a definition | First sentence contains *is a, is the, are the, refers to, known as, defined as* |
| Collective conclusion | Last sentence contains *we, our, us, humanity, human, collective* |
| Final-sentence cadence | Last sentence's words over the mean sentence's words |

Every measure is a rate or a ratio, so none is raw length. One configuration adds log word
count and log sentence count.

**Configurations,** each a standardized logistic regression fit on the true-pair passages
of the training partition:

| Configuration | Blocks | `C` grid |
| --- | --- | --- |
| `rubric` | The 15 measures | 0.01, 0.1, 1 |
| `rubric_length` | The 15 measures plus log words and log sentences | 0.01, 0.1, 1 |
| `qwen+rubric` | Frozen Qwen plus the 15 measures, the rubric block scaled to Qwen's total variance as in the Claude–GLM specialist | 0.005, 0.05, 0.5 |

**Integration, selection and promotion,** exactly as for the pair specialist:
- `p′ₖ = (1 − a) pₖ + a M qₖ` on the two versions, with `a ∈ {¼, ½, ¾, 1}`.
- The inner folds choose `s` as the incumbent does, then one of 37 options: the
  incumbent, or a configuration, `C` and `a`.
- Ties go to the incumbent, then the smaller `a`, then the configuration order above,
  then the smaller `C`.
- The candidate is promoted only with a gain of at least +0.5 points and a 95% paired
  prompt-bootstrap interval above zero (10,000 resamples, seed 2026). Only then does it
  repeat on the eight-topic split and get packaged.

**Diagnostics.** None of these is part of the decision.
- Binary balanced accuracy on the 300 true-pair passages, for every configuration and
  `C`, and for a length-only control (log words and log sentences).
- **Known prompt groups.** Not deployable, because the test has no prompt identifier.
  - The rubric and length measures centred on the mean over each prompt's 18 passages.
  - The incumbent's own probabilities, with each prompt's passages assigned to distinct
    models by a maximum-likelihood assignment over the 18 labels.
