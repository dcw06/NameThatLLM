**Gemini 3.7 / 3.8 Flash marker study — protocol version 2, October 28, 2026.**

This is a prospective execution plan for one small exploratory study. Implementation,
preflight, and specification freezing precede all model fitting. No proposed classifier
has been evaluated under this protocol. The marker lists followed exposure to labeled
passages, so neither nested validation nor a favorable interval makes the study
confirmatory. The deliverable is a diagnostic report, reproducible predictions, and
a recorded engineering-eligibility decision, including null or harmful results.

This protocol supersedes the unrun [15-feature rubric proposal](gemini-rubric.md)
and version 1 of this document. The rubric classifier study will not also run. Its
proposal, code, and descriptive claim checks remain historical records; no completed
rubric-classifier evaluation is recorded. These are revisions leading to one declared
marker evaluation, not two opportunities to obtain a favorable result.

This revision changes scope documentation, provenance, reporting burden, and adoption
policy. It leaves the C grid, binary-C-then-integration-weight selection procedure,
three models, seven features, and numerical eligibility thresholds unchanged.

The two questions are whether the seven marker features add information beyond the
specified length control, and whether the style-only specialist improves the complete
18-class incumbent. Binary separation, direct-swap correction, and complete-system gains
are separate outcomes. Only the style-only model is eligible for integration.

**1. Keep the scope fixed.** Use seven features and three new logistic-regression models:

| Model | Inputs | Role |
| --- | --- | --- |
| Length-only | Natural log of word count | Diagnostic control |
| Style-only | The seven rates below | The sole integration candidate |
| Length + style | Natural log of word count and the seven rates | Incremental-information diagnostic |

Use the existing five outer prompt folds, two inner prompt folds, complete incumbent
recipe, and probability-redistribution method. Add no markers, inflections, embeddings,
calibration models, learned thresholds, class weighting, training resampling, feature selection,
or additional validation layer. Refitting existing incumbent components to construct
partition-correct predictions is required and does not introduce a new candidate model.

The expected gain is modest: previous pair specialists scored about 54–61% against
the incumbent's 59.7%, and these markers are sparse. These observations motivate
restrained expectations; they do not establish an accuracy ceiling. Earlier rubric
p-values concern different dictionaries and are descriptive context, not tests of
these exact features.

One net correction among the two 150-support classes contributes 0.037037 percentage
points to 18-class balanced accuracy; 14 contribute 0.518519 points. This conversion
applies only when changed correctness concerns those classes. Changes involving
other classes still require the full macro-recall calculation.

Toy specialists illustrate sensitivity to artificial accuracies, score amplitudes,
error relationships, integration weights, and behavior on non-pair passages. They
do not estimate this study's power or chance of passing the complete interval-plus-
topic gates. Save a simulation's generator, seed, trial count, and assumptions if
citing it; the reported percentage table has not been independently reproduced here.
A neutral S = 0.5 check on the locked incumbent gives +0.037 points at a = 0.25 and
-0.074 points at a = 0.5. This is limited evidence about neutral redistribution,
not proof that redistribution can never pass a gate. Simulations do not select or
change this study's settings.

**2. Fix the population and references.** The binary population is every official passage
labeled gemini-3.7-flash or gemini-3.8-flash: 300 rows, 150 per version, from 150 complete
prompt pairs. Set y = 0 for 3.7 and y = 1 for 3.8. Include passages routed elsewhere and
passages the incumbent gets right. The 92 direct swaps and the earlier 570-row Gemini
cohort do not define this fitting population.

Integration is evaluated on every one of the 2,699 official training rows, with all
18 labels in canonical column order. Recompute these preflight references:

| Reference | Required value |
| --- | ---: |
| Incumbent forced binary choice on the 300 rows | 179/300 = 0.5966666666666667 |
| Incumbent 18-class correct predictions on those rows | 139/300 |
| Complete incumbent prompt-grouped balanced accuracy | 0.4700745712155108 |
| Complete incumbent eight-topic balanced accuracy | 0.423027591349739 |

Stop on a row, truth, grouping, fold, probability, or unexplained reference mismatch.
Do not fix a mismatch by changing the expected score or silently substituting artifacts.

**3. Freeze the feature extractor exactly.** Tokenize the original, unmodified text with
Python's Unicode regular-expression semantics:

```python
TOKEN_RE = re.compile(r"\b\w+(?:['’\-]\w+)*\b", re.UNICODE)
matches = list(TOKEN_RE.finditer(text))
tokens = [match.group(0).casefold() for match in matches]
W = len(tokens)
```

Retain the original token spans. Perform no Unicode normalization, apostrophe conversion,
hyphen removal, stemming, lemmatization, or markup stripping. Match dictionary entries
against complete casefolded tokens, counting every occurrence.

| Feature | Complete frozen marker list |
| --- | --- |
| Action markers | shatter, ignite, collapse, bloom, demands, pierce, erupt |
| Qualifying expressions | merely, essentially, partially, relatively, not simply |
| Formal connectors | wherein, whereby, thus, consequently, yielding |
| Systemic terminology | paradigm, framework, syntax, subroutines, equilibrium |
| Sensory terminology | void, abyss, boiling, froth, molten, violent, tears |
| Semicolons | Literal ASCII semicolon, U+003B |
| Colons | Literal ASCII colon, U+003A |

Count "not simply" once for each adjacent token pair equal to not and simply whose
intervening original substring is nonempty and consists entirely of whitespace
(str.isspace()). Spaces, tabs, newlines, and nonbreaking spaces qualify; punctuation,
hyphens, and zero-width spaces do not. There is no stopword removal before matching.

Count literal punctuation everywhere, including quotations, URLs, ratios, Markdown,
and code. Fullwidth punctuation is not converted. The rate for every feature j is
f_j = 100 * occurrences_j / W. Use natural log(W), not log1p(W), for the length input.
Numeric tokens and underscores count as words because the original audit regex uses
Unicode \w. This is a marker-frequency representation, not a grammatical parser.

Use a few focused extractor checks before fitting: case-insensitive complete-token
matching and compounds, repeated counts, whitespace-only phrase matching versus
punctuation, and literal ASCII punctuation. These protect the declared definitions;
an extensive Unicode test suite is not required.

Require W > 0 for all official evaluation rows during preflight. The current 2,699 rows
satisfy this (minimum 36 tokens). Stop and investigate if the frozen inputs violate it;
do not drop rows or silently change the denominator. For future inference only, W = 0
uses the complete incumbent vector unchanged and records that fallback; no division,
logarithm, or specialist score is computed for that row.

Generate all descriptive counts, per-passage rates, length bins, and inference features
with this same extractor. Keep pooled occurrences per 100 pooled words separate from
the arithmetic mean of passage-level rates.

The descriptive tokenizer audit gives 32,672 words across the pair population, with
mean counts 117.13 for 3.7 and 100.69 for 3.8. Marker coverage remains 89/300 all-seven-zero
rows (45/44 by version) and 138/300 all-five-lexical-zero rows (73/65). Recompute and save
these under the implemented extractor. They describe sparsity and do not select features.
Within one fitted style-only model, all-seven-zero passages share S. Length-control
scores can still differ with W, and integrated predictions can still differ because
their incumbent vectors differ.

Before freezing, add a short marker-selection history to the study record. Preserve
available prompts, notes, and analyses, and distinguish documented passage reading,
label-wise counts, tried alternatives, and external suggestions. Record what is
known and mark missing details as unknown. The original audit program and a complete
selection log are not currently recorded in the repository; the tokenizer was supplied
in the discussion and is pinned explicitly above.

The available record does not establish how the exact forms demands, subroutines,
and tears, the sensory list, or the punctuation features were selected. Do not call
the process reading-only, assert that label-count searching occurred, or invent a
retrospective history. Missing history remains an exploratory limitation; do not
start another word search to fill the gap.

The CPU pipeline's access to related words and punctuation does not make these pooled,
normalized rates an identical representation or imply that any gain must be selection
bias. Both useful representation choices and label-informed selection may contribute;
this repeatedly examined dataset cannot cleanly separate their effects.

**4. Audit construction separately from artifact identity.** Use these existing inputs:

| Purpose | Existing artifact |
| --- | --- |
| Official training data | data/raw/train.csv |
| Pair export for row-by-row checks | artifacts/datasets/gemini-3.7-3.8-flash.csv |
| Original outer and inner folds | artifacts/lora/plan.csv |
| Locked prompt incumbent | artifacts/styledistance-adapt/reference-prompts.json and reference-prompts.npz |
| Prompt base-blend OOF | artifacts/qwen-blend/oof.parquet |
| Prompt base-blend inner caches | artifacts/claude-glm/current-inner/fold0.parquet through fold4.parquet |
| Eight-topic mapping | artifacts/claude-glm/groups-topics8.csv |
| Topic outer and inner folds | artifacts/lora/plan-topics8.csv |
| Locked topic incumbent | artifacts/styledistance-adapt/reference-topics8.json and reference-topics8.npz |
| Topic base-blend OOF | artifacts/topic-cv/topics-8/blend/oof.parquet |
| Topic base-blend inner caches | artifacts/claude-glm/current-inner-topics8/fold0.parquet through fold4.parquet |
| Existing frozen Qwen features for the incumbent | artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz |
| Existing frozen StyleDistance features for the incumbent | artifacts/claude-glm/styledistance-train.npz |

Freeze file SHA-256 values, dataset content fingerprint, exact row/label order,
fold/group identities, feature provenance, all ten cache hashes, source hashes,
dependency-lock hash, and runtime versions in a manifest. The official content
fingerprint is d76a05125fad49d450e61ec43e78ac5fee9d6a291607386a73ce1046511a301a.

Hashes establish which artifacts were used. Separately audit the cache-building code
and partition membership: every supervised head fit, vocabulary, and scaler excludes
the inner-validation groups it predicts. The base blend's own internal tuning remains
inside that inner fitting partition, as its existing recipe requires. All fitting and
selection exclude the complete outer holdout. Preserve explicit fitting/validation ID
and group manifests. Existing cache row coverage and grouping checks alone do not
establish construction correctness; document reconstruction or independent checks.
Rebuild any cache whose construction cannot be established, into a fresh location
under the same declared recipe, and freeze its identity before use.

The incumbent's s and the new C and a choices may use inner-validation labels in
their declared sequence. Their resulting inner scores are selection criteria, not
independent performance estimates or data for an additional calibration fit. This
distinction preserves the existing procedure without requiring a new validation layer.

The inner cache files contain CPU–Qwen base-blend probabilities, not the complete
incumbent. Reconstruct the Claude–GLM stage on each inner fitting partition and retain
the existing procedure for selecting its weight s. Do not substitute globally saved
OOF predictions or a final all-data classifier for these inner predictions.

Verify that the reconstructed outer incumbent agrees with its lock: identical IDs,
labels, folds, argmax predictions, and numerically matching complete vectors, recording
the maximum absolute vector difference (required <= 1e-12). Use the locked vectors as
the outer comparison reference. The a = 0 branch returns its supplied incumbent
vectors unchanged, verified by np.array_equal, rather than recomputing or renormalizing.

**5. Fix the model and numerical settings.** Each new classifier uses a fresh
StandardScaler followed by logistic regression, with float64 inputs and these settings:

| Setting | Frozen choice |
| --- | --- |
| Regularization | L2; C in ascending order [0.01, 0.1, 1] |
| Solver | lbfgs |
| Intercept | Fitted and unpenalized |
| Class/sample weighting | None |
| Maximum iterations | 2,000 |
| Solver tolerance | 1e-6 |
| Seed | 2026 |
| Standardization | Training means and standard deviations only; constant columns use scale 1 |
| Binary decision | Predict 3.8 only when S > 0.5; otherwise 3.7 |
| Selection tie tolerance | Existing ntl.selection tolerance, 1e-12 |

Pin the resolved software environment before fitting. Treat ConvergenceWarning or
nonfinite inputs, parameters, or probabilities as a failed run; do not silently accept
an unfinished fit, discard a failing C, or widen the grid. Record n_iter_ and convergence
status with the fit settings. Detailed per-fit coefficient and scaler exports are
optional; source, inputs, settings, and saved predictions must permit reconstruction.

**6. Evaluate the complete selection procedure once.** For each original outer fold:

1. Exclude its 30 complete prompts from all fitting and selection. This leaves
   240 true-pair fitting passages and 60 true-pair holdout passages.
2. Use the two existing inner prompt splits. Each fits on 120 true-pair rows and
   validates on the other 120; the incumbent stages use their corresponding full
   18-class fitting partitions. Verify both labels occur in every binary partition.
3. For each of the three models and each C, fit a fresh scaler and classifier using
   only inner-fitting pair rows. Score on inner-validation pair rows. For style-only,
   also retain scores on every inner-validation row for integration.
4. Select C separately for each model by the unweighted mean of the two inner binary
   balanced accuracies. Scores tied within 1e-12 choose the smaller C. No outer
   result enters this choice.
5. For style-only at its selected C, select a from [0, 0.25, 0.5] by mean inner
   18-class balanced accuracy after integration into the partition-correct complete
   incumbent. Ties choose 0, then 0.25, then 0.5. Retain the incumbent's own declared
   selection of s; do not replace it with a globally selected value.
6. Refit each of the three selected heads on all 240 outer-fitting pair passages.
   Record their binary holdout probabilities. Apply the locked style-only C and a
   to all outer holdout rows in the 18-class task. Predict that outer partition once.

The C-then-a order is intentional. Reusing these inner validation scores is part of
the declared selection procedure; inner scores are not unbiased performance claims.
Do not choose among the three models from outer scores or retune the threshold.

**7. Integrate with the existing pair-mass rule.** For every eligible row, let
M = p_3.7 + p_3.8 and let S be the style-only estimate favoring 3.8:

```text
p'_3.8 = (1 - a) * p_3.8 + a * M * S
p'_3.7 = (1 - a) * p_3.7 + a * M * (1 - S)
```

Leave the other 16 columns unchanged. S on a non-pair passage is a conditional
preference, not evidence of Gemini authorship. Preserve canonical 18-class argmax
tie handling. Check finite nonnegative probabilities, sums of one and conserved pair
mass (absolute tolerance 1e-12), and exact equality of the untouched columns. The
a = 0 branch must preserve the complete supplied vectors exactly.

For binary probability comparisons use q0 = p_3.8 / M. If M is exactly zero, set q0
to 0.5 and record the case; integration preserves the zero pair mass. Apply the same
normalization to the integrated system for its binary probability diagnostics.

**8. Predeclare reports and interpretation.** Save pooled held-out results and all
five fold results. The principal binary diagnostic is BA(length + style) minus
BA(length-only). Also compare each model against the forced incumbent's 179/300,
and compare the integrated system's forced binary choice against that reference.

For every binary predictor report balanced accuracy, ordinary accuracy, both class
recalls, correct counts, paired gains/losses, Brier score, and natural-log log loss.
Use raw probabilities for Brier score and classification; clip to [1e-15, 1-1e-15]
only for log-loss calculation. A reliability plot is optional: if generated, use ten
equal-width bins [0, 0.1), ..., [0.9, 1] with counts and observed rates. It is descriptive
and does not fit a calibration adjustment.

Report coverage and results in word-count bins below 60, 60–99, 100–149, and at least
150, using this extractor's W. Include row, prompt, and per-class counts. If a stratum
lacks either class, its two-class balanced accuracy is undefined, not an average over
the remaining class.

For the complete system report official macro recall over all 18 classes, per-class
recall changes, gains/losses, and these distinct mechanisms:

| Outcome | Required separation in reporting |
| --- | --- |
| Binary version separation | Performance on all 300 true-pair passages |
| Direct swaps | 3.7→3.8 and 3.8→3.7 corrected versus introduced |
| Routing changes | Pair→other and other→pair transitions, stratified by true class |
| Overall benefit | Balanced-accuracy change across all 2,699 rows |

An overall gain caused by reducing the pair's maximum probability is a redistribution
benefit. It does not by itself establish better version identification. A positive
length-plus-style contrast supports additional information beyond this particular
length model; it does not establish length independence or causal stylistic traits.
Distinguish exploratory positive evidence, an interval spanning zero, and evidence
of deterioration. Binary results are not an additional engineering-eligibility gate.

**9. Fix the uncertainty calculation.** Use 10,000 paired whole-prompt bootstrap
resamples, seed 2026, with sorted original prompt IDs. Keep the recorded held-out
predictions fixed: do not refit, reselect, or search dictionaries inside these resamples.
Use the same draws for predictor differences and recompute the appropriate classwise
macro recall in each draw. Use percentile 2.5% and 97.5% bounds and the shared numerical
tie rule for differences close to zero. Do not substitute row bootstrap or fold standard
errors.

Length-band counts and point estimates are sufficient for the core report. Subgroup
bootstrap intervals are optional. If produced,
resample whole prompts before applying the fixed stratum mask, since paired responses
can occupy different bands. Record missing-class draws and the interval convention;
do not silently average over fewer classes. These optional reports never select a
model, setting, threshold, or eligibility outcome.

The intervals describe uncertainty in the recorded cross-validated predictions.
They do not fully capture overlapping training sets, retraining variability, or the
earlier hypothesis searches. Ten thousand resamples are not ten thousand independent
data groups. Do not present the intervals as selection-adjusted confirmation.

**10. Preserve the engineering-eligibility and stopping rules.** The prompt stage passes only when:

```text
unrounded candidate BA - unrounded incumbent BA >= 0.005
AND paired 95% interval lower bound > 0
```

The corresponding point threshold is 0.4750745712155108. Compute official 18-class
macro recall; do not convert net correct counts directly into balanced accuracy.
Failing this gate stops further eligibility checks with the incumbent retained.
Still report all completed binary diagnostics and the full-system outcome. Descriptive
binary success does not override the gate or authorize selecting another candidate model.

Only if the prompt gate passes, repeat the same complete procedure under the frozen
eight-topic mapping: rebuild every necessary incumbent prediction under those groups,
rerun all three C selections and the style-only a selection, and fit nothing across
held-out topic boundaries. Use the topic incumbent's exact 0.423027591349739 baseline.
The topic gate passes only with positive overall gain and nonnegative changes in at
least three of the five outer folds, using the shared tolerance rule. Report numbers
of positive, zero, and negative folds, all fold sizes, and selected a values.

Topic grouping changes partition sizes; do not impose the prompt-stage 240/60 or
120/120 row counts. The existing topic outer folds contain 738, 198, 432, 864, and
467 total rows and hold out 2, 1, 1, 2, and 2 topics respectively. If topic intervals
are reported, resample whole topics, not original prompts or rows, and acknowledge
that there are only eight groups. This is a robustness check on the same data, not
independent confirmation. Failure retains the incumbent. Passing both gates records
an exploratory candidate as eligible for a separate engineering decision; it is not
independent confirmation or automatic adoption.

**11. Keep adoption downstream of this study.** The study ends with its report and gate
decision. It does not automatically replace the incumbent, fit a final candidate,
package a model, or submit to Kaggle. Record any later engineering decision separately;
the incumbent remains in use pending that decision.

If a later decision selects the candidate after both gates pass, create and freeze the
full-data two-prompt-fold assignment using the existing
ntl.lora_final.final_plan(train, AdapterConfig(inner_folds=2, seed=2026)), which uses
seed 2031. Do not inherit the best outer fold's settings, take modal settings, choose
the best diagnostic model, or prefer settings from whichever grouping scored higher.

Repeat the same binary-C-then-18-class-a procedure on these two full-data selection
folds. Each binary split fits on 150 pair passages and validates on 150. Construct
the complete incumbent within each corresponding fitting partition using its own
unchanged training-only selection recipe. Refit the final style-only scaler and
classifier on all 300 pair passages at the selected C; retain the selected a and the
incumbent's unchanged final fitting procedure. If final a = 0, retain the incumbent
without claiming a deployed marker benefit. Length controls remain diagnostic.

Saving experimental predictions or a reproducibility artifact is not evidence of
confirmed generalization. An earlier CV gain of about 1.6 points appearing as about
0.8 on the public leaderboard does not establish a universal shrinkage factor or
that a 0.5-point change is undetectable there. Leaderboard sensitivity depends on
sample size, class composition, changed predictions, and distribution shift.
Leaderboard feedback does not choose settings in this study.

**12. Implement a small wrapper around tested components.** Use a small new module and
a fresh artifact directory, preserving the superseded rubric proposal and code as
historical records. Avoid duplicating the incumbent pipeline:

| Existing component | Reuse |
| --- | --- |
| gemini_pair.evaluate inner reconstruction | Extract or share a helper for inner fitting/validation rows, complete incumbent vectors, and selected s |
| claude_glm.redistribute | Existing pair-mass update, with gemini_pair.PAIR_INDEX |
| ensemble.prompt_bootstrap | Complete-system paired balanced-accuracy interval |
| selection.first_best | Existing 1e-12 tie rule with the declared option order |

Do not invoke gemini_pair.evaluate wholesale: its current selection orchestration
differs from this protocol. Reuse the reconstruction while keeping the declared
selection steps above explicit. This is a thin experiment wrapper, not a new framework.

Before fitting, retain focused checks for the extractor, partition/scaler isolation,
label and tie handling, selection order, convergence, probability/mass invariants,
and exact fallback vectors. Reuse existing tested checks where applicable. Tests use
synthetic inputs and fixed references; they do not tune the real-data specification.

Freeze the completed specification, source, environment, input manifest, and fold
files only after implementation checks pass and before the first real-data fit.
Store outputs under artifacts/gemini-marker-study/v2/ with these compact records:

| Record | Contents |
| --- | --- |
| manifest.json | Frozen settings, source/environment/input hashes, fold identities, compact construction-audit and reference/fallback results, and known/unknown marker-selection history |
| features.parquet | One table of IDs, W, counts, rates, zero-feature flags, and length bins |
| prompts/ | Inner scores and selected settings, convergence records, outer binary/full-system probabilities, primary metrics and paired intervals, and correction/regression counts |
| topics8/ | The same records under topic grouping, only if the first gate passes |
| decision.json and report.md | One report and decision record, including gate outcomes, distinct interpretations, deviations, limitations, and no automatic adoption |

Calibration plots, subgroup intervals, extensive descriptive tables, and detailed
coefficient/scaler exports are optional reports generated from saved records. They
are not prerequisites for completing the core evaluation. Any later final fitting
uses a separate run record rather than silently extending this study into deployment.

Do not overwrite outputs or revise the dictionaries, grids, thresholds, bins, candidate
eligibility, or gates after seeing the results. A genuine implementation failure aborts
the run; preserve its records and document any correction and invalidated results
before rerunning the unchanged scientific specification. It is not an opportunity to
search a different hypothesis.

Implementation and preflight precede the single declared exploratory evaluation.
This planning document changes no existing source or cached predictions and launches
no fits.

## Result (run October 28, 2026)

```sh
PYTHONPATH=src .venv/bin/python -m ntl.gemini_markers preflight ... --output artifacts/gemini-marker-study/v2
PYTHONPATH=src .venv/bin/python -m ntl.gemini_markers evaluate --output artifacts/gemini-marker-study/v2 --stage prompts
```

**Preflight** passed before any marker model was fit, and froze the specification in
`manifest.json`:
- The extractor reproduces the descriptive audit exactly: 32,672 words, means 117.13 and
  100.69, all-seven-zero 45/44, all-lexical-zero 73/65. The minimum W is 36.
- The locked incumbents score exactly 0.4700745712155108 (prompts) and 0.423027591349739
  (topics). The forced choice gets 179/300 right, and 18-class prediction 139/300.
- Rebuilt in every outer fold, the complete incumbent matches its lock exactly (maximum
  difference 0) with `s` = ¼, ¼, ½, ¼, ¼.
- **Cache audit.** The ten prompt inner caches were rebuilt by their recipe
  (`lora_compare.current_on_split`, which fits only on inner fitting rows) into a fresh
  folder. The rebuild is bit-identical: same file hashes and same blend weights.
- All 105 fits converged, within at most 11 iterations.

**Prompt gate: failed.** The study stops here, with the incumbent retained. The topic stage
was not run, and nothing was fitted for deployment.

| System | Prompt-grouped BA | Change | 95% interval |
| --- | ---: | ---: | --- |
| Complete incumbent | 0.4701 | — | — |
| With the style-only markers | 0.4682 | −0.19 | [−0.52, +0.19] |

**Fold results:**
- The fold changes are −0.19, −0.19, 0.00, −0.19 and −0.37 points.
- The selected `a` values are ½, ½, ¼, ¼ and ½. Every fold chose a nonzero `a`, by inner
  margins of 0.05–0.24 points.
- The candidate gains 8 passages and loses 13, all among the four Gemini classes: 3.7 and
  3.8 Flash each +3 / −6, 3.6 Flash +1 / −1, 3.1 Pro +1.

**Direct swaps:**
- 3.7 → 3.8: 47 → 46, with 3 corrected and 4 introduced.
- 3.8 → 3.7: 45 → 47, with 3 corrected and 5 introduced.

**Binary separation** on the 300 true-pair passages (outer holdouts):

| Predictor | BA | 3.7 recall | 3.8 recall | Brier |
| --- | ---: | ---: | ---: | ---: |
| Length-only | 56.3% | 54.7% | 58.0% | 0.247 |
| Style-only | 58.7% | 71.3% | 46.0% | 0.244 |
| Length + style | 59.3% | 61.3% | 57.3% | 0.238 |
| Incumbent, forced | 59.7% | 60.7% | 58.7% | 0.242 |
| Integrated system, forced | 58.3% | 59.3% | 57.3% | 0.234 |

**Contrasts:**
- **Principal contrast.** Length + style minus length-only is +3.0 points, interval
  [−2.3, +8.3]. That is exploratory and spans zero.
- **Style-only against the forced incumbent:** −1.0 points [−7.0, +5.0].

**Interpretation:**
- The markers carry little version signal beyond length, and none beyond what the
  incumbent already uses.
- Style-only leans heavily toward 3.7. Its 89 all-zero passages share one score.
- Integration lowers the Brier score slightly, but changes few decisions, and more of those
  for the worse.

Records are in `artifacts/gemini-marker-study/v2/`: `manifest.json`, `features.parquet`,
`prompts/`, `history/`, `decision.json` and `report.md`.

## Selection-history addendum (after the run)

**What was supplied.** After the run, a label-wise quantification of these marker lists over
all 300 pair passages was supplied (`history/label-wise-quantification.md`). It gives:
- counts by version and paired comparisons;
- weights from a standardized logistic regression fit on all 300 passages, with length
  relative to the prompt mean as the largest term.

**What it adds to the selection history:**
- The features' label-wise behaviour was examined on every passage, including every outer
  holdout.
- Participle openers were considered and dropped as uninformative.

**Check against the frozen extractor** (`history/quantification-check.py`; descriptive,
nothing fitted):
- **Reproduced exactly:** semicolons, colons, and the paired word-count comparison (3.7
  longer in 90 prompts, 3.8 in 60, no ties).
- **Not reproduced: the five word-list features.** Under complete-token matching:

  | Feature | Frozen extractor (3.7 / 3.8) | Supplied |
  | --- | --- | --- |
  | Kinetic verbs | 10 / 30 | 33 / 55 |
  | Systemic terms | 19 / 14 | 76 / 64 |
  | Sensory terms | 43 / 59 | 72 / 89 |
  | Hedges | 36 / 27 | 37 / 28 |
  | Formal connectors | 16 / 9 | 19 / 11 |

- **Broader matching does not reproduce them either.** Prefix, stem and substring matching
  were tried. The five listed systemic words never occur more than 19 / 15 times under any
  of them.
- **So the supplied table describes other word lists or matching rules** than those frozen
  and tested, and its weights do not apply to the tested features.

**Under the frozen definitions** the paired differences are:
- **Kinetic verbs:** higher for 3.8 in 21 prompts, for 3.7 in 4 (sign test p = 0.001).
- **Sensory terms:** 30 against 15 (p = 0.04).
- **The other features:** none differs at p < 0.15.

These counts are descriptive and were examined after the run.

**The study's result is unchanged.** It evaluated the frozen extractor, and its prompt gate
failed. The suggested score also cannot be used as given:
- its relative-length term needs prompt groups, which the test file lacks;
- weights fit on all 300 passages have no held-out estimate on this data.
