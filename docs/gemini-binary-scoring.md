# Gemini 3.7 vs. 3.8: Simple Scoring Experiment

The plan below is recorded as supplied, before any model was fit. It runs after the
[marker study](gemini-marker-study.md), whose style-only model used the same seven features
with C chosen per fold. This fixed-C variant is therefore exploratory on passages that have
already been examined.

Train one logistic regression model to score each passage from **0 (closer to Gemini 3.7 Flash)** to **1 (closer to Gemini 3.8 Flash)**.

## Features

Use these seven features as counts per 100 words:

| Feature | Exact markers |
| --- | --- |
| Action-word markers | shatter, ignite, collapse, bloom, demands, pierce, erupt |
| Qualifying expressions | merely, essentially, partially, relatively, not simply |
| Formal connectors | wherein, whereby, thus, consequently, yielding |
| Systemic terminology | paradigm, framework, syntax, subroutines, equilibrium |
| Sensory terminology | void, abyss, boiling, froth, molten, violent, tears |
| Semicolons | `;` |
| Colons | `:` |

Match whole tokens case-insensitively, without stemming. Count repeated occurrences. Count “not simply” once when the words are separated only by whitespace. Count literal ASCII punctuation everywhere.

Use the earlier audit's word-token pattern:

```regex
\b\w+(?:['’\-]\w+)*\b
```

For each feature, calculate `100 × count / word_count`. Reject empty passages. Keep length out of this first model.

## Model and Score

- Standardize features using training data only.
- Fit logistic regression with L2 regularization and fixed `C = 0.1`.
- Learn the weights and intercept from the data; do not hard-code the proposed heuristic weights.
- Encode Gemini 3.7 as `0` and Gemini 3.8 as `1`.

The score is the model's predicted probability of Gemini 3.8, conditional on authorship belonging to this pair:

```text
score = sigmoid(intercept + weighted sum of standardized features)
```

- Above `0.5`: predict Gemini 3.8.
- At or below `0.5`: predict Gemini 3.7.
- Near `0.5`: little preference between the versions.

## Evaluation

Use all **300 passages** and the existing **five prompt-grouped folds**. Keep both responses to each prompt together. In each fold, train on 240 passages and predict the remaining 60; fit the scaler only on the training partition.

Compare the combined held-out predictions against the incumbent's **59.67% binary accuracy (179/300)**.

Report:

- Binary accuracy and recall for each version.
- Existing errors corrected versus correct predictions lost.
- Saved held-out scores and fitted weights.

Freeze the features, regularization, and threshold before fitting. Treat the result as exploratory because these passages helped generate the feature hypotheses.

## Implementation choices the plan leaves open

These were fixed before fitting:
- **Extractor.** `ntl.gemini_markers.extract`, the marker study's frozen implementation of
  exactly these features, is used unchanged.
- **Solver.** lbfgs, at most 2,000 iterations, tolerance 1e-6, seed 2026, no class
  weighting. A convergence warning or any nonfinite value fails the run.
- **Folds.** The outer folds of `artifacts/lora/plan.csv`.
- **Incumbent.** The incumbent's forced choice is its locked out-of-fold probabilities
  (`artifacts/styledistance-adapt/reference-prompts.npz`), predicting 3.8 when
  p₃.₈ / (p₃.₇ + p₃.₈) > 0.5.
- **Weights.** Weights are saved for each fold's model. A fit on all 300 passages is also
  saved for reference. It has no held-out estimate.
- **Supplementary.** As in earlier records, a paired whole-prompt bootstrap (10,000
  resamples, seed 2026) of the accuracy change is reported. It is not a gate.

## Result (run October 28, 2026)

```sh
PYTHONPATH=src .venv/bin/python -m ntl.gemini_binary freeze --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv \
  --reference artifacts/styledistance-adapt/reference-prompts.json \
  --output artifacts/gemini-binary-scoring/v1
PYTHONPATH=src .venv/bin/python -m ntl.gemini_binary evaluate --output artifacts/gemini-binary-scoring/v1
```

**Before fitting:**
- The freeze recorded 300 passages, none empty (minimum 40 words).
- Its feature table is identical to the marker study's for these passages.
- Every fold trained on 240 passages from 120 prompts and held out 60.
- All six fits (five folds and the all-300 reference fit) converged in 7–9 iterations.

| Predictor | Correct | Accuracy | 3.7 recall | 3.8 recall |
| --- | ---: | ---: | ---: | ---: |
| Scoring model, held out | 175 / 300 | 58.3% | 71.3% | 45.3% |
| Incumbent, forced choice | 179 / 300 | 59.7% | 60.7% | 58.7% |

**Against the incumbent's forced choice:**
- **Change: −1.3 points.** The paired prompt-bootstrap interval is [−7.3, +4.7], and 68%
  of resamples are at or below zero.
- **Errors corrected and correct predictions lost: 60 and 64.** Both are right on 115
  passages and both wrong on 61.
- **By fold**, correct out of 60 (model against incumbent): 35–32, 35–38, 36–38, 33–36 and
  36–35.

**Fitted weights.** These are standardized; positive favours 3.8.

| Term | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | All 300 (reference) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Intercept | +0.012 | +0.024 | +0.020 | +0.024 | +0.022 | +0.024 |
| Action words | +0.418 | +0.457 | +0.420 | +0.507 | +0.454 | +0.475 |
| Semicolons | +0.183 | +0.189 | +0.243 | +0.144 | +0.191 | +0.198 |
| Sensory terms | +0.145 | +0.122 | +0.141 | +0.082 | +0.122 | +0.129 |
| Qualifiers | −0.136 | −0.038 | −0.162 | −0.193 | −0.053 | −0.119 |
| Formal connectors | −0.120 | −0.040 | −0.076 | +0.030 | −0.118 | −0.069 |
| Systemic terms | −0.049 | −0.045 | +0.011 | −0.112 | −0.105 | −0.060 |
| Colons | +0.030 | −0.092 | +0.039 | +0.051 | +0.066 | +0.022 |

**Interpretation:**
- **No improvement.** The scoring model does not beat the incumbent's forced choice; it is
  four passages lower, well within noise. It disagrees with the incumbent on 124 of 300
  passages, and trades corrections for losses almost one for one.
- **Weak preferences.** Half the scores lie between 0.45 and 0.55.
- **The model leans to 3.7.** It predicts 3.8 for only 111 passages. The 89 passages with no
  marker at all share one score per fold and are all called 3.7; half of them are right.
- **The weights are stable across folds:**
  - action words, semicolons and sensory terms favour 3.8;
  - qualifiers favour 3.7;
  - formal connectors, systemic terms and colons are small or change sign.
- **Agreement with the earlier heuristic.** These directions match the heuristic weights
  supplied earlier, although that heuristic's counts did not reproduce.

**Consistency.** The result matches the marker study: its style-only model, with `C` chosen
per fold, got 176 of 300.

**A post-hoc observation, not a result.** On the 211 passages with at least one marker, the
model is right on 130 and the incumbent on 125. A rule that uses the model only on such
passages was not declared, and has not been evaluated.

**Records** in `artifacts/gemini-binary-scoring/v1/`:
- `manifest.json`, written before fitting;
- `features.parquet`;
- `scores.parquet`: held-out scores, predictions and the incumbent's forced choice;
- `weights.json`: each fold's weights and scaler, and the all-300 reference fit;
- `comparison.json`.
