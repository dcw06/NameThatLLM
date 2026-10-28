# Gemini 3.7 / 3.8 Flash pair specialist

**Question.** Can a classifier trained only on the Gemini 3.7 and 3.8 Flash passages recover
the complete incumbent's errors between the two? The incumbent recalls 45.3% of 3.7 and
47.3% of 3.8 Flash. It calls 47 true 3.7 passages 3.8 and 45 true 3.8 passages 3.7.

**References to beat:**
- 47.01% grouped CV (the complete incumbent: blend plus Claude–GLM specialist);
- 42.30% on the eight-topic split;
- 0.429 on the public leaderboard.

## Design

This section was written before any evaluation ran. It uses existing features and CPU
fitting only.

**Data.** All 300 passages whose true label is Gemini 3.7 or 3.8 Flash, whatever the
incumbent predicted. Every one of the 150 prompts has one passage of each, so every
partition is balanced. The original five prompt folds (`artifacts/lora/plan.csv`) give
240 training and 60 validation passages per outer fold, and 120 / 120 on each inner split.

**Two binary specialists.** Every vocabulary and scaler is fit on training rows only, and
the feature definitions are fixed.

| Configuration | Features | `C` grid |
| --- | --- | --- |
| CPU | The CPU baseline's character and word TF-IDF and style features, with its class-weighted logistic regression | 0.1, 1, 4 |
| Qwen | Cached frozen Qwen3-8B-Base features, standardized, with the frozen recipe's logistic regression | 0.005, 0.05, 0.5 |

**Integration.** With `M = p₃.₇ + p₃.₈` the incumbent's mass on the pair and `q` the
specialist's distribution, `p′ₖ = (1 − a) pₖ + a M qₖ` for the two versions. The other 16
probabilities are unchanged, and `a = 0` is the incumbent exactly. The specialist scores
every passage, so it only moves passages on which the incumbent already puts mass on the pair.

**Selection.** For each of the original five prompt folds:
1. The inner folds choose the Claude–GLM specialist's `s` as the incumbent does. The
   incumbent is refit within each inner training partition: the blend with its own nested
   weight, then that specialist at `s`.
2. The same inner folds then choose one option from 25: the unchanged incumbent, or a
   configuration and `C` at `a ∈ {¼, ½, ¾, 1}`. Options are scored by mean inner balanced
   accuracy over all 18 classes.
3. Ties use the shared tolerance rule (`ntl.selection.first_best`), in this order:
   - the unchanged incumbent;
   - then the smaller `a`;
   - then CPU before Qwen;
   - then the smaller `C`.
4. The locked choice is fit on the outer training partition and scored once on the
   outer fold.

## Reports

**Diagnostic.** These are not the competition result.
- Binary balanced accuracy on the 300 true-pair passages, for:
  - the specialist;
  - the incumbent forced to choose between 3.7 and 3.8;
  - the candidate system forced the same way.
- In a fold that keeps the incumbent, the specialist reported is its inner-best
  configuration and `C`. Each fixed configuration and `C` is also reported over all folds.

**Competition.**
- Overall 18-class balanced accuracy, with a paired prompt bootstrap against the
  incumbent (10,000 resamples, seed 2026).
- End-to-end recall for each Gemini version.
- Direct 3.7↔3.8 swaps corrected and introduced.
- Passages gained and lost, by class.
- As a hindsight diagnostic only, each option applied in every fold.

**Promotion rule.** The candidate is promoted only if:
- its gain is at least +0.5 points overall (about 14 more correct Gemini passages, if no
  other class changes);
- its 95% interval lies above zero.

Only then does it repeat the selection on the eight-topic split, with topic groups used
throughout selection and the bootstrap. It passes there if, against 42.30%, its gain is
positive overall and not negative in most held-out folds. Only a candidate that passes
both is packaged and benchmarked.

## Result: original prompt folds

```sh
PYTHONPATH=src .venv/bin/python -m ntl.gemini_pair --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv \
  --qwen-features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --styledistance-features artifacts/claude-glm/styledistance-train.npz \
  --current-oof artifacts/qwen-blend/oof.parquet \
  --current-inner artifacts/claude-glm/current-inner/fold{0,1,2,3,4}.parquet \
  --output-dir artifacts/gemini-pair/prompts
```

The incumbent option reproduces the incumbent's 47.01% predictions exactly, on all 2,699
passages, and the specialist's `s` by fold (¼, ¼, ½, ¼, ¼) is the incumbent's.

### Competition result

| System | Grouped balanced accuracy | Change | 95% interval |
| --- | ---: | ---: | --- |
| Incumbent: blend + Claude–GLM specialist | 0.4701 | — | — |
| + Gemini pair specialist | 0.4701 | 0.00 | [−0.37, +0.41] |

The resamples put the change at or below zero 52% of the time. The candidate gains 14
passages and loses 14.

**Selection.** The inner folds chose a CPU specialist in every fold, by small margins over
the incumbent. Those margins did not carry over to the held-out folds:

| Fold | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Selected | CPU, `C` 0.1, `a` ¼ | CPU, 1, ¼ | CPU, 4, ¾ | CPU, 0.1, ¾ | CPU, 4, ¾ |
| Inner margin (points) | +0.09 | +0.19 | +0.23 | +0.05 | +0.51 |
| Held-out change (points) | 0.0 | +0.19 | 0.0 | +0.56 | −0.74 |

**No option would have helped even with hindsight.** As a diagnostic, never a selection,
each option was also applied in every fold. None beats the incumbent. The best are CPU
(`C` 0.1, `a` ¼) and Qwen (`C` 0.5, `a` ½), both at −0.04 points, and losses mostly grow
with `a`:

| `a` | CPU, `C` 0.1 / 1 / 4 | Qwen, `C` 0.005 / 0.05 / 0.5 |
| --- | --- | --- |
| ¼ | −0.04 / −0.11 / −0.15 | −0.19 / −0.30 / −0.30 |
| ½ | −0.19 / −0.15 / −0.15 | −0.41 / −0.11 / −0.04 |
| ¾ | −0.19 / −0.22 / −0.41 | −0.37 / −0.15 / −0.37 |
| 1 | −0.33 / −0.44 / −0.52 | −0.33 / −0.44 / −0.78 |

**Gemini recall** (incumbent → candidate):

| Version | Recall | Gained / lost |
| --- | --- | --- |
| 3.1 Pro | 80.0% → 81.3% | +2 / −0 |
| 3.6 Flash | 49.3% → 51.3% | +3 / −0 |
| 3.7 Flash | 45.3% → 44.0% | +3 / −5 |
| 3.8 Flash | 47.3% → 44.7% | +5 / −9 |

The only other change is one Claude Opus 4.8 passage gained.

**Direct swaps.** The candidate corrects 7 of the incumbent's 92 direct 3.7↔3.8 swaps and
introduces 9:

| True → predicted | Incumbent | Candidate | Corrected | Introduced |
| --- | ---: | ---: | ---: | ---: |
| 3.7 → 3.8 | 47 | 48 | 2 | 2 |
| 3.8 → 3.7 | 45 | 46 | 5 | 7 |

(Counts also change when a passage moves between the other pair version and a third
class.)

**The gains outside the pair do not come from telling the versions apart.** The six
gained 3.1 Pro, 3.6 Flash and Opus 4.8 passages were ones the incumbent called 3.7 or 3.8.
On each of them the specialist splits the pair's mass more evenly than the incumbent
did (for example 0.48 against 0.26 for 3.8), so the pair's top probability falls below
the true class. Of the 14 pair passages lost, 9 go to the other pair version, 4 to 3.6
Flash and 1 to 3.1 Pro.

### Diagnostic: binary separation

Balanced accuracy between 3.7 and 3.8 on the 300 true-pair passages:

| Predictor | Binary balanced accuracy |
| --- | ---: |
| Incumbent, forced to choose between the two | 59.7% |
| Selected specialist (CPU in every fold) | 56.0% |
| Candidate system, forced | 57.7% |
| CPU, `C` = 0.1 / 1 / 4 (every fold) | 54.0% / 54.3% / 56.0% |
| Qwen, `C` = 0.005 / 0.05 / 0.5 (every fold) | 60.0% / 61.3% / 61.0% |

**Pair-specific training does not separate the versions better than the 18-class
incumbent already does.** The CPU specialists are worse than the incumbent's forced
choice. The best Qwen specialist is 1.7 points better, 5 passages, which is within
noise.

The specialists also make mostly the same mistakes as the incumbent:

| Specialist | Both right | Incumbent only | Specialist only | Both wrong | Agree |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen, `C` 0.05 | 155 | 24 | 29 | 92 | 82% |
| CPU, `C` 4 | 143 | 36 | 25 | 96 | 80% |

So there is little complementary signal for the integration to exploit. The inner folds
chose CPU even though it separates the pair worse; with so little signal, their 18-class
margins (0.05–0.51 points) reflect noise.

### Review

An independent review recomputed the run, including every inner score, specialist
probability and prediction, and found no defect in the module. No vocabulary or scaler
sees held-out rows, and no choice was decided by a float-split tie. It found two gaps in
the tests, now closed:
- The synthetic texts let every CPU specialist separate the pair perfectly, so the tests
  could not tell the configurations apart. Both versions now share one text marker.
- The selection was checked only against the module's own mean scores; the means are now
  checked against the inner scores.

All 20 code mutations used to probe the tests now fail them, including outer-fold
leakage into the specialist and a candidate that ignores the chosen `a`.

### Decision

**Not promoted.** The change is 0.00 points, against a required +0.5 with an interval
above zero. As declared:
- the eight-topic check was not run;
- nothing was packaged or submitted.

The blend plus Claude–GLM specialist (47.01% CV, public 0.429) stays the submitted system.

**What this answers.** Training on the pair alone, with these features (240 passages per
fold), does not recover the incumbent's Gemini 3.7/3.8 errors. The two versions are
barely separable by either the TF-IDF and style features or frozen Qwen features: about
60% between two classes, where 50% is chance. The specialists and the incumbent fail
on largely the same passages (about 90 of the 300). That points to the limited version
signal in these features, rather than the training set, as the constraint.
