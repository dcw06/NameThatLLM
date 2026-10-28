# 18-class Qwen + StyleDistance head

**Question.** StyleDistance helped the Claude–GLM specialist; the Qwen-only specialist did
not. Does the same signal also help the other classes? The largest confusions left after
the specialist are within GPT-5.6 (Luna 36.7%, Sol 31.3% and Terra 34.7% recall) and
among the Gemini Flash versions.

**References to beat:**
- 47.01% grouped CV (the complete incumbent: blend plus Claude–GLM specialist);
- 42.30% on the eight-topic split;
- 0.429 on the public leaderboard.

## Design

This section was written before any evaluation ran. It reuses the cached embeddings; no
feature is extracted.

**Head.** The specialist's classifier, trained on all 18 classes of each training
partition: standardization, the StyleDistance block scaled by √(8192/768) to the Qwen
block's total variance, then logistic regression with `C = 0.005`.

**Integration.** The head's distribution `r` is blended with the complete incumbent's:
`p = (1 − b) p_incumbent + b r`. `b = 0` is the incumbent exactly.

**Selection.** For each of the original five prompt folds:
1. The inner folds choose the specialist's `s` exactly as the incumbent does. The
   incumbent is refit within each inner training partition: the blend with its own nested
   weight, then the specialist at that `s`.
2. The same inner folds then choose `b` from `{0, ¼, ½, ¾, 1}`, scored on all 18 classes.
   Ties keep the smaller `b`.

The inner incumbent's `s` is chosen on the same inner folds that then choose `b`, as in the
added-encoder experiment. That favours the incumbent slightly, so it can only make a
nonzero `b` harder to select.

**Control (diagnostic).** A Qwen-only 18-class head is selected and blended the same way.
It is the blend's own Qwen head, so it separates StyleDistance from simply giving the Qwen
head more weight. It takes no part in promotion.

**Evaluation.** All 2,699 passages on the original five prompt folds
(`artifacts/lora/plan.csv`), compared with the incumbent by a paired prompt bootstrap
(10,000 resamples, seed 2026). The `b = 0` path must reproduce the incumbent's 47.01%
predictions exactly.

**Where the change lands.** Reported for the candidate against the incumbent:
- recall for all 18 classes, and passages gained and lost by class;
- the Luna/Sol/Terra confusion matrix, and the share of their passages predicted as
  another of the three;
- the share of the overall change contributed by Luna, Sol and Terra (each class
  contributes its recall change divided by 18).

**Promotion rule.** The candidate is promoted only if:
- its gain is at least +0.5 points overall;
- its 95% paired prompt-bootstrap interval lies above zero.

Only then does it repeat the selection on the eight-topic split. It passes there if,
against 42.30%, its gain is positive overall and not negative in most held-out folds.
A candidate that passes both would be benchmarked offline on 4,000 passages, frozen and
submitted once against 0.429.

## Result: original prompt folds

```sh
PYTHONPATH=src .venv/bin/python -m ntl.styledistance_head --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv \
  --qwen-features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --styledistance-features artifacts/claude-glm/styledistance-train.npz \
  --current-oof artifacts/qwen-blend/oof.parquet \
  --current-inner artifacts/claude-glm/current-inner/fold{0,1,2,3,4}.parquet \
  --output-dir artifacts/styledistance-head/prompts
```

The `b = 0` path reproduces the incumbent's 47.01% predictions exactly, on all 2,699
passages, and the specialist's `s` by fold (¼, ¼, ½, ¼, ¼) is the incumbent's.

| System | Grouped balanced accuracy | Change | 95% interval | `b` by fold |
| --- | ---: | ---: | --- | --- |
| Incumbent: blend + Claude–GLM specialist | 0.4701 | — | — | — |
| + Qwen + StyleDistance head (candidate) | 0.4690 | −0.11 | [−0.67, +0.41] | 0, 0, ¼, 0, ¼ |
| + Qwen-only head (control) | 0.4701 | 0.00 | — | 0, 0, 0, 0, 0 |

The candidate's resamples put the change at or below zero 63% of the time. By fold the
change is 0.0, 0.0, −0.9, 0.0 and +0.4 points. Only two folds chose the head: fold 4,
whose inner folds preferred `b = ¼` by 0.83 points, and fold 2, by 0.05.

**The control is never selected.** In every outer fold, the inner mean scores the Qwen-only
head below the incumbent at every `b > 0`. More weight on the Qwen head already in the
blend does not help.

**No fixed `b` would have passed even with hindsight.** As a diagnostic, never a
selection, each `b` was also applied in every fold:

| `b` | ¼ | ½ | ¾ | 1 |
| --- | ---: | ---: | ---: | ---: |
| Candidate head | 0.4723 (+0.22) | 0.4642 | 0.4557 | 0.4516 |
| Control head | 0.4578 | 0.4475 | 0.4445 | 0.4386 |

### Where the change lands

**The gain does concentrate on GPT-5.6, but only on Terra, and the other classes lose
more.** Terra contributes +0.22 points; Luna and Sol are unchanged. The other 15 classes
contribute −0.33 points: they gain 24 passages and lose 33.

| Class | Incumbent recall | Candidate recall | Gained / lost |
| --- | ---: | ---: | --- |
| GPT-5.6 Luna | 36.7% | 36.7% | +3 / −3 |
| GPT-5.6 Sol | 31.3% | 31.3% | +3 / −3 |
| GPT-5.6 Terra | 34.7% | 38.7% | +9 / −3 |
| Largest moves elsewhere | | | |
| Qwen 3.8 Max | 51.3% | 48.0% | +2 / −7 |
| Gemini 3.7 Flash | 45.3% | 43.3% | +1 / −4 |
| Gemini 3.8 Flash | 47.3% | 46.0% | +2 / −4 |
| Claude Sonnet 5 | 51.3% | 53.3% | +3 / −0 |

**The head barely untangles Luna, Sol and Terra.** Passages predicted as another of the
three fall from 140 to 134 (31.1% to 29.8% of the family):

| True \ predicted | Luna | Sol | Terra | Outside the three |
| --- | ---: | ---: | ---: | ---: |
| Luna | 55 → 55 | 24 → 23 | 19 → 17 | 52 → 55 |
| Sol | 28 → 27 | 47 → 47 | 23 → 23 | 52 → 53 |
| Terra | 21 → 20 | 25 → 24 | 52 → 58 | 52 → 48 |

Terra's nine gained passages had been predicted as GPT-5.5 (3), Luna (2), Sol (2), Kimi
and Astra; its three lost ones go to Luna, Sol and GLM 5.3.

### StyleDistance in the head alone

On its own, the 18-class Qwen + StyleDistance head beats the Qwen-only head: 0.4516
against 0.4386, +1.30 points (95% interval [−0.04, +2.67]). The Qwen-only head reproduces
the recorded frozen Qwen score (0.4386). Most of the gain is on the Claude and GLM classes
(Opus 4.6 +5.3, GLM Flash +6.0, Fable +3.4, Sonnet +3.3, GLM +3.3 points of recall),
which the specialist already covers. Terra gains +4.7 and GPT-5.5 +2.7, but the Gemini
versions lose: 3.7 Flash −4.0, 3.6 Flash −2.7, 3.1 Pro −2.0.

So StyleDistance adds to Qwen in an 18-class head too, though that interval just reaches
below zero, and most of what it adds is on the classes the specialist already covers.
What the incumbent does not already have is mostly a little more on Terra, and it comes
with losses elsewhere.

### Review correction

An independent review found that the first run broke the declared tie rule. In fold 3,
`b = 0` and `b = ¼` tie exactly on the inner folds: each inner fold has 60 passages per
class, and both weights get 902 of 2,160 balanced-accuracy points in total. Their float
means differ in the last bit (5.6 × 10⁻¹⁷), and the larger `b` won. That run reported
+0.22 points ([−0.44, +0.89]); it is kept at
`artifacts/styledistance-head/prompts-v1-float-tie`.

Selection now treats scores within 10⁻¹² as tied (`ensemble.first_best`). Distinct
balanced accuracies on these folds differ by far more than that. The corrected run keeps
fold 3 at the incumbent; the decision does not change.

The tests now also:
- recompute every inner score independently from the inner training rows;
- check the tie rule, including a float-split tie;
- check the paired statistic, the gained and lost counts and the Luna/Sol/Terra matrix
  against the saved predictions.

All 11 code mutations the review used to probe the tests now fail them.

**Every earlier selection was audited for the same defect.** Every nested selection site
used the same exact float comparison, so all of them now share the tolerant rule
(`ntl.selection.first_best`). An audit then checked 154 recorded selections:
- the blend weight `w`, including the 22 inner refits whose scores were never saved,
  which it recomputed;
- the specialist's `s`, on prompts, topics and the final recipe;
- the added-encoder options;
- QLoRA's settings and weights;
- the CPU and frozen-embedding Gemini experiments.

Six held exact ties, and only one of those had been broken by rounding: fold 3 of the
superseded run-2 QLoRA control (corrected in [the QLoRA record](lora.md)). Nothing behind
the submitted system came close to a tie. The smallest margin in its validation and final
selections is 6.4 × 10⁻⁵, so the change leaves every choice behind 0.429 unchanged.

### Decision

**Not promoted.** The change is −0.11 points, against a required +0.5 with an interval
above zero. As declared:
- the eight-topic check was not run;
- nothing was packaged or submitted.

The blend plus Qwen + StyleDistance specialist (47.01% CV, public 0.429) stays the
submitted system. StyleDistance in a full 18-class head adds a little on Terra, which the
other classes give back.
