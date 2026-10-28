# Claude–GLM specialist with frozen StyleDistance features

**Question.** Can a specialist recover the current system's weakest classes? The weakest
are Claude Fable 5 (12% recall), GLM 5.3 (24%) and GLM 5.3 Flash (25%). Fable is often
predicted as a GLM, and the GLMs as Claude models. The experiment therefore covers all
five Claude models and both GLMs, seven classes in total.

## Design

The design was fixed before any result was seen.

**Two specialist heads.** Both use the frozen Qwen recipe's classifier: standardization,
then logistic regression with `C = 0.005`. Each is trained only on the seven classes' rows
of each training partition.
- **Control:** the existing frozen Qwen3-8B-Base features (8,192 dimensions).
- **Candidate:** Qwen plus frozen StyleDistance features (768 dimensions). After
  standardization, the StyleDistance block is scaled by √(8192/768), so that it carries the
  same total variance as the Qwen block. The Qwen block is treated exactly as in the
  control.

Comparing the two separates the effect of specialization from that of the added features.

**Integration.** The specialist redistributes the current system's probability mass on the
seven classes: `p' = (1 − s) p + s M q`. Here `M` is the current mass on the seven, and `q`
is the specialist's distribution. The other 11 classes keep their probabilities, and
`s = 0` is the current system exactly.

**Selection.** The inner prompt folds of each outer training partition choose `s` from
`{0, ¼, ½, ¾, 1}`, scored on all 18 classes. Ties go to the smaller weight. On those inner
folds the current system is refit with its own nested weight. The folds are those of the
QLoRA experiment (`artifacts/lora/plan.csv`).

**Evaluation.** All 2,699 passages are scored on the original five prompt folds, with
paired prompt-bootstrap intervals (10,000 resamples). The candidate passes only if its
95% interval against the current system (45.41%) lies above zero.

**StyleDistance features.** The pinned revision `b7df5f0` uses masked-mean pooling,
L2 normalization and 512 tokens; no passage is truncated. The features were extracted
locally on CPU for all 2,699 passages. On the 600 Gemini passages extracted earlier on a
Kaggle T4, they match within 1.7 × 10⁻⁶ (minimum cosine 0.9999997).

```sh
PYTHONPATH=src .venv/bin/python -m ntl.claude_glm current-inner --outer-fold K ...   # K = 0..4
PYTHONPATH=src .venv/bin/python -m ntl.claude_glm evaluate --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv \
  --qwen-features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --styledistance-features artifacts/claude-glm/styledistance-train.npz \
  --current-oof artifacts/qwen-blend/oof.parquet \
  --current-inner artifacts/claude-glm/current-inner/fold{0,1,2,3,4}.parquet \
  --output-dir artifacts/claude-glm/prompts
```

## Result: original prompt folds

| System | Grouped balanced accuracy | Change from current | 95% interval | `s` by fold |
| --- | ---: | ---: | --- | --- |
| Current CPU + frozen Qwen | 0.4541 | — | — | — |
| + Qwen-only specialist (control) | 0.4556 | +0.15 | [−0.52, +0.81] | ½, 0, ½, ¼, ¼ |
| + Qwen + StyleDistance specialist | **0.4701** | **+1.60** | **[+0.78, +2.41]** | ¼, ¼, ½, ¼, ¼ |

**The gain comes from the StyleDistance features, not from specialization.**
- The candidate is +1.45 points over the control (95% interval [+0.74, +2.11]).
- The control alone is no gain over the current system.

**Consistency.** The candidate improves every fold:

| Fold | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | .476 | .426 | .487 | .430 | .452 |
| Candidate | .498 | .450 | .498 | .439 | .465 |

**Seven-way separation.** On the true Claude and GLM passages, balanced accuracy over the
seven classes is:

| System | Seven-class balanced accuracy |
| --- | ---: |
| Current system | 0.426 |
| Control | 0.440 |
| Candidate | 0.462 |

**By class.** The candidate gains 82 passages and loses 39.

| Class | Current recall | Candidate recall | Gained / lost |
| --- | ---: | ---: | --- |
| Claude Fable 5 | 12.1% | 20.1% | +12 / −0 |
| Claude Opus 4.6 | 60.0% | 65.3% | +15 / −7 |
| Claude Opus 4.8 | 37.3% | 39.3% | +7 / −4 |
| Claude Opus 5 | 72.0% | 76.7% | +11 / −4 |
| Claude Sonnet 5 | 46.7% | 51.3% | +11 / −4 |
| GLM 5.3 | 24.0% | 26.7% | +9 / −5 |
| GLM 5.3 Flash | 25.3% | 32.0% | +15 / −5 |
| The other 11 classes | — | at most −1.3 points each | +2 / −10 |

**Effects outside the seven classes.**
- The stronger Claude classes do not regress.
- Outside the seven classes, a few passages are lost when the extra Claude probability
  overtakes a correct prediction.
- The fits produce no convergence warnings, and a rerun reproduces the result exactly.

## Robustness: eight held-out subject areas

The same evaluation was rerun with whole topics held out, using the topic plan
(`artifacts/lora/plan-topics8.csv`) and the topic-grouped current system
(`artifacts/topic-cv/topics-8/blend/oof.parquet`). `--groups` regroups the passages by
topic, as `ntl.topic_cv` did. The current system's inner weight choice, the specialist's
inner folds and the bootstrap then all hold out and resample whole topics.

The pass rule was fixed before running: the gain must stay positive and must not reverse in
most folds. With only 8 topics, the interval is reported but not required to exclude zero.

| System | Balanced accuracy | Change from current | 95% interval (8 topics) |
| --- | ---: | ---: | --- |
| Current, topic-grouped | 0.4156 | — | — |
| + Qwen-only specialist | 0.4156 | 0.00 | [−0.42, +0.48] |
| + Qwen + StyleDistance specialist | **0.4230** | **+0.74** | [−0.05, +1.70] |

- **No fold gets worse.** By fold the change is +0.2, 0.0, +3.2, +0.5 and 0.0 points.
- **Six of the seven classes gain; Opus 4.8 is unchanged.**
- **The candidate still beats the control:** +0.74 points (95% interval [+0.24, +1.28]).
- **The gain is about half the prompt-fold gain.** That fits a style signal that partly
  depends on topic.

The robustness check passes.

## Final recipe

The final `s` is chosen once, on inner prompt folds over all 2,699 rows
(`artifacts/lora/final_plan.csv`). On each half the current system is refit with its own
nested weight; it chose 0.25 and 0.5. Both halves choose **`s = 0.25`**, as did four of
the five outer folds.

The recipe (`artifacts/claude-glm/recipe.json`) records:
- the specialist classes and the feature set;
- both model revisions;
- the identity of the current system's recipe;
- both validations.

**Fit and prediction refuse:**
- a different current system;
- different training rows;
- a fitted specialist that would be overwritten.

## Offline submission notebook

`scripts/prepare_specialist_submission.py` builds two notebooks:
- **A model-input notebook** (CPU, Internet on). It saves the pinned StyleDistance
  snapshot with its manifest. Its Kaggle run produced the same 14 files and sizes as the
  local snapshot.
- **The submission notebook** (T4, Internet off). It has three inputs: the Qwen snapshot,
  the StyleDistance snapshot and the neural wheels. It runs three steps:
  1. `ntl.ensemble fit`, as for the submitted blend;
  2. `ntl.claude_glm fit`, using the blend fit's own Qwen training features plus
     StyleDistance;
  3. `ntl.claude_glm predict`, which keeps the blend's probabilities, redistributes them
     and writes `submission.parquet`.

**Local preflight.** The notebook's cells were run locally against a fake `/kaggle` tree.
Cached features stood in for Qwen; StyleDistance used the real snapshot on CPU.
- Every cell completes.
- The specialist is fit on 1,049 Claude and GLM passages.
- The 4,000-row rehearsal writes a valid submission.

**Kaggle benchmark (T4, Internet off, version 1).**

| Step | Time |
| --- | ---: |
| Blend refit, including Qwen features for 2,699 passages | 632 s |
| Specialist fit, including StyleDistance on the T4 (109 s) | 134 s |
| Visible test scored | 807 s after the start |
| 4,000-row rehearsal predict | 771 s |
| Whole run including the fit | 1,579 s |

- The run is far inside Kaggle's 9-hour limit.
- The Qwen training features extracted on Kaggle are bit-identical to the archive the
  validation used (hash `a0d8ad8c…`).
- The rehearsal writes a valid 4,000-row submission.

**Review fix.** A review of the final path found that a fitted head saved by
`python -m ntl.claude_glm` referenced its block-weight function as `__main__`. Only that
same command could load it. The notebook's predictions were unaffected, because fit and
predict both run that way. The module's command line now runs the imported module, so the
head loads from any entry point. A refit with the fix has identical coefficients.

Version 2 carries the fix and is the version submitted. Its output is identical to
version 1's, and its run took 1,401 s.

## Submission result

| System | Grouped CV | Eight-topic CV | Public leaderboard |
| --- | ---: | ---: | ---: |
| CPU baseline (57676158) | 0.3597 | 0.337 | 0.360 |
| Current CPU + frozen Qwen blend (57685450) | 0.4541 | 0.4156 | 0.421 |
| + Claude–GLM specialist (57721458) | **0.4701** | **0.4230** | **0.429** |

**The specialist adds +0.8 public points.** That is half its prompt-fold gain (+1.60), and
about the gain it kept with whole subject areas held out (+0.74). The topic-held-out
estimate therefore predicted this leaderboard change better than the prompt-fold estimate.

This fits a hidden test whose topics differ from the training prompts. It is a single
public-subset score, though, so it does not settle the question. The specialist's
CV-to-leaderboard gap (−4.1 points) is a little wider than the blend's (−3.3).
