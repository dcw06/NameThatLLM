# GLM 5.3 against GLM 5.3 Flash, with matched external passages

**Question.** Do matched external passages teach a binary classifier to tell GLM 5.3 from
GLM 5.3 Flash better than the official passages alone?

**The target.** The incumbent has 58 direct swaps between the two versions: 28 GLM 5.3
passages called Flash, and 30 Flash passages called GLM 5.3
([error inspection](fable-glm-errors.md)).

**Why test it this way.** In the [CPU-model experiment](external-data.md), adding external
passages raised the recall of their classes and lowered their precision, a shift whose cause
is not established. A pairwise comparison against an official-only control tests whether
the passages carry version-distinguishing information at all.

**References:**
- the incumbent forced to choose between the two GLM versions, on the 300 official passages
  of the pair;
- the complete incumbent: 47.01% grouped CV, 0.429 public.

## Design

This section was written before any model was fit.

**Population.** The 300 official passages labelled GLM 5.3 or GLM 5.3 Flash (150 each, one
per version in every prompt). They use the original five prompt folds
(`artifacts/lora/plan.csv`): in each fold, 240 training passages and 60 validation passages.
Only official passages are ever validated.

**Matched external passages** (from the audited dataset in [external-data.md](external-data.md)):
- **Source.** `zaiglm` rows only. Both labels come from the official Z.AI provider, pinned
  with no fallbacks, through the same generator and settings. The `orglm` rows are excluded:
  the two labels came through different third-party providers there (Relace and
  OpenInference), so a classifier could learn the provider instead of the version.
- **Identifiers.** Only rows whose recorded `served_model` matches the label (all `zaiglm`
  rows do).
- **Shared prompts.** Only prompts with passages from both versions. A prompt contributes
  the same number of passages for each version, the first by `row_id`. The result is 158
  pairs: equal external weight for the two classes.
- **Overlap exclusions, as before.** An external prompt linked to a competition prompt (the
  topic links of the pool, τ = 0.186) is used only in folds where that competition prompt is
  on the training side. That affects 12 rows.

**Models:** two binary logistic regressions on the existing CPU features, with the existing
regularization: char and word TF-IDF plus style features, `C = 4`, class-balanced weights,
lbfgs with at most 1,000 iterations, seed 2026. Every vocabulary and scaler is fit on the
fold's training passages.

| Model | Training passages per fold |
| --- | --- |
| Control | The 240 official training passages |
| Augmented | The same plus the eligible matched external passages |

Each model scores P(Flash), and above 0.5 predicts Flash.

**Comparators.** Both models are compared with the incumbent's forced choice: Flash when
p_Flash / (p_GLM + p_Flash) > 0.5, from its locked out-of-fold probabilities.

**Reported, pooled and by fold:**
- binary balanced accuracy, accuracy and both recalls;
- swaps in each direction;
- swaps corrected against introduced, relative to both the control and the incumbent's
  forced choice;
- paired prompt-bootstrap intervals (10,000 resamples, seed 2026) for the augmented model
  against each comparator;
- as a diagnostic, how many of the incumbent's 58 direct swaps each model calls correctly.

**Gate for integration.** Integration is considered only if separation improves: the
augmented model's pooled binary balanced accuracy must exceed both the control's and the
incumbent's forced choice.

**If it passes, stage 2.**
- **Integration.** The binary model redistributes the complete incumbent's mass on the pair:
  `p′ₖ = (1 − a) pₖ + a M qₖ`, with M = p_GLM + p_Flash.
- **Selection.** The inner folds choose `a` between the incumbent and {¼, ½, ¾, 1}, with the
  incumbent rebuilt on each inner partition. Ties keep the incumbent, then the smaller `a`.
- **Promotion,** unchanged: +0.5 points, with a 95% paired prompt-bootstrap interval above
  zero.

**Deferred** until this comparison gives a reason to continue:
- the Qwen features of external passages (a GPU run);
- the unverified rows.

## Result: original prompt folds (October 28, 2026)

```sh
PYTHONPATH=src .venv/bin/python -m ntl.glm_pair --train data/raw/train.csv --plan artifacts/lora/plan.csv \
  --external data/external/ntl-openrouter-openai/ntl_openrouter_openai_generations.parquet \
  --pool artifacts/external-data/pool.parquet \
  --reference artifacts/styledistance-adapt/reference-prompts.json --output-dir artifacts/glm-pair/prompts
```

**The matched external passages:**
- 316 passages: 158 per version from 158 shared prompts, all Z.AI-served and verified.
- 12 are topic-linked, so each fold used 312–316 alongside its 240 official passages. 0 to
  4 linked passages were held back, always in equal numbers per version.

**Binary separation on the 300 official GLM passages:**

| Predictor | Binary BA | GLM 5.3 recall | GLM Flash recall | GLM → Flash | Flash → GLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Incumbent, forced choice | **57.3%** | 59.3% | 55.3% | 61 | 67 |
| Control: official passages only | 51.7% | 49.3% | 54.0% | 76 | 69 |
| Official + matched external | 53.3% | 45.3% | 61.3% | 82 | 58 |

**Paired comparisons:**

| Comparison | Corrected / introduced | Change | 95% interval |
| --- | --- | ---: | --- |
| Augmented against the control | 41 / 36 | +1.7 | [−4.0, +7.3] |
| Augmented against the incumbent's forced choice | 50 / 62 | −4.0 | [−10.3, +2.0] |
| Control against the incumbent's forced choice | 41 / 58 | −5.7 | [−11.0, −0.3] |

**By fold** (control → augmented): 48.3 → 53.3, 51.7 → 50.0, 48.3 → 50.0, 56.7 → 51.7 and
53.3 → 61.7. The augmented model is better in three folds and worse in two.

**The incumbent's 58 direct swaps.** The control calls 17 of them correctly, and the
augmented model 19.

**Gate: failed.** The augmented model (53.3%) does not exceed the incumbent's forced choice
(57.3%). As declared, no integration was built, and the full system is unchanged.

**What this answers:**
- **The CPU features barely separate the two GLM versions.** Trained on official passages
  alone, they reach 51.7%, near chance. The incumbent's own forced choice, which carries the
  Qwen + StyleDistance specialist, is clearly better.
- **The matched external passages do not teach a useful distinction with these features.**
  Adding 316 provider- and prompt-matched passages moves the score by +1.7 points, within
  noise. The model also tilts toward Flash: GLM 5.3 recall falls 4 points and Flash recall
  rises 7.
- **What is untested.** The matched set is small, and the passages may still differ from the
  competition's in ways the matching did not control. This tests the existing CPU
  representation only.
