# GPT specialist

**Question.** Does a specialist trained on the five GPT classes improve the complete incumbent
by redistributing its GPT probability mass? The five classes are GPT-5.5, GPT-5.6 Luna, Sol
and Terra, and GPT-6 Astra.

**References to beat:**
- 47.01% grouped CV (the complete incumbent: blend plus Claude–GLM specialist);
- 42.30% on the eight-topic split;
- 0.429 on the public leaderboard.

## Why the GPT classes

The incumbent's recall is 36.7% for Luna, 31.3% for Sol and 34.7% for Terra. Its out-of-fold
predictions on the 750 true GPT passages:

| True ↓ / predicted → | GPT-5.5 | Luna | Sol | Terra | Astra | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.5 | 69 | 8 | 12 | 11 | 10 | 40 |
| Luna | 10 | 55 | 24 | 19 | 18 | 24 |
| Sol | 13 | 28 | 47 | 23 | 11 | 28 |
| Terra | 20 | 21 | 25 | 52 | 8 | 24 |
| Astra | 7 | 16 | 7 | 8 | 103 | 9 |

83% of true GPT passages are already routed to some GPT class. Forced to choose among the
five, the incumbent reaches 50.0% balanced accuracy. The weak versions are confused with
GPT-5.5 and Astra as well as with each other, so the specialist covers all five.

Raising Luna, Sol and Terra recall by 10 points each would add about 1.67 points of
balanced accuracy if nothing else changed. This is a plausible target, not an expected
gain: the [18-class StyleDistance head](styledistance-head.md) did not reliably help Luna or
Sol.

## Design

This section was written before any fitting.

**Specialist.** This is the Claude–GLM specialist's recipe, unchanged except for its
classes:
- fit on the true GPT passages of each training partition, whatever the incumbent predicted;
- a StandardScaler fit on training rows only;
- for Qwen + StyleDistance, the StyleDistance block scaled by √(8192/768);
- logistic regression with `C = 0.005`, at most 2,000 iterations, seed 2026;
- five probabilities, in canonical class order, for every scored passage.

| Feature set | Blocks |
| --- | --- |
| `qwen` | Cached frozen Qwen3-8B-Base features (8,192) |
| `qwen+styledistance` | The same plus cached frozen StyleDistance (768) |

**Integration.** With `M` the incumbent's mass on the five GPT classes and `q` the
specialist's distribution:
- `p′ₖ = (1 − w) pₖ + w M qₖ` for the five classes;
- the other 13 probabilities are unchanged, so `w = 0` is the incumbent exactly.

**Selection,** within the original five prompt folds and their two inner prompt folds
(`artifacts/lora/plan.csv`). For each outer fold:
1. The complete incumbent is rebuilt on each inner training partition. That is the
   base-blend inner cache, plus the Claude–GLM specialist at the `s` its inner folds choose.
   The rebuilt outer incumbent must equal the locked reference within 1e-12.
2. The inner folds choose one of nine options, by mean inner 18-class balanced accuracy:
   the incumbent, or a feature set at `w ∈ {¼, ½, ¾, 1}`.
3. Ties use the shared 1e-12 rule, in this order: the incumbent, then the smaller `w`, then
   Qwen before Qwen + StyleDistance.
4. The chosen option is fit on the outer training partition and applied once, to the locked
   incumbent's vectors for the outer fold.

**Promotion rule,** unchanged:
- an unrounded gain of at least +0.5 points;
- a 95% paired prompt-bootstrap interval above zero (10,000 whole-prompt resamples, seed
  2026).

Only a passing candidate repeats the procedure on the eight-topic split, as earlier
specialists did. Only after both is anything packaged.

**Reported:**
- overall balanced accuracy, with per-class recall and precision changes and gains and losses
  across every class;
- the GPT confusion block and the GPT routing (passages moving into or out of the GPT classes);
- the selected options;
- **the Qwen against Qwen + StyleDistance comparison:** each feature set on its own, with `w`
  chosen by the same inner folds, against the incumbent and against each other.

**Diagnostics.** These are not part of the decision.
- Each specialist's five-way balanced accuracy on the 750 true GPT passages, against the
  incumbent forced to choose among the five (50.0%).
- Every option applied in every fold, in hindsight.

## Result: original prompt folds (October 28, 2026)

```sh
PYTHONPATH=src .venv/bin/python -m ntl.gpt_specialist --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv \
  --qwen-features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --styledistance-features artifacts/claude-glm/styledistance-train.npz \
  --current-oof artifacts/qwen-blend/oof.parquet \
  --current-inner artifacts/claude-glm/current-inner/fold{0,1,2,3,4}.parquet \
  --reference artifacts/styledistance-adapt/reference-prompts.json \
  --output-dir artifacts/gpt-specialist/prompts
```

**Checks.** In every fold, the rebuilt outer incumbent equals the locked reference exactly
(maximum difference 0), with `s` = ¼, ¼, ½, ¼, ¼. The run takes under a minute on the CPU.

| System | Grouped BA | Change | 95% interval | Gained / lost |
| --- | ---: | ---: | --- | --- |
| Incumbent | 0.4701 | — | — | — |
| **Candidate** (feature set and `w` chosen by the inner folds) | 0.4727 | **+0.26** | [−0.30, +0.81] | 33 / 26 |
| Qwen + StyleDistance alone | 0.4727 | +0.26 | [−0.30, +0.85] | 35 / 28 |
| Qwen alone | 0.4653 | −0.48 | [−1.00, +0.04] | 24 / 37 |

**Not promoted.** The gain is +0.26 points, against a required +0.5 with an interval above
zero; 20% of resamples are at or below zero. As declared, the eight-topic check was not run,
and nothing was packaged or submitted.

**Selection:**

| Fold | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Selected | Qwen + SD, `w` ¼ | Incumbent | Qwen, `w` ¼ | Qwen + SD, `w` ¼ | Qwen + SD, `w` ¼ |
| Inner margin (points) | +0.74 | — | +0.23 | +0.69 | +0.69 |
| Held-out change (points) | −0.19 | 0.00 | −0.56 | +1.30 | +0.74 |

No fold chose more than `w = ¼`. Applied in every fold in hindsight, Qwen + StyleDistance at
`w = ¼` is the only option above the incumbent (+0.37 points). Every larger weight loses: by
1.3–2.1 points at `w` = ¾ or 1.

**Qwen against Qwen + StyleDistance.** With StyleDistance the system scores 0.74 points
higher than without it (interval [+0.22, +1.30]). Qwen-only features lose to the incumbent.

**GPT recall and precision** (incumbent → candidate):

| Class | Recall | Precision | Gained / lost |
| --- | --- | --- | --- |
| GPT-5.5 | 46.0% → 48.7% | 43.7% → 43.5% | +5 / −1 |
| GPT-5.6 Luna | 36.7% → 40.7% | 34.6% → 35.9% | +6 / −0 |
| GPT-5.6 Sol | 31.3% → 33.3% | 35.3% → 36.8% | +6 / −3 |
| GPT-5.6 Terra | 34.7% → 36.7% | 36.9% → 39.9% | +10 / −7 |
| GPT-6 Astra | 68.7% → 68.7% | 62.8% → 61.7% | +4 / −4 |

**The cost falls outside the GPT classes.** Qwen3.8 Max recall falls from 51.3% to 46.7%
(−7), and Kimi K3 from 44.0% to 42.7% (+2 / −4). Eight true Qwen3.8 Max passages move into
GPT predictions, as the specialist sharpens the GPT mass on passages the incumbent had nearly
split. No other class changes recall; a few precisions move slightly.

**Five-way separation on the 750 true GPT passages.** This is a diagnostic.

| Predictor | Five-way BA |
| --- | ---: |
| Incumbent, forced among the five | 50.0% |
| Candidate, forced among the five | 52.5% |
| Qwen + StyleDistance specialist alone | 50.7% |
| Qwen specialist alone | 49.2% |

**What this says:**
- On its own, the specialist does not separate the GPT versions better than the incumbent.
- The gain comes from mixing it in at a small weight.
- The combination moves Luna, Sol and Terra each up by 2–4 points of recall, short of the 10
  needed for the hoped-for 1.67 points.
