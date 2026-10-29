# GLM 5.3 against GLM Flash on Qwen + StyleDistance, with matched external passages

**Question.** With the stronger Qwen + StyleDistance representation, do the same 316 matched
external passages improve binary separation of GLM 5.3 from GLM 5.3 Flash over an
official-only control?

**Why.** The [CPU-feature test](glm-pair.md) started from a weak control (51.7%), so its
failure does not show whether external passages help this representation.

**References:**
- the incumbent's forced choice between the two versions on the 300 official passages:
  57.3%;
- the complete incumbent: 47.01% grouped CV, 0.429 public.

**Dataset version.** Version 2 of the external dataset (SHA-256 `ed59c651…fd390e`) is
used. Version 1's 1,120 rows built with Kaggle Benchmarks quota were withdrawn, and none of
them is among the 316 matched passages. Version 2 yields the identical matched set and the
same 12 topic links (τ = 0.185 with the version-2 pool).

## Features of the external passages

These must match the official passages' cached features exactly in recipe.

- **Qwen3-8B-Base** (`49e3418…`):
  - the first 24 blocks, masked mean of blocks 7 and 22;
  - NF4 with float16 compute, at most 512 tokens, batch size 2, right padding;
  - extracted with `ntl.ensemble extract` on a Kaggle T4, as the official passages' were.
  
  The private kernel built by `scripts/prepare_external_qwen_features.py` does this:
  - it attaches the public dataset;
  - it checks the audited parquet's SHA-256;
  - it selects exactly the 316 matched passages by `row_id`.

  In the same run it re-extracts the first 16 official passages of `train.csv`, in their
  original batch pairs, so the new environment can be compared with the cached official
  features.
- **StyleDistance** (`b7df5f0…`): FP32 on this machine's CPU with
  `ntl.encoders.extract_embeddings`, at most 512 tokens, masked mean including special tokens,
  L2-normalized, as the cached official features were. Re-extracting official passages
  checks reproduction first.

## Design

This section was written before any external feature was extracted.

**Unchanged from the CPU test:**
- the 300 official passages and the original five prompt folds;
- the 316 matched external passages (Z.AI-served for both versions, verified, 158 shared
  prompts with equal counts per version);
- the topic-link exclusions (τ = 0.186; 12 linked rows);
- validation on official passages only.

**Models.** Two binary classifiers with identical features and settings: the Claude–GLM
specialist's recipe on Qwen + StyleDistance. That is a StandardScaler, the StyleDistance
block scaled to Qwen's total variance, and logistic regression with `C = 0.005`, at most
2,000 iterations, seed 2026. The scaler is fit on each fold's training passages.

| Model | Training passages per fold |
| --- | --- |
| Control | The 240 official training passages |
| Augmented | The same plus the eligible matched external passages |

Each model scores P(Flash), and above 0.5 predicts Flash.

**Reported, as in the CPU test:**
- binary balanced accuracy and both recalls, pooled and by fold;
- swaps in each direction;
- corrected against introduced;
- paired prompt-bootstrap intervals (10,000 resamples, seed 2026) against the control and
  against the incumbent's forced choice;
- the incumbent's 58 direct swaps called correctly.

**A convincing binary gain, required before any integration.** Both must hold:
- the augmented model beats its control, with the paired prompt-bootstrap 95% interval
  entirely above zero;
- its pooled binary balanced accuracy exceeds the incumbent's forced choice (57.3%).

**If it passes, stage 2.**
- **Integration.** The binary model redistributes the complete incumbent's mass on the pair:
  `p′ₖ = (1 − a) pₖ + a M qₖ`.
- **Selection.** The inner folds choose `a` between the incumbent and {¼, ½, ¾, 1}.
- **Promotion,** unchanged: +0.5 points, with a 95% paired prompt-bootstrap interval above
  zero.

**Out of scope:** further CPU variants and the unverified rows.

## Result: original prompt folds (October 29, 2026)

```sh
PYTHONPATH=src .venv/bin/python -m ntl.glm_pair_qwen styledistance --train data/raw/train.csv \
  --external data/external/ntl-openrouter-openai-v2/ntl_openrouter_openai_generations.parquet \
  --pool artifacts/external-data-v2/pool.parquet --model-path models/styledistance \
  --cached artifacts/claude-glm/styledistance-train.npz --output artifacts/glm-pair/styledistance-matched.npz
PYTHONPATH=src .venv/bin/python scripts/prepare_external_qwen_features.py --username daichongwei06
kaggle kernels push -p artifacts/external_qwen_features   # private; version 2 of the kernel
PYTHONPATH=src .venv/bin/python -m ntl.glm_pair_qwen evaluate --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv \
  --external data/external/ntl-openrouter-openai-v2/ntl_openrouter_openai_generations.parquet \
  --pool artifacts/external-data-v2/pool.parquet \
  --qwen-features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --styledistance-features artifacts/claude-glm/styledistance-train.npz \
  --external-qwen artifacts/external-qwen-results/v2/features/qwen3-8b-base-external-glm.npz \
  --external-qwen-reference artifacts/external-qwen-results/v2/features/qwen3-8b-base-official-reference.npz \
  --external-styledistance artifacts/glm-pair/styledistance-matched.npz \
  --reference artifacts/styledistance-adapt/reference-prompts.json --output-dir artifacts/glm-pair-qwen/prompts
```

**Feature checks:**
- **StyleDistance.** Re-extracting the first 32 official passages reproduced their cached
  features exactly (maximum difference 0.0).
- **The first Kaggle run stopped at its own safeguard.** The attached dataset had become
  version 2, whose hash did not match the audited version 1, so nothing was extracted.
- **The rerun used version 2,** after the checks described above.
- **Qwen.** The Kaggle T4 run (NF4, float16, none truncated) re-extracted the first 16
  official passages in their original batches. It reproduced their cached features exactly
  (maximum difference 0.0), so the external passages' features come from an identical
  environment.

**Binary separation on the 300 official GLM passages** (312–316 external passages per fold,
0–4 held back by topic links):

| Predictor | Binary BA | GLM 5.3 recall | GLM Flash recall | GLM → Flash | Flash → GLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Incumbent, forced choice | **57.3%** | 59.3% | 55.3% | 61 | 67 |
| Control: Qwen + StyleDistance, official only | 56.7% | 58.0% | 55.3% | 63 | 67 |
| Official + matched external | 54.0% | 46.7% | 61.3% | 80 | 58 |

**Paired comparisons:**

| Comparison | Corrected / introduced | Change | 95% interval |
| --- | --- | ---: | --- |
| Augmented against the control | 39 / 47 | −2.7 | [−7.7, +2.3] |
| Augmented against the incumbent's forced choice | 49 / 59 | −3.3 | [−8.7, +2.0] |
| Control against the incumbent's forced choice | 20 / 22 | −0.7 | [−4.7, +3.3] |

**By fold** (control → augmented): 61.7 → 56.7, 53.3 → 38.3, 58.3 → 65.0, 48.3 → 46.7 and
61.7 → 63.3.

**The incumbent's 58 direct swaps.** The control calls 8 correctly and the augmented model 22.
The augmented model introduces more new errors elsewhere than that.

**Gate: failed.** The augmented model does not beat its control (−2.7 points, with an
interval spanning zero), and it falls below the incumbent's forced choice. As declared, no
integration was built.

**What this answers:**
- **A strong control, no gain from the external passages.** On the stronger representation,
  the official-only control nearly matches the incumbent's forced choice (56.7% against
  57.3%). Adding the 316 matched passages gives no demonstrated improvement: the point
  estimate is −2.7 points, and the interval (−7.7 to +2.3) still allows a small gain or a
  larger loss.
- **The same tilt toward Flash as with CPU features.** GLM 5.3 recall falls about 11 points
  and Flash recall rises about 6.
- **A possible cause, not tested.** The consistent tilt suggests the external GLM 5.3
  passages differ from the competition's GLM 5.3 passages, for example in generation
  settings. This experiment cannot establish the cause.
