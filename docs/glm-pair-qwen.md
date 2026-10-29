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
