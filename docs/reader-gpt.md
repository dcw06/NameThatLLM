# READER fingerprints for the GPT classes

**Question.** Do READER's activation-trajectory fingerprints separate GPT-5.5, GPT-5.6 Luna /
Sol / Terra and GPT-6 Astra better than the current features? And does adding them to the
[GPT specialist](gpt-specialist.md) improve the complete incumbent?

**References** (as for [DeTeCtive](detective-gpt.md)):
- five-way balanced accuracy on the 750 true GPT passages: 50.0% for the incumbent forced
  among the five, 50.7% for the Qwen + StyleDistance GPT specialist;
- that specialist gained +0.26 points in the complete system, and was not promoted;
- the complete incumbent: 47.01% grouped CV, 42.30% on eight topics, 0.429 public.

## READER

READER (Liu et al., arXiv 2606.10794; code `LeoJeshua/READER`, MIT licence) builds a
fingerprint from how a frozen proxy model reads a passage:
- **Input:** the response alone, tokenized without special tokens or a chat template,
  truncated at 1,024 tokens. This is READER's "R" mode, which matches the competition's
  passages.
- **Activations:** the hidden-state trajectory at one decoder layer.
- **DC component:** the length-normalized DCT-II mode 0 over the response tokens, which is the
  token mean.
- **AC component:** mode 1, √2 · Σₜ cos(π(t + ½)/N) hₜ / N, the trajectory's first-half
  against second-half trend.
- **Fingerprint:** DC and AC concatenated.

READER's own probe standardizes the features and fits a zero-initialized multinomial linear
layer: full-batch Adam, learning rate 1e-3, cosine schedule over 100 steps, stopped at 40
steps, L2 = ‖W‖² / (2 · C · N) with C = 1.

**The proxy used: READER's official Qwen3-8B preset,** `Qwen/Qwen3-8B` read at layer 2 (the
output of the second decoder block, before any final norm).

**Why not the canonical proxy, Qwen3.5-9B at layer 18?** It cannot run here:
- the project's pinned `transformers` 4.57.1 does not support the Qwen3.5 architecture;
- at 9.65 billion parameters it does not fit this machine without a GPU.

A canonical run would need a separate environment or a GPU, and is left as a follow-up.

**Deviations from the authors' runs:**
- **Precision and device.** FP32 on the CPU, where the authors used bf16 on CUDA. This is
  for determinism; the numbers differ only at floating-point level.
- **Truncated model.** Only the embeddings and the first two decoder blocks are loaded, from
  the pinned `model-00001-of-00005.safetensors` (revision
  `b968826d9c46dd6066d109eabc6255188de91218`, Apache-2.0). That is exactly what READER's early
  exit computes.
- **No authors' code.** None of the authors' code is run, and no `trust_remote_code` is used.

## Design

This section was written before any weights were downloaded.

**Stage 1: can the fingerprints separate the GPT versions?** This is diagnostic. Five-way
balanced accuracy is measured on the 750 true GPT passages. Each fold's model is fit on the
true GPT passages outside that original prompt fold:

| Predictor | Recipe |
| --- | --- |
| READER DC + AC | READER's probe, as above |
| READER DC only | READER's probe |
| READER AC only | READER's probe |
| READER DC + AC alone | The specialist recipe: StandardScaler, logistic regression, `C = 0.005` |
| Qwen + StyleDistance + READER | The specialist recipe, each block after Qwen scaled to Qwen's total variance |

**Stage 2: does it improve the complete incumbent?** This is the decision.
`ntl.gpt_specialist` is used unchanged, with one feature set, Qwen + StyleDistance + READER
(DC + AC):
- `p′ₖ = (1 − w) pₖ + w M qₖ` on the five GPT classes;
- the inner folds rebuild the complete incumbent and choose between it and `w ∈ {¼, ½, ¾, 1}`;
- ties keep the incumbent, then the smaller `w`;
- the rebuilt outer incumbent must equal the locked reference.

**Promotion rule,** unchanged:
- an unrounded gain of at least +0.5 points;
- a 95% paired prompt-bootstrap interval above zero (10,000 resamples, seed 2026).

Only a passing candidate repeats the procedure on the eight-topic split.

**Also reported.** The candidate against the earlier Qwen + StyleDistance GPT specialist.

**Not used: READER's released response data.** Its Agent500 pool includes 500 responses each
from sources named `gpt55`, `gpt56_luna`, `gpt56_sol`, `gpt56_terra`, `kimi_k3`, `claude_opus46`
and `claude_opus48`. Whether external data may be used depends on the competition's rules and
the data's terms ("subject to applicable source-model and API-provider terms"). That is a
separate decision, and none of it is used here.

## Result: original prompt folds (October 28, 2026)

```sh
PYTHONPATH=src .venv/bin/python -m ntl.reader_fingerprint extract --train data/raw/train.csv \
  --model-dir models/qwen3-8b-reader --output artifacts/reader/reader-qwen3_8b-layer2-train.npz
PYTHONPATH=src .venv/bin/python -m ntl.reader_fingerprint evaluate --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv \
  --qwen-features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --styledistance-features artifacts/claude-glm/styledistance-train.npz \
  --reader-features artifacts/reader/reader-qwen3_8b-layer2-train.npz \
  --current-oof artifacts/qwen-blend/oof.parquet \
  --current-inner artifacts/claude-glm/current-inner/fold{0,1,2,3,4}.parquet \
  --reference artifacts/styledistance-adapt/reference-prompts.json \
  --earlier artifacts/gpt-specialist/prompts/predictions.parquet --output-dir artifacts/reader/prompts
```

**Extraction:**
- The pinned shard matches its published SHA-256.
- Its embeddings and blocks 0–1 load into a two-block Qwen3 model. Only the unused final
  norm is missing.
- On tiny models, the two-block early exit equals the full model's layer-2 hidden state.
- All 2,699 passages took about 7 minutes (0.16 s each on the CPU, 8.2 GB peak memory).
  Response lengths run from 44 to 381 tokens (median 130), with none truncated, and repeated
  fingerprints are identical.
- The rebuilt outer incumbent equals the locked reference in every fold.

### Stage 1: separating the GPT versions

Five-way balanced accuracy and recall on the 750 true GPT passages. Chance is 20%.

| Predictor | Five-way BA | GPT-5.5 | Luna | Sol | Terra | Astra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent, forced among the five | 50.0% | 61.3% | 41.3% | 36.7% | 36.7% | 74.0% |
| Qwen + StyleDistance specialist | 50.7% | 59.3% | 47.3% | 40.0% | 42.0% | 64.7% |
| Qwen + StyleDistance + READER | 49.7% | 61.3% | 44.7% | 35.3% | 40.7% | 66.7% |
| READER DC + AC, READER's probe | 42.7% | 51.3% | 37.3% | 28.0% | 31.3% | 65.3% |
| **READER DC only, READER's probe** | **48.7%** | 60.0% | 42.7% | 30.0% | 37.3% | 73.3% |
| READER AC only, READER's probe | 27.6% | 31.3% | 22.7% | 20.0% | 25.3% | 38.7% |
| READER DC + AC, specialist recipe | 44.3% | 52.7% | 36.7% | 29.3% | 35.3% | 67.3% |

**What stage 1 shows:**
- **The DC part is strong on its own.** The mean of Qwen3-8B's layer-2 states reaches 48.7%,
  close to the incumbent's forced choice and far above DeTeCtive (25%).
- **The AC part is weak.** The trajectory trend that READER adds carries little (27.6%), and
  including it lowers READER's own probe from 48.7% to 42.7%.
- **Nothing beyond the existing features.** Added to Qwen + StyleDistance, READER does not
  improve the five-way separation (49.7% against 50.7%).

### Stage 2: the complete system

| System | Grouped BA | Change | 95% interval | Gained / lost |
| --- | ---: | ---: | --- | --- |
| Incumbent | 0.4701 | — | — | — |
| GPT specialist on Qwen + StyleDistance + READER | 0.4716 | +0.15 | [−0.48, +0.81] | 50 / 46 |

**Not promoted.** The gain is +0.15 points, against a required +0.5 with an interval above
zero; 35% of resamples are at or below zero. As declared, the eight-topic check was not run,
and nothing was packaged.

**Selection and its held-out effect:**
- By fold, the inner folds chose `w` = ¼, the incumbent, ½, ¼ and ½, by inner margins of
  +0.19 to +0.88 points.
- The held-out changes were +0.19, 0.00, −0.74, +1.67 and −0.37 points.
- In hindsight, `w = ¼` in every fold would gain +0.48 points. `w = ½` would lose 0.22, and
  `w = 1` would lose 1.96.

**Against the earlier Qwen + StyleDistance GPT specialist,** built the same way: −0.11
points (interval [−0.63, +0.41]).

**Recall:**
- **GPT classes:** GPT-5.5 rises from 46.0% to 48.7%, Luna from 36.7% to 38.7%, Sol from
  31.3% to 33.3%, Terra from 34.7% to 37.3%, and Astra from 68.7% to 73.3%.
- **Forced among the five,** the candidate reaches 53.7%.
- **Other classes pay for it.** Eleven Qwen3.8 Max and ten Kimi K3 passages are pulled into
  GPT predictions:
  - Qwen3.8 Max falls from 51.3% to 44.7% (−10);
  - Kimi K3 falls from 44.0% to 39.3% (+1 / −8).

**What this answers:**
- **The preset matches, but doesn't add.** READER's Qwen3-8B preset captures GPT-version
  information comparable to the current Qwen features, but not beyond them.
- **The same trade-off.** It repeats the earlier GPT specialists' pattern: GPT recall rises
  a few points, Qwen3.8 Max and Kimi K3 pay for it, and the net change is within noise.

**Not tried:**
- **The canonical Qwen3.5-9B proxy at layer 18.** It needs a newer `transformers` than the
  project's pinned 4.57.1, and a GPU or a much larger machine.
- **READER's released response data.** This remains a separate decision, as above.
