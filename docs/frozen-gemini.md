# Frozen representations and regularized Gemini-version classification

This experiment compares frozen StyleDistance and Qwen3-Embedding-0.6B representations
followed by L2-regularized logistic regression. The target is the four Gemini versions,
using the same original five prompt folds and 570 correctly routed evaluation IDs.
The CPU reference gets 291/570 right (51.05%); the earlier three-epoch DeBERTa
fine-tuning experiment gets 166/570 right (29.12%).

| Encoder | Immutable revision | Representation |
| --- | --- | --- |
| `StyleDistance/styledistance` | `b7df5f0b0480773c097ba3121d83ca32b71015ca` | Native masked mean, including unmasked special tokens; 768 dimensions |
| `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | Native last non-padding token; 1,024 dimensions |

Both use the complete backbone, evaluation mode, disabled gradients, float32 inference,
batch size 16, and a 512-token limit. Neither receives an instruction prefix.
Embeddings are L2-normalized per passage. This is native to the Qwen representation;
for StyleDistance it is an explicit preprocessing choice after its native mean pooling.
There is no encoder fine-tuning, quantization, layer selection, or learned projection.
The snapshot's Sentence Transformers module and pooling configurations are checked
before extraction, using native Transformers models with no remote code.

## Validation and regularization

Frozen extraction is pointwise and independent of labels, so the 600 true-Gemini
passages are embedded once and cached. Each outer fold fits the classifier on 480
passages from the other 120 prompts, then predicts 120 passages from 30 held-out prompts.
The 570-row selection affects reporting only; it never filters the fitting population.

Within each outer fitting set, three inner prompt-grouped folds choose
`C` from `[0.001, 0.01, 0.1, 1, 10]` by mean balanced accuracy. Smaller `C` wins ties.
Every inner candidate fits a fresh `StandardScaler` and logistic-regression model on
its fitting rows only. The chosen pipeline is then refitted on the complete outer
fitting partition. No outer validation labels or scores select regularization.
Logistic regression uses L2 regularization, the LBFGS solver, and at most 2,000 iterations.

The result is exact-version accuracy on a conditional, correctly routed Gemini cohort.
It does not measure family routing or an 18-model hidden-test submission. All-600
results are saved separately, as are paired corrections/regressions against the CPU
reference, per-version recalls, confusion matrices, and full per-entry probabilities.

## Reproduction and artifacts

```sh
.venv/bin/python scripts/prepare_frozen_gemini_experiment.py \
  --username YOUR_USERNAME --fresh-runtime
.venv/bin/kaggle kernels push -p artifacts/frozen_gemini_experiment \
  -t 3600 --accelerator NvidiaTeslaT4
```

The private GPU notebook pins the baseline output and an optional previously completed
runtime. New environments and model downloads live under `/tmp`, keeping large binaries
out of saved notebook output. Only source, feature caches, reports, and small metadata
are retained. No competition submission is created.

Each pickle-free NPZ records exact IDs and text hashes, embedding bytes, model revision,
pooling, normalization, precision, token limits, truncation counts, and runtime versions.
The evaluator verifies those hashes and aligns rows by ID before fitting anything.
Changing a passage or using another cohort fails validation.

## Measured results, October 26, 2026

The completed private run is
[`name-that-llm-frozen-gemini-embeddings`, version 2](https://www.kaggle.com/code/daichongwei06/name-that-llm-frozen-gemini-embeddings).
It uses the fixed protocol above. Both frozen models improve on the earlier
DeBERTa recipe, but neither improves on the CPU reference.

| Exact version accuracy on the same 570 entries | Correct | Accuracy | Difference from CPU |
| --- | ---: | ---: | ---: |
| CPU reference | 291 | 51.05% | — |
| Frozen StyleDistance + logistic regression | 253 | 44.39% | -6.67 percentage points |
| Frozen Qwen3-Embedding-0.6B + logistic regression | 186 | 32.63% | -18.42 percentage points |
| Earlier fine-tuned DeBERTa-v3-base | 166 | 29.12% | -21.93 percentage points |

| True Gemini version | Selected rows | CPU correct | StyleDistance correct | Qwen embedding correct |
| --- | ---: | ---: | ---: | ---: |
| `gemini-3.1-pro-preview` | 147 | 114 (77.55%) | 90 (61.22%) | 65 (44.22%) |
| `gemini-3.6-flash` | 143 | 60 (41.96%) | 50 (34.97%) | 36 (25.17%) |
| `gemini-3.7-flash` | 143 | 68 (47.55%) | 61 (42.66%) | 38 (26.57%) |
| `gemini-3.8-flash` | 137 | 49 (35.77%) | 52 (37.96%) | 47 (34.31%) |

On the selected cohort, StyleDistance corrects 84 CPU errors while losing 122
previously correct predictions. Qwen corrects 71 and loses 176. Across all 600
true-Gemini passages, StyleDistance gets 266 correct (44.33%) and Qwen gets 198
correct (33.00%). These are measurements of the specified feature and classifier
recipes; they do not establish the best possible performance of either encoder.

The inner folds select StyleDistance `C` values `[0.1, 0.01, 0.01, 0.1, 0.1]`
and Qwen values `[0.001, 0.001, 10, 0.01, 10]`, in outer-fold order. There is no
hyperparameter selection against these reported outer scores.

Both feature matrices cover all 600 exact input IDs and passages and contain
finite, L2-normalized float32 vectors. No passages are truncated: the longest
StyleDistance input is 351 tokens and the longest Qwen input is 348. On one T4,
extraction takes 9.0 seconds for StyleDistance and 50.3 seconds for Qwen, including
model loading. Peak allocated GPU memory is about 0.66 GiB and 2.52 GiB respectively.
Nested classifier fitting and reporting take another 9.6 and 5.2 seconds.
These times exclude environment installation and checkpoint download.

Version 1 stalled during recursive discovery of a large attached environment;
version 2 uses a fresh temporary environment and bounded input paths. This setup
change does not alter the encoder or evaluation protocol. The reusable notebook
also limits runtime probes to 180 seconds and logs setup stages.

Downloaded reports are under `artifacts/frozen-gemini-results/`. Each model has
`selected_predictions.csv`, full OOF probabilities, metrics, confusion matrices,
and inner-fold records. `predictions_570_comparison.csv` aligns CPU, DeBERTa,
StyleDistance, and Qwen predictions for every requested entry. These raw artifacts
and feature caches remain excluded from Git. No new competition submission is made.

Sources: [StyleDistance model card](https://huggingface.co/StyleDistance/styledistance),
[Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
