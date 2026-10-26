# Gemini version experiment with DeBERTa-v3-base

This diagnostic uses `microsoft/deberta-v3-base`, pinned to
`8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, to predict one of the four Gemini versions.
The upstream model is pretrained rather than an authorship classifier, so a new
four-class head and the backbone are fine-tuned for this task.

The requested evaluation cohort contains the **570 true-Gemini passages** for which
the saved first CPU model also predicted a Gemini label. The CPU model identified the
exact version for **291 / 570 = 51.05%**. This denominator differs from its 43.83%
precision across all 664 Gemini predictions, which include 94 non-Gemini passages.

## Fixed validation protocol

- Use the exact out-of-fold predictions and fold manifest from
  `daichongwei06/name-that-llm-cpu-first-submission/1`.
- Keep the original five prompt-grouped folds, with 30 complete prompts held out each time.
- In each fold, initialize a fresh pinned pretrained backbone and four-class head;
  fine-tune on all **480 true-Gemini training-fold passages**.
- Predict the 120 held-out true-Gemini passages. Across five folds, every one of the
  600 Gemini passages receives exactly one prediction made without its prompt in training.
- Apply the fixed 570-row selection only when reporting. It does not filter the
  fitting population or select epochs, checkpoints, thresholds, or hyperparameters.
- Train for three fixed epochs with AdamW, learning rate `2e-5`, weight decay `0.01`,
  warmup ratio `0.1`, gradient clipping `1.0`, microbatch eight, accumulation two,
  dynamic padding, and maximum token length 512. Use float32 parameters and CUDA
  float16 autocast with GradScaler. Each fold uses seed `2026 + fold`.

The pinned tokenizer audit found no truncation at 512 tokens in either the 600-row
population or the 570-row selected cohort; the longest passage is 311 tokens.

The selected cohort is conditioned on true labels and the CPU router's success. Its
result is a conditional four-version diagnostic, not an 18-model leaderboard score.
The experiment also saves results across all 600 true-Gemini passages. Comparisons
in that secondary cohort must account for the specialist being told the true family.

## Reproduction

```sh
.venv/bin/python scripts/prepare_gemini_experiment.py --username YOUR_USERNAME
.venv/bin/kaggle kernels push -p artifacts/gemini_experiment \
  -t 7200 --accelerator NvidiaTeslaT4
```

The notebook is private and uses free Kaggle GPUs. Internet is enabled for a pinned
dependency installation and model download; training reads local files. Its clean
environment is created without `ensurepip`, using the notebook's existing pip and
its `--python` option to install into that environment.

Outputs include per-passage OOF predictions/probabilities, the selected 570-row file,
confusion matrices, class metrics, paired corrections/regressions against the CPU
baseline, configuration and source fingerprints, and runtime/GPU memory logs.
No competition submission is created. No full trained checkpoints are retained.

## Measured result, October 26, 2026

The completed private run is
[`daichongwei06/name-that-llm-gemini-deberta-run`, version 1](https://www.kaggle.com/code/daichongwei06/name-that-llm-gemini-deberta-run).
It used the fixed protocol above. The exact selected IDs were independently checked
against the original CPU OOF predictions and the official training data.

| Exact version accuracy on the same 570 entries | Correct | Accuracy |
| --- | ---: | ---: |
| CPU reference | 291 | 51.05% |
| Fine-tuned DeBERTa-v3-base | 166 | 29.12% |

This recipe loses **21.93 percentage points**. It corrects 67 CPU mistakes while
changing 192 previously correct predictions to incorrect ones; both models get 99
entries right and 212 entries wrong.

| True Gemini version | Selected rows | CPU correct | DeBERTa correct |
| --- | ---: | ---: | ---: |
| `gemini-3.1-pro-preview` | 147 | 114 (77.55%) | 36 (24.49%) |
| `gemini-3.6-flash` | 143 | 60 (41.96%) | 17 (11.89%) |
| `gemini-3.7-flash` | 143 | 68 (47.55%) | 61 (42.66%) |
| `gemini-3.8-flash` | 137 | 49 (35.77%) | 52 (37.96%) |

Across all 600 true-Gemini passages, the specialist gets 175 correct (29.17%).
Mean training cross-entropy moves from 1.3940 to 1.3801 over the three epochs,
near the four-class uniform reference of `ln(4) = 1.3863`. Predictions are also near
uniform. This suggests little learning under this initial recipe; it is not evidence
that the architecture cannot work with other training choices or more data.

All five folds complete 90 optimizer updates with zero AMP skips. Fine-tuning and
evaluation take 201.6 seconds, excluding environment setup and model download; peak
allocated GPU memory is about 5.70 GiB on one T4. Earlier environment and profiler
failures occurred before training; they did not change the model-training protocol.

Local predictions and reports are in `artifacts/gemini-deberta-results/gemini-diagnostic/`.
`predictions_570.csv` contains every requested ID, true version, CPU prediction,
DeBERTa prediction, fold, correctness flags, and confidence. Raw per-row artifacts
remain excluded from Git. No new competition submission was made.

## Training diagnostic, October 26, 2026

The near-uniform loss raised the question of whether this loop can fit at all.
`ntl.deberta_probe` calls the unchanged `_fit_predict_fold` and observes it through
wrappers: pre-clipping gradient norms by parameter group, per-update parameter movement,
learning rate, and AMP scale. A test confirms that predictions and losses are identical
with and without the wrappers. Each probe trains on rows from outer fold 0's fitting
partition only and then scores those same rows in evaluation mode. No outer validation
row is used.

Both probes ran on local CPU in float32 with the same torch 2.8.0 and transformers 4.57.1
versions as the Kaggle run. Autocast and GradScaler are disabled on CPU.

| Run | Rows | Updates | Mean training loss by epoch | Fit on the same rows afterwards |
| --- | ---: | ---: | --- | --- |
| Small subset, 40 epochs, recipe otherwise unchanged | 32 (8 per version) | 80 | near 1.38 through epoch 7, then falling to 0.449 at epoch 40 | 32/32 correct, cross-entropy 0.406 |
| Full fold-0 fitting partition, exact recipe | 480 | 90 | 1.391, 1.388, 1.369 | 226/480 (47.1%), cross-entropy 1.359 |
| Kaggle T4 float16-AMP run, fold 0 (from above) | 480 | 90 | 1.395, 1.390, 1.374 | not measured |

- Every parameter receives a finite gradient at every update, and every group moves.
  Pre-clipping gradient norms have medians of 3.2 (subset) and 2.0 (full fold), so 100%
  and 99% of updates are clipped at 1.0. With AdamW, this rescales each step's gradient
  and does not by itself stop learning.
- Over the 90 full-fold updates, encoder weights move 0.21% relative to their initial
  norm and the new classifier 0.50%. On the subset, they move 0.39% and 1.4%.
- Linear warmup starts at zero, so the first optimizer update has learning rate 0.
  This is harmless.
- The float32 run reproduces the AMP loss curve to within about 0.005 per epoch.
  Float16 autocast and loss scaling therefore do not explain the flat loss.

**Interpretation.** The loop can fit data, with observable gradients and parameter
changes. The flat loss is consistent with an optimization budget that is too small. On
32 rows, the loss stayed near uniform for about 14 updates (seven passes) before
falling. The full recipe makes three passes over 480 rows, after which it fits its own
training passages at only 47% accuracy, with near-uniform confidence (mean maximum
probability 0.27). Those predictions lean heavily to `gemini-3.1-pro-preview` (247 of
480) and rarely choose `gemini-3.6-flash` (8). This shows underfitting of the training
data. It is not evidence of a bug, and it says nothing about whether a longer schedule
would generalize to unseen prompts. A future run should fix a longer schedule in
advance, or choose it within inner folds, and confirm training-set fit before any outer
evaluation.

```sh
PYTHONPATH=src .venv/bin/python -m ntl.deberta_probe --train data/raw/train.csv \
  --baseline-oof artifacts/kaggle-version-1/cpu-cv/oof.parquet \
  --fold-manifest artifacts/kaggle-version-1/cpu-cv/folds.csv \
  --model-path models/deberta-v3-base --output-dir artifacts/deberta-probe/lr2e-5
# Full fitting partition with the exact recipe:
#   --per-class 0 --epochs 3 --output-dir artifacts/deberta-probe/full-fold0-fp32
```

The two probes took 9.0 and 11.6 minutes on a 10-core Apple CPU. Per-update records
are in each `artifacts/deberta-probe/*/probe.json` and remain excluded from Git.

Sources: [upstream model card](https://huggingface.co/microsoft/deberta-v3-base),
[pinned model configuration](https://huggingface.co/microsoft/deberta-v3-base/blob/8ccc9b6f36199bec6961081d44eb72fb3f7353f3/config.json),
[pip interpreter targeting](https://pip.pypa.io/en/stable/topics/python-option/).
