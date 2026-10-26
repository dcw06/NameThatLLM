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

Sources: [upstream model card](https://huggingface.co/microsoft/deberta-v3-base),
[pinned model configuration](https://huggingface.co/microsoft/deberta-v3-base/blob/8ccc9b6f36199bec6961081d44eb72fb3f7353f3/config.json),
[pip interpreter targeting](https://pip.pypa.io/en/stable/topics/python-option/).
