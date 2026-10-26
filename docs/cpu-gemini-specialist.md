# Four-version CPU specialist and StyleDistance blend

These experiments ask whether the CPU reference improves when its classification task is
narrowed to the four Gemini versions, and whether frozen StyleDistance scores add to it.
They use the same 600 true-Gemini passages, original five prompt folds, and 570 correctly
routed reporting IDs as [the DeBERTa](gemini-deberta.md) and
[frozen-embedding](frozen-gemini.md) diagnostics.

## Protocol

- **Control: four-version CPU specialist.** The 18-class reference's exact character
  3–5-gram and word 1–2-gram TF-IDF, stylometry block, and class-weighted logistic
  regression (`C=4`) are refit in each outer fold on the 480 true-Gemini fitting passages
  only, then predict the 120 held-out passages. Only the label set differs.
- **Blend.** `(1 - w) * CPU probabilities + w * StyleDistance probabilities`, using the
  cached StyleDistance embeddings and the frozen experiment's scaled L2 logistic readout.
  Within each outer fitting partition, the frozen experiment's three inner prompt folds
  choose readout `C` from `[0.001, 0.01, 0.1, 1, 10]` and `w` from `0, 0.1, …, 1`
  jointly, by mean inner balanced accuracy. Ties choose smaller `w`, then smaller `C`;
  `w = 0` is the control. Each outer partition is predicted once with the chosen pair.
- The control's outer predictions are the blend's CPU component, so the two differ only
  by the StyleDistance term. Every inner and outer fit is fresh and uses fitting rows only.
- Paired uncertainty resamples whole prompts (10,000 bootstrap draws).
- **End-to-end score.** The saved 18-class router's predictions on all 2,699 rows are kept,
  except that each of the 570 true-Gemini rows routed to Gemini receives the specialist's
  version. The 94 non-Gemini passages routed to Gemini are scored incorrect, as they
  would be under any Gemini version; the 30 true-Gemini passages routed elsewhere stay
  wrong. Balanced accuracy and accuracy over all 2,699 rows are therefore exact for this
  two-stage system. Each extra correct Gemini row adds 1/2,700 to balanced accuracy.

```sh
PYTHONPATH=src .venv/bin/python -m ntl.cpu_gemini --train data/raw/train.csv \
  --baseline-oof artifacts/kaggle-version-1/cpu-cv/oof.parquet \
  --fold-manifest artifacts/kaggle-version-1/cpu-cv/folds.csv \
  --features artifacts/frozen-gemini-results/features/styledistance.npz \
  --output-dir artifacts/cpu-gemini-specialists
```

The run takes about 32 seconds on CPU. Before it, a local rerun of the frozen
StyleDistance evaluation on the cached features reproduced the Kaggle result exactly:
the same per-fold `C`, 253/570 and 266/600, and identical predictions.

## Measured results, October 26, 2026

Local Python 3.12.13, scikit-learn 1.7.2. The reference predictions come from the Kaggle
CPU run (scikit-learn 1.6.1), so the control also differs from it by environment.

| Exact version accuracy on the same 570 entries | Correct | Accuracy | Change vs CPU reference, 95% prompt-bootstrap interval |
| --- | ---: | ---: | --- |
| 18-class CPU reference | 291 | 51.05% | — |
| Four-version CPU specialist (control) | 296 | 51.93% | +0.88 pp [−1.57, +3.32] |
| Specialist + StyleDistance blend | 295 | 51.75% | +0.70 pp [−1.98, +3.45] |

| True Gemini version | Selected rows | CPU reference | Specialist | Blend |
| --- | ---: | ---: | ---: | ---: |
| `gemini-3.1-pro-preview` | 147 | 114 | 118 | 112 |
| `gemini-3.6-flash` | 143 | 60 | 62 | 60 |
| `gemini-3.7-flash` | 143 | 68 | 61 | 65 |
| `gemini-3.8-flash` | 137 | 49 | 55 | 58 |

**Specialization does not measurably help.** The specialist corrects 28 reference
errors and loses 23 reference successes; 91% of entries keep the same correctness.
The +5 entries are well inside prompt-resampling noise.

**The blend adds nothing to the control.** Against the specialist it gains 19 entries
and loses 20 (−0.18 pp, interval [−2.32, +2.09]). Inner folds selected
`w = [0.4, 0.1, 0.3, 0.1, 0.1]` and `C = [0.001, 0.01, 0.001, 1, 1]` in outer-fold
order. In every fold, the winning
inner candidate beat the control's inner score by 0.4–3.7 points, but outer gains
averaged zero, consistent with selecting the best of 55 noisy candidates.

A post-hoc curve over fixed weights confirms this is not a selection failure. It reuses
outer labels, so it is diagnostic only and is not an estimate. Blending the control with
the frozen experiment's own StyleDistance predictions gives 296, 295, 291, 286 … 253
correct as `w` rises from 0 to 1 in steps of 0.1; no weight beats the control. The
models' errors do differ: StyleDistance is right on 78 entries where the specialist is
wrong, and either model is right on 374/570. But averaged probabilities do not identify
which model to trust.

| End-to-end, all 2,699 official rows | Correct | Balanced accuracy |
| --- | ---: | ---: |
| 18-class CPU reference (Kaggle OOF) | 971 | 0.359667 |
| Router + four-version CPU specialist | 976 | 0.361519 |
| Router + specialist/StyleDistance blend | 975 | 0.361148 |

The reference value matches the recorded Kaggle grouped CV exactly. Neither change is
distinguishable from noise, and neither candidate is worth an extra inference stage.

On all 600 true-Gemini passages, the specialist gets 310 right and the blend 310. That
cohort is not a fair comparison with the reference: the specialist is told the family,
and 14 of its gains are among the 30 passages the router sent elsewhere.

Outputs are in `artifacts/cpu-gemini-specialists/`: `comparison.json` (paired and
end-to-end results), `inner_cv.json`, per-variant `oof.parquet` and
`selected_predictions.csv`, class reports, and `end_to_end_predictions.parquet`. These
remain excluded from Git. No competition submission was made.
