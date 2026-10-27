# Frozen Qwen3-8B-Base and the CPU–Qwen blend

This is the validation behind the second submission candidate: the CPU classifier and
a frozen Qwen3-8B-Base feature classifier both predict all 18 models, and a weighted
average combines their scores. Neither the Gemini specialists nor any external data
are included.

## Protocol

**Frozen features (step 1).** The existing recipe is pinned `Qwen/Qwen3-8B-Base` at
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`:
- the first 24 decoder blocks, in NF4 with float16 compute;
- no language-model head and no cache;
- masked token means of blocks 7 and 22, concatenated into 8,192 values;
- a 512-token limit, batch size 2 and right padding.

Extraction uses no labels and handles each passage independently, so every official
passage is embedded once on a Kaggle T4. The resulting archive pins the exact ids, the
`(id, text)` pairs, the array bytes and the training fingerprint.

**Qwen head.** A fresh `StandardScaler` and `LogisticRegression(C=0.005)` are fit on
each training partition of the original five prompt folds. This gives one held-out
18-class score vector per passage.

**Blend (step 2).** `(1 - w) * CPU + w * Qwen`. Rows are aligned by passage id and
columns by the canonical label order, and both models score every passage. Within each
outer training partition, three inner prompt folds refit **both** classifiers and choose
`w` from `{0, 0.25, 0.5, 0.75, 1}` by mean balanced accuracy, with ties going to the
smaller `w`. Each outer partition is then predicted once.

**Gate (step 3).** This was written into `ntl.ensemble.BlendConfig` before the first
blend result. The blend must:
- beat the recorded 0.359667 by at least one point;
- beat the same-environment CPU in at least four of five folds;
- select `w = 0` in at most one fold.

```sh
.venv/bin/python scripts/prepare_qwen_validation.py --username YOUR_USERNAME
.venv/bin/kaggle kernels push -p artifacts/qwen_validation -t 7200 --accelerator NvidiaTeslaT4
PYTHONPATH=src .venv/bin/python -m ntl.ensemble blend-cv --train data/raw/train.csv \
  --fold-manifest artifacts/kaggle-version-1/cpu-cv/folds.csv \
  --features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --reference-oof artifacts/kaggle-version-1/cpu-cv/oof.parquet --output-dir artifacts/qwen-blend
```

## Measured results, October 26, 2026

Feature extraction ran in private notebook
[`name-that-llm-qwen-frozen-validation`, version 1](https://www.kaggle.com/code/daichongwei06/name-that-llm-qwen-frozen-validation).
Embedding 2,699 passages took 509 s on one T4, with a peak of 5.25 GB of allocated GPU
memory. No passage was truncated; the longest has 381 tokens. The blend ran locally
on the cached features in 246 s.

| All 2,699 rows, original five prompt folds | Balanced accuracy | Folds 0–4 |
| --- | ---: | --- |
| Recorded CPU reference (submitted; scikit-learn 1.6.1) | 0.3597 | .380 .354 .369 .337 .359 |
| Same-environment CPU refit (scikit-learn 1.7.2) | 0.3634 | .380 .363 .372 .343 .359 |
| Frozen Qwen3-8B-Base head | 0.4386 | .444 .435 .452 .437 .424 |
| **Nested CPU–Qwen blend** | **0.4541** | .476 .426 .487 .430 .452 |

Paired changes in balanced accuracy, with 95% intervals from 10,000 prompt-resampling
bootstrap draws:

| Comparison | Change | Interval | Rows gained / lost |
| --- | ---: | --- | ---: |
| Blend vs recorded reference | +9.45 pp | [+7.67, +11.22] | 420 / 165 |
| Blend vs same-environment CPU | +9.08 pp | [+7.37, +10.78] | 406 / 161 |
| Qwen vs same-environment CPU | +7.52 pp | [+5.37, +9.60] | 524 / 321 |
| Same-environment CPU vs recorded reference | +0.37 pp | [−0.07, +0.81] | 22 / 12 |

The inner folds selected `w = [0.25, 0.25, 0.25, 0.25, 0.5]`. **The gate passes**: the
gain is +9.45 points, the blend beats the CPU in 5 of 5 folds, and no fold selected
zero. The blend raises recall over the recorded CPU for 17 of the 18 models; only
`gemini-3.7-flash` falls (45.3% → 44.7%). Family routing rises from 70.0% to 75.7%.

| Model | Recorded CPU | Qwen | Blend |
| --- | ---: | ---: | ---: |
| `gpt-5.5` | 40.0 | 40.0 | 47.3 |
| `gpt-5.6-luna` | 31.3 | 36.0 | 37.3 |
| `gpt-5.6-sol` | 18.7 | 33.3 | 31.3 |
| `gpt-5.6-terra` | 20.7 | 36.7 | 34.7 |
| `gpt-6-astra` | 60.0 | 63.3 | 70.0 |
| `gemini-3.1-pro-preview` | 76.0 | 74.7 | 80.0 |
| `gemini-3.6-flash` | 40.0 | 46.7 | 50.0 |
| `gemini-3.7-flash` | 45.3 | 42.7 | 44.7 |
| `gemini-3.8-flash` | 32.7 | 48.0 | 48.0 |
| `claude-fable-5` | 10.1 | 22.1 | 12.1 |
| `claude-opus-4-6` | 46.7 | 56.0 | 60.0 |
| `claude-opus-4-8` | 23.3 | 38.0 | 37.3 |
| `claude-opus-5` | 58.7 | 68.7 | 72.0 |
| `claude-sonnet-5` | 36.7 | 44.0 | 46.7 |
| `qwen3.8-max` | 50.7 | 44.0 | 51.3 |
| `glm-5.3` | 14.7 | 26.7 | 24.0 |
| `glm-5.3-flash` | 16.0 | 25.3 | 25.3 |
| `kimi-k3` | 26.0 | 43.3 | 45.3 |

**Where the gain comes from.** Most of it comes from the Qwen features. The blend adds
about 1.6 points over Qwen alone, and only three of the five folds show that extra gain.
Of the +9.45 points over the submitted model, about 0.37 come from the scikit-learn
version rather than from Qwen. The submission therefore builds its CPU component with
the validated scikit-learn 1.7.2.

**Qwen-only precision.** Kaggle's identical Qwen-only head scored 0.43823, against
0.43860 locally. The difference is one passage near a tie, caused by floating-point
differences between linear-algebra libraries (Linux OpenBLAS versus macOS Accelerate).
Read the Qwen-only score as about 0.438 ± 0.001. Substituting Kaggle's Qwen
probabilities leaves the blend unchanged at 0.454119.

## Independent audit

Four independent auditors examined the code and artifacts. Their lenses were extraction
and alignment, fold and selection leakage, a recomputation without the project package,
and negative controls. None reported a medium or high finding.
- **Recomputation.** Every reported balanced accuracy, per-fold value and recall
  reproduces exactly with plain numpy/scikit-learn. A rerun of `evaluate_blend` with
  every fit and predict call recorded reproduced `oof.parquet` bit for bit. All 40 fits
  used only their own training rows.
- **Negative controls.** Permuting training labels gives 0.049–0.056 balanced accuracy
  (chance is 0.056). Shifting the features one row out of alignment gives 0.055.
- **Prompts and duplicates.** No prompt spans two folds, and no near-duplicate passages
  exist (the highest cross-fold text cosine is 0.32). Excluding two topic-twin prompt
  pairs moves the Qwen score by −0.0009.
- **Extraction.** Each feature vector is independent of its batch partner. The
  notebook's embedded source is byte-identical to the repository, and token-length
  statistics reproduce exactly with the local tokenizer.
- **Follow-up.** `load_archive` now also checks batch size and padding side, because
  both affect float16 numerics slightly.

## Final recipe

The validated selection procedure was applied once to all 2,699 rows: three inner
prompt folds, with seed offset 5 so that it differs from outer folds 0–4. It selected
**`w = 0.25`**. The mean inner balanced accuracies were 0.354, 0.437, 0.430, 0.422 and
0.424 for `w` = 0, 0.25, 0.5, 0.75 and 1. `artifacts/qwen-blend/recipe.json` records the
weight, both model configurations, the backbone pin, the feature options, the training
fingerprint and this validation summary.

```sh
PYTHONPATH=src .venv/bin/python -m ntl.ensemble freeze --train data/raw/train.csv \
  --features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --validation artifacts/qwen-blend/comparison.json --output artifacts/qwen-blend/recipe.json
```

## Offline submission packaging

`scripts/prepare_blend_submission.py` builds three private notebooks:

| Notebook | Settings | Output |
| --- | --- | --- |
| [`name-that-llm-qwen-model-input`](https://www.kaggle.com/code/daichongwei06/name-that-llm-qwen-model-input) v1 | CPU, Internet on | The pinned snapshot: 5 safetensors shards (16.4 GB), tokenizer, configuration and `snapshot_manifest.json` |
| [`name-that-llm-neural-wheels`](https://www.kaggle.com/code/daichongwei06/name-that-llm-neural-wheels) v1 | CPU, Internet on | Hash-locked wheels for Linux x86_64, CPython 3.13: 55 wheels, 4.1 GB |
| [`name-that-llm-qwen-blend-submission`](https://www.kaggle.com/code/daichongwei06/name-that-llm-qwen-blend-submission) | T4, **Internet off** | `submission.parquet`, `rehearsal.json`, `blend-model/fit.json` |

**What the submission notebook does.** It attaches both input notebooks as pinned
versions and installs the runtime with `pip --no-index --require-hashes` into `/tmp`.
It then:
- runs the NF4 probe;
- writes the embedded, hash-checked recipe;
- extracts frozen features for every training passage;
- refits the CPU pipeline and the Qwen head on all 2,699 rows, after checking the
  training fingerprint;
- scores the current `test.csv` with `w = 0.25` and validates the submission schema
  and id order.

**Rehearsal.** When the visible test has fewer than 1,000 rows, the notebook also runs
the identical predict path on 4,000 training passages and records the time. The hidden
scored rerun (about 4,000 rows) skips this step.

**Pre-flight checks before spending GPU time:**
- **Local run of every cell.** All notebook cells were executed in order against a
  local tree that mimics Kaggle's observed mounts. That tree used the real data, the
  real snapshot manifest and configuration, sparse placeholders for the shards and a
  `.cache` decoy. Only the pip installation, the CUDA probe and GPU feature extraction
  were stubbed; extraction returned the validated Kaggle features.
- **Discovery.** It selected the neural wheelhouse and the single real snapshot.
- **Fit, predict and rehearsal.** The CLI fit and predict ran. The 3-row submission and
  the 4,000-row rehearsal both passed the submission validator. The refit blend labels
  its own training passages with accuracy 1.0, which confirms the label alignment.
- **Wheelhouse match.** The real wheelhouse targets CPython 3.13 on Linux x86_64 (the
  GPU image runs 3.13.15) and its lock hash matches the notebook. Its 55 wheels are
  exactly the packages and versions that the online install on the GPU image used.

**Offline run, version 1 (T4, Internet off), October 27, 2026 UTC.** The run completed.

| Stage | Elapsed |
| --- | ---: |
| Offline install of all 55 pinned wheels, then the version check | 12–153 s |
| Discovery and NF4 probe | 153–164 s |
| Fit: extraction of 2,699 passages (527 s) and both refits | 596 s |
| Predict and validate the 3-row visible test | to 792 s |
| 4,000-row rehearsal of the predict path | 728 s |

- **Scored-rerun estimate.** For about 4,000 hidden rows, the scored rerun should take
  about 25 minutes: setup, fit and a 4,000-row predict, far below the nine-hour limit.
- **Identical features.** The training features extracted offline are bit-identical to
  the validated archive (maximum absolute difference 0.0). The submitted model is
  therefore the validated recipe refit on exactly the validated inputs.
- **Valid submission.** The 3-row `submission.parquet` passed the schema, label and
  id-order checks in Kaggle and again locally.

## Submission result

Kaggle submission **57685450** was created from saved version 1 on October 27, 2026 UTC.
It completed with public balanced accuracy **0.421**. The CPU baseline (submission
57676158) scored 0.360.

| System | Grouped CV | Public leaderboard | Gap |
| --- | ---: | ---: | ---: |
| CPU baseline | 0.3597 | 0.360 | +0.000 |
| Qwen only, `w = 1` (diagnostic, submission 57688014) | 0.4386 | 0.399 | −0.040 |
| CPU–Qwen blend, `w = 0.25` | 0.4541 | 0.421 | −0.033 |

**The Qwen-only diagnostic.** It changes only the blend weight: the notebook and fitted
components are identical, and the training features are bit-identical. It scores 0.399.
The CPU therefore adds 2.2 points on the public leaderboard, at least as much as the
1.5 points it adds in cross-validation. The shortfall against cross-validation comes
from the Qwen component: 3.9 points for Qwen alone, against none for the CPU. The
topic-grouped check below rules out topic shift as the cause. No leaderboard weight
search was run.

**Interpretation.**
- **Gain.** On the public leaderboard the blend gains 6.1 points over the CPU baseline,
  about two thirds of the validated 9.45.
- **CV gap.** The CPU model's cross-validation predicted its public score almost
  exactly; the blend's did not.
- **Possible causes.** The public score covers an unknown subset of the hidden test, so
  part of the gap may be sampling noise. A shift between training prompts and
  hidden-test topics is another possibility: the audit found the frozen Qwen features
  dominated by prompt topic, while the CPU's character, word and style features are not.
  One leaderboard number cannot separate these explanations.
- **Final standings.** The private score is unavailable until the competition ends.

## Topic-grouped validation

**Question.** Is the frozen Qwen head more sensitive to topic than the CPU classifier?
If so, that would explain why the blend lost more against its cross-validation score
than the CPU did.

**Method.** `ntl.topic_cv` reruns the identical nested blend under splits of increasing
strictness.
- **Topics.** The data have no prompt text, so each prompt's topic is the content-word
  TF-IDF of its 18 responses. No labels are used. Prompts are grouped with k-means on
  60 LSA components.
- **Holding out.** Whole groups are held out in both the outer folds and the inner
  weight-selection folds.
- **Coarsest level (8 groups).** The groups are subject areas: history and scribes, mind
  and self, sound and senses, Earth and chemistry, mathematics, society, the cosmos, and
  cell biology.
- **Topic distance.** "Nearest similarity" is the median, over prompts, of the TF-IDF
  cosine between a held-out prompt and its most similar training prompt.

```sh
PYTHONPATH=src .venv/bin/python -m ntl.topic_cv --train data/raw/train.csv \
  --features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --output-dir artifacts/topic-cv --k 30 15 8
```

| Split | Held-out groups | Nearest similarity | CPU | Qwen | Blend | Qwen − CPU | Blend − CPU (95% interval) | Selected `w` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Random rows | 2,699 | 1.000 | 0.184 | 0.262 | 0.263 | +7.8 | +7.9 [+6.1, +9.6] | 1, .75, .75, 1, 1 |
| Prompts (original) | 150 | 0.181 | 0.363 | 0.439 | 0.454 | +7.5 | +9.1 [+7.4, +10.8] | .25, .25, .25, .25, .5 |
| 30 topics | 30 | 0.155 | 0.350 | 0.429 | 0.437 | +7.9 | +8.6 [+6.7, +10.7] | .5, .25, .25, .25, .25 |
| 15 topics | 15 | 0.155 | 0.353 | 0.428 | 0.446 | +7.5 | +9.3 [+6.7, +11.9] | .25, .25, .25, .5, .25 |
| 8 subject areas | 8 | 0.147 | 0.337 | 0.412 | 0.416 | +7.5 | +7.9 [+4.8, +11.8] | .5, .5, .25, 1, .5 |

The prompt split reproduces the main experiment exactly.

**Qwen's advantage does not shrink with topic distance.** Holding out whole subject
areas costs every system 2.7–3.9 points. Qwen keeps a 7.5–7.9-point lead over the
CPU at every level. The analysis by topic distance, within the prompt folds, agrees:

| Held-out prompts by nearest-training similarity | CPU | Qwen | Blend |
| --- | ---: | ---: | ---: |
| Least similar third (median 0.143) | 0.333 | 0.391 | 0.409 |
| Middle third (0.181) | 0.363 | 0.459 | 0.471 |
| Most similar third (0.252) | 0.394 | 0.466 | 0.483 |

All three systems do better on familiar topics. Across prompts, however, the correlation
between topic similarity and Qwen's gain over the CPU is essentially zero (Spearman
0.008; 0.036 for the blend's gain). Topic shift therefore does not explain why the
blend fell further below its cross-validation score than the CPU did. Under the
strictest split the CPU also loses 2.7 points, whereas on the leaderboard it lost
nothing.

**Same-prompt passages actively mislead both models.** Random row splits are not
optimistic here; they are far worse (0.18 CPU, 0.26 Qwen). Each prompt has exactly one
response per model, so the other responses to the same prompt are guaranteed to carry
other labels, and topic similarity pulls predictions toward them. The CPU's leaderboard
score (0.360) matches its prompt-grouped score, not this collapse. That indicates the
hidden test uses new prompts and that prompt-grouped validation is the right protocol.

**What the blend adds over Qwen alone is fragile.** It is +1.5 points with prompt
folds but +0.4 with subject-area folds, where one fold is worse than Qwen alone. Under
stronger shift, the inner folds also move the weight toward Qwen.

**Remaining explanations for the leaderboard gap.**
- Sampling noise in the public subset.
- A shift in the hidden test other than topic, such as length or formatting.

Neither can be tested without more leaderboard evidence. Adapting the neural
representation with QLoRA did not help either; see [the QLoRA record](lora.md).

Raw OOF scores, features and reports are excluded from Git, as are the audit scripts in
the session scratch directory.
