# External training data

**Question.** Do more examples of the target models, added only to training, help the
existing representation on the original prompt-grouped validation passages?

**References:**
- 47.01% grouped CV (the complete incumbent: blend plus Claude–GLM specialist);
- 42.30% on the eight-topic split;
- 0.429 on the public leaderboard.

## Source

**[Name That LLM: OpenRouter + OpenAI Generations](https://www.kaggle.com/datasets/maniceet/name-that-llm-openrouter-openai-generations)**
(Maniceet Sahay, 2026; CC BY 4.0, credit required), downloaded to
`data/external/ntl-openrouter-openai/`:
- **File hashes (SHA-256):** parquet `b205a680…8cc268`, CSV `5dd65c74…05fa4`.
- **Scope:** 7,753 single-paragraph passages over 10 competition labels.
- **Generation:** each prompt asks for "a single paragraph of about N words", with N drawn
  from the competition training set's lengths.
- **Prompt licences:** some prompt bodies come from two further CC BY 4.0 datasets
  (`lxdlxdlxd/ntl-synth-generations-v1`, `prvsiyan/prvsiyans-extended-name-that-llm-dataset`),
  which are credited too.

Kaggle's competition overview permits freely available public external data. Each dataset's
own terms still apply.

**READER's response pool** (`LeoJeshua/READER`, data README at commit `2dd0d58`) is audited
for a possible later arm but not used here. Its responses are capped at 128 tokens, use
agent-style prompts, and remain "subject to applicable source-model and API-provider terms".

## Audit

This was done before any model was fit.

| Check | Result |
| --- | --- |
| Labels | All 10 are competition labels |
| Generating-model identifiers | Recorded and matching the label for 6,223 rows: `google/gemini-3.7-flash`, `z-ai/glm-5.3`, `z-ai/glm-5.3-flash`, `moonshotai/kimi-k3`, `qwen/qwen3.8-max-0902`, `gpt-5.6-luna` / `-sol` / `-terra`. **Missing for the 1,530 `own` rows** (the publisher's direct OpenAI script, raw responses not retained). Those rows include every GPT-5.5 and GPT-6 Astra passage, and their finish reasons are not logged either |
| Row identity | Every `row_id` is the documented hash of its text; no duplicate rows or texts |
| Competition text | No external passage equals a competition passage |
| Near duplicates | Nine pairs within the external set reach word TF-IDF cosine ≥ 0.8, all answers by two GPT versions to the same simple prompt. They are kept |
| Lengths | Median 105 words (competition: 101) |
| Providers | Most GLM rows came through third-party providers (Relace, OpenInference) rather than Z.AI, so a provider mismatch is possible |
| Prompt overlap | No external prompt exceeds word TF-IDF cosine 0.25 with any competition passage, as the publisher states. Some passages do answer competition topics (up to 0.51; for example Monty Hall, the solar wind, the Rosetta Stone, Shannon entropy) |

**Verified rows by label** (the 6,223 used):

| Label | Rows |
| --- | ---: |
| GLM 5.3 Flash | 2,828 |
| GLM 5.3 | 1,839 |
| GPT-5.6 Luna | 688 |
| GPT-5.6 Sol | 389 |
| GPT-5.6 Terra | 389 |
| Kimi K3 | 63 |
| Gemini 3.7 Flash | 22 |
| Qwen3.8 Max | 5 |

There are no verified GPT-5.5 or GPT-6 Astra rows.

## Design

This section was written before any model was fit.

**Rule 1: verified identifiers only.** A row is used only if its recorded `served_model`
matches its label. The 1,530 rows without one are excluded; they could be a later
sensitivity check.

**Rule 2: topic links keep validation clean.** Each external passage is matched to its most
similar competition passage (word TF-IDF fit on both sets, English stop words removed,
sublinear term frequency).
- **The threshold** is set from the competition data alone. τ is the 95th percentile of a
  competition passage's highest similarity to any other prompt: 0.186 with the vectorizer fit
  on the verified rows (0.187 in the first look, with every external row in the vocabulary).
- **The match.** An external prompt is linked to the competition prompt with the most
  matches above τ; ties go to the higher similarity. All its passages follow that link.
- **Use.** A linked passage is used only when its competition prompt is on the training side
  of the split being fit. That holds in outer, inner and blend-weight splits alike. Unlinked
  passages may always be used.

**Rule 3: controlled weight and class balance.** In each fit, a class contributes at most as
many external passages as it has competition passages in that fitting partition. They are
taken in a fixed order: a per-class permutation, seed 2026, of the eligible rows. External
and competition passages carry equal per-row weight, and the existing class-balanced
weighting then equalizes the 18 classes.

**Evaluation.** Every reported score is on the original prompt-grouped validation passages:
the five folds of `artifacts/lora/plan.csv` and their inner folds. External passages never
appear in any validation set.

**Stage A: the CPU component** (runs locally; this is the decision in this document). The
CPU model (char and word TF-IDF plus style features, class-balanced logistic regression,
`C = 4`) gains the eligible external passages in every fit:
- **The blend weight.** The CPU–Qwen blend's weight is chosen exactly as before, with the
  augmented CPU model refit inside its own inner splits.
- **Unchanged.** The Qwen head and the Claude–GLM specialist are unchanged and fit on
  competition passages only, because they need Qwen features of the external passages.
- **The specialist's `s`.** It is re-chosen by the inner folds, as the incumbent's is.
- **Faithfulness check.** With no external rows, the augmented code must reproduce the saved
  inner caches and the blend's outer predictions exactly.

**Stage B: the Qwen-based components** (needs the external passages' Qwen features). These
must come from the same NF4 extraction on a Kaggle T4 (`ntl.ensemble extract`), so that
they match the competition passages' features. Then the Qwen head and the Claude–GLM
specialist gain the same eligible rows. This is a follow-up.

**Promotion rule,** unchanged:
- an unrounded gain of at least +0.5 points;
- a 95% paired prompt-bootstrap interval above zero (10,000 resamples, seed 2026).

Only a passing candidate repeats the procedure on the eight-topic split.

**Reported:**
- overall balanced accuracy;
- recall and precision for all 18 classes, especially the GPT, GLM, Qwen and Kimi classes;
- gains and losses by class;
- the selected blend weights and `s`;
- the number of external passages used per fit and class.

## Result: stage A, original prompt folds (October 28, 2026)

```sh
PYTHONPATH=src .venv/bin/python -m ntl.external_data pool --train data/raw/train.csv \
  --external data/external/ntl-openrouter-openai/ntl_openrouter_openai_generations.parquet \
  --output-dir artifacts/external-data
# for F in 0..4: the augmented inner caches and outer blend
PYTHONPATH=src .venv/bin/python -m ntl.external_data inner --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv \
  --qwen-features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --pool artifacts/external-data/pool.parquet --fold F \
  --output artifacts/external-data/augmented/inner-foldF.parquet
PYTHONPATH=src .venv/bin/python -m ntl.external_data outer ...same arguments... \
  --output artifacts/external-data/augmented/outer-foldF.parquet
PYTHONPATH=src .venv/bin/python -m ntl.external_data evaluate --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv ... --current-oof artifacts/external-data/augmented/oof.parquet \
  --current-inner artifacts/external-data/augmented/inner-fold{0,1,2,3,4}.parquet \
  --reference artifacts/styledistance-adapt/reference-prompts.json \
  --output-dir artifacts/external-data/prompts
```

**The pool:**
- 6,223 verified rows are used, and 1,530 unverified rows are excluded.
- τ = 0.186. 187 external passages exceed it, which links 101 external prompts (363
  passages) to 53 competition prompts.
- Each outer CPU fit adds about 690 external passages to about 2,160 competition passages:
  - 120 per class for the two GLMs and three GPT-5.6 versions;
  - all 63 Kimi, 22 Gemini 3.7 and 5 Qwen3.8 Max passages.
- Each blend-weight fit adds about 488.

**Faithfulness check.** With no external rows, the augmented code reproduces the saved fold-0
inner caches and the blend's fold-0 outer predictions exactly: maximum difference 0, with
the same blend weights.

| System | Grouped BA | Change | 95% interval | Gained / lost |
| --- | ---: | ---: | --- | --- |
| Incumbent | 0.4701 | — | — | — |
| External passages in the CPU model | 0.4593 | **−1.07** | [−2.19, +0.08] | 99 / 128 |

**Not promoted.**
- **It loses in every fold:** −2.59, −0.37, −1.11, −0.93 and −0.37 points.
- **Almost every resample agrees:** 97% put the change at or below zero.
- **The base CPU–Qwen blend loses too,** from 0.4541 to 0.4445.
- **The inner folds lowered the CPU model's weight.** They moved the blend weight to Qwen
  `w` = ½ in four folds (¼ before), and the specialist's `s` to ½–¾ (¼–½ before).

**The CPU model on its own,** recovered as (blend − w · Qwen) / (1 − w), since the Qwen head
is unchanged:

| Class (* gained external rows) | Recall | Precision | Predicted |
| --- | --- | --- | --- |
| * GLM 5.3 | 14.0% → 27.3% | 15.2% → 16.2% | 138 → 253 |
| * GLM 5.3 Flash | 15.3% → 32.0% | 21.5% → 19.2% | 107 → 250 |
| * GPT-5.6 Sol | 19.3% → 30.0% | 29.9% → 22.5% | 97 → 200 |
| * GPT-5.6 Terra | 21.3% → 29.3% | 27.4% → 22.9% | 117 → 192 |
| * GPT-5.6 Luna | 32.0% → 38.7% | 30.6% → 24.3% | 157 → 239 |
| * Gemini 3.7 Flash | 45.3% → 53.3% | 37.8% → 35.6% | 180 → 225 |
| * Qwen3.8 Max | 52.7% → 42.7% | 42.7% → 44.4% | 185 → 144 |
| * Kimi K3 | 26.7% → 27.3% | 30.1% → 27.9% | 133 → 147 |
| GPT-5.5 | 40.7% → 25.3% | 35.7% → 36.2% | 171 → 105 |
| GPT-6 Astra | 59.3% → 36.7% | 47.1% → 54.5% | 189 → 101 |
| Claude Opus 5 | 58.0% → 36.0% | 37.8% → 53.5% | 230 → 101 |
| Claude Sonnet 5 | 38.0% → 28.7% | 32.9% → 42.6% | 173 → 101 |
| Claude Fable 5 | 9.4% → 4.7% | 18.4% → 14.9% | 76 → 47 |

Its balanced accuracy falls from 36.3% to 33.9%.

**In the complete system** (recall change in points):
- **Gains:** Sol +8.0, GLM +2.7, GLM Flash +1.3, Terra +1.3.
- **Losses:** GPT-5.5 −6.7, Astra −6.0, Qwen3.8 Max −4.7, Sonnet 5 −4.0, Gemini 3.6 Flash
  −4.0, Gemini 3.1 Pro −3.3, Gemini 3.8 Flash −3.3.
- **Kimi K3:** +0.7.

**What this answers:**
- **Only 8 of the 18 classes got more examples,** and the model expanded around those
  classes. Their predictions roughly doubled, their recall rose and their precision fell.
- **The other classes lost recall,** including the GPT versions and Claude classes it was
  meant to separate.
- **The controls did not stop this.** The per-class cap and the class-balanced weighting
  hold each class's total weight equal, but not its coverage of the feature space.
- **Qwen3.8 Max and Kimi, with 5 and 63 external rows, gain nothing.**
- **The doubled classes do gain in the CPU model:** GLM recall roughly doubles, so the
  external passages carry signal for their own classes. But the cost to the 10 classes
  without external data outweighs it.

**Not run:**
- **Stage B** (the Qwen head and the specialist with external passages) needs the external
  passages' Qwen features from a Kaggle run. It would meet the same imbalance unless that
  were addressed first.
- **The 1,530 unverified rows** (all GPT-5.5 and GPT-6 Astra) and **READER's pool** were not
  tried.
