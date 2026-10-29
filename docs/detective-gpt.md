# DeTeCtive features for the GPT classes

**Question.** Do DeTeCtive's authorship embeddings separate GPT-5.5, GPT-5.6 Luna / Sol / Terra
and GPT-6 Astra better than the current features? And does adding them to the
[GPT specialist](gpt-specialist.md) improve the complete incumbent?

**References:**
- 47.01% grouped CV (the complete incumbent: blend plus Claude–GLM specialist);
- 42.30% on the eight-topic split;
- 0.429 on the public leaderboard.

On the 750 true GPT passages, the five-way balanced accuracy is:
- 50.0% for the incumbent forced among the five;
- 50.7% for the Qwen + StyleDistance GPT specialist.

That specialist, mixed into the incumbent, gained +0.26 points and was not promoted.

## The encoder

DeTeCtive (He et al., NeurIPS 2024, arXiv 2410.20964) is trained with multi-level contrastive
learning to separate texts by the model that wrote them. The checkpoints
(`heyongxin233/DeTeCtive`, revision `cefc9833290189b414a683d0775edb900c1832c1`, Apache-2.0) are
pickled PyTorch state dicts, about 499 MB each, one per training benchmark.

**The checkpoint used: `Deepfake_best.pth` alone,** chosen before any feature was computed. It
was trained to separate 27 generators across seven families:
- OpenAI's gpt-3.5-turbo and text-davinci-002 / 003;
- LLaMA, GLM-130B, FLAN-T5, OPT, BLOOM / T0, and GPT-J / NeoX.

That is the most model-level variety among the five checkpoints. None of these generators is
a GPT-5.x model, so transfer is the open question. The M4, OUTFOX and TuringBench checkpoints
are not tried; choosing among them on these folds would be a search.

**Loading, without the authors' code.** The authors' `TextEmbeddingModel` is a standard
`AutoModel` over `princeton-nlp/unsup-simcse-roberta-base`. Each checkpoint holds that
module's state dict. It is reproduced here with stock `transformers` RoBERTa classes:
- the configuration and tokenizer come from that base model at revision
  `db28710348cf9f33a2be25c505f98f0fbbbfe768`, and its weights are not used;
- the checkpoint is checked against its published SHA-256;
- it is read with `torch.load(weights_only=True)`, which refuses arbitrary pickled objects. A
  checkpoint that needs more than that is not loaded;
- it must match the RoBERTa parameters exactly;
- no `trust_remote_code` is used.

**Embedding, as the authors computed it:**
- truncate and pad every passage to 512 tokens;
- sum the last hidden states over all 512 positions, padding included;
- divide by the number of real tokens, then L2-normalize.

The sum includes padding because the authors' `masked_fill` result is discarded. This is how
the checkpoint was trained and how its reference databases were built, so it is reproduced
rather than corrected. It runs in FP32 on the CPU, and nothing is fitted to this data.

## Design

This section was written before the checkpoint was downloaded.

**Stage 1: can the embeddings separate the GPT versions?** This is diagnostic. Five-way
balanced accuracy is measured on the 750 true GPT passages. Each fold's model is fit on the
true GPT passages outside that original prompt fold (`artifacts/lora/plan.csv`):

| Predictor | Recipe |
| --- | --- |
| DeTeCtive alone | The specialist recipe: StandardScaler, logistic regression, `C = 0.005` |
| Qwen + StyleDistance | The same; the existing GPT specialist |
| Qwen + StyleDistance + DeTeCtive | The same, each block after Qwen scaled to Qwen's total variance |
| DeTeCtive nearest neighbours | The authors' retrieval: cosine similarity, `K = 5`, majority vote over the training passages; ties go to the higher summed similarity |

**Stage 2: does it improve the complete incumbent?** This is the decision. The GPT
specialist's protocol is used unchanged (`ntl.gpt_specialist`), with one feature set,
Qwen + StyleDistance + DeTeCtive:
- `p′ₖ = (1 − w) pₖ + w M qₖ` on the five GPT classes;
- the inner folds of each original prompt fold rebuild the complete incumbent and choose
  between it and `w ∈ {¼, ½, ¾, 1}`;
- ties keep the incumbent, then the smaller `w`;
- the rebuilt outer incumbent must equal the locked reference.

**Promotion rule,** unchanged:
- an unrounded gain of at least +0.5 points;
- a 95% paired prompt-bootstrap interval above zero (10,000 resamples, seed 2026).

Only a passing candidate repeats the procedure on the eight-topic split.

**Also reported:**
- the candidate against the earlier Qwen + StyleDistance GPT specialist, built the same way,
  to isolate what DeTeCtive adds;
- GPT recall and precision;
- the GPT confusion block;
- gains and losses across every class.

## Result: original prompt folds (October 28, 2026)

```sh
PYTHONPATH=src .venv/bin/python -m ntl.detective extract --train data/raw/train.csv \
  --model-dir models/detective --output artifacts/detective/detective-deepfake-train.npz
PYTHONPATH=src .venv/bin/python -m ntl.detective evaluate --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv \
  --qwen-features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --styledistance-features artifacts/claude-glm/styledistance-train.npz \
  --detective-features artifacts/detective/detective-deepfake-train.npz \
  --current-oof artifacts/qwen-blend/oof.parquet \
  --current-inner artifacts/claude-glm/current-inner/fold{0,1,2,3,4}.parquet \
  --reference artifacts/styledistance-adapt/reference-prompts.json \
  --earlier artifacts/gpt-specialist/prompts/predictions.parquet --output-dir artifacts/detective/prompts
```

**Loading checks:**
- The checkpoint matches its published SHA-256.
- Read with `weights_only=True`, it is a plain state dict of 199 float32 tensors, all
  `model.`-prefixed.
- Those tensors match a stock RoBERTa model exactly: none missing, none unexpected, every
  shape equal.

**Embedding:**
- All 2,699 passages were embedded in about 4 minutes (0.09 s each, 8 CPU threads).
- Repeating the embedding gives identical vectors.
- The rebuilt outer incumbent equals the locked reference in every fold.

### Stage 1: separating the GPT versions

Five-way balanced accuracy and recall on the 750 true GPT passages. Chance is 20%.

| Predictor | Five-way BA | GPT-5.5 | Luna | Sol | Terra | Astra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent, forced among the five | 50.0% | 61.3% | 41.3% | 36.7% | 36.7% | 74.0% |
| Qwen + StyleDistance specialist | 50.7% | 59.3% | 47.3% | 40.0% | 42.0% | 64.7% |
| Qwen + StyleDistance + DeTeCtive | 49.1% | 56.7% | 46.0% | 36.7% | 42.0% | 64.0% |
| **DeTeCtive alone** (logistic) | **25.3%** | 44.7% | 23.3% | 14.7% | 23.3% | 20.7% |
| **DeTeCtive nearest neighbours** (`K` = 5) | **24.8%** | 30.0% | 31.3% | 22.7% | 14.7% | 25.3% |

**DeTeCtive's embedding space barely separates these GPT versions:**
- either way of using it alone, it is only about 5 points above chance;
- added to Qwen + StyleDistance, it lowers that specialist from 50.7% to 49.1%.

### Stage 2: the complete system

| System | Grouped BA | Change | 95% interval | Gained / lost |
| --- | ---: | ---: | --- | --- |
| Incumbent | 0.4701 | — | — | — |
| GPT specialist on Qwen + StyleDistance + DeTeCtive | 0.4701 | 0.00 | [−0.59, +0.63] | 37 / 37 |

**Not promoted.** The change is exactly zero, and 53% of resamples are at or below zero. As
declared, the eight-topic check was not run, and nothing was packaged.

**Selection and its held-out effect:**
- Four folds chose `w = ¼`, by inner margins of +0.28 to +0.60 points. Fold 1 kept the
  incumbent.
- The held-out changes were −0.37, 0.00, −0.93, +1.11 and +0.19 points.
- In hindsight, `w = ¼` in every fold would lose 0.04 points, and every larger weight loses
  more, down to −2.04 at `w = 1`.

**Against the earlier Qwen + StyleDistance GPT specialist,** built the same way: −0.26
points (interval [−0.67, +0.15]). Adding DeTeCtive makes that specialist slightly worse, not
better.

**Recall:**
- **GPT classes:** GPT-5.5 rises from 46.0% to 47.3%, Luna from 36.7% to 38.7%, Sol from
  31.3% to 32.7% and Terra from 34.7% to 37.3%. Astra falls from 68.7% to 66.7%.
- **Other classes pay for it:** Qwen3.8 Max falls from 51.3% to 48.0%, and Kimi K3 from
  44.0% to 41.3%.
- **Why.** The specialist pulls six Qwen3.8 Max and six Kimi passages into GPT
  predictions, as the earlier specialist did.

**What this answers.** The Deepfake checkpoint, trained to separate 2022–23 generators,
transfers almost nothing to separating GPT-5.5, 5.6 Luna / Sol / Terra and GPT-6 Astra. The
GPT recall gains in stage 2 come from the Qwen + StyleDistance part of the specialist, and
that specialist alone did better (+0.26 points).

**Not tried:**
- DeTeCtive's other four checkpoints;
- the other suggested encoders: READER, DeepStylometry, and the ModernBERT
  attribution model.
