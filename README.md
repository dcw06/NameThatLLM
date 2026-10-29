# Name That LLM

Reproducible setup for [Kaggle Name That LLM](https://www.kaggle.com/competitions/name-that-llm).
Local development uses Python 3.12. Training and scoring target free Kaggle NVIDIA GPUs.

The selected initial neural backbone is **Qwen/Qwen3-8B-Base**, pinned to revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. The first executable neural experiment
uses the first 24 decoder blocks with frozen hidden-state features and a linear classifier.
It removes the unused language-model output head. NF4 is an explicit memory-saving
configuration that needs its own validation. This is an initial choice based on public
competition evidence, not a result of our own model comparison.

Read [the selection evidence](docs/model-selection.md), [Kaggle workflow](docs/kaggle.md),
and [experiment protocol](docs/experiments.md). For a first API submission without
preparing neural weights or wheel datasets, see [the API guide](docs/kaggle-api.md).
The [QLoRA experiment](docs/lora.md) adapts Qwen3-8B-Base with nested prompt-grouped
selection and an adapters-disabled control.
The [DeBERTa Gemini-version diagnostic](docs/gemini-deberta.md) records a completed
five-fold experiment on the 570 correctly routed Gemini passages.
The [frozen embedding comparison](docs/frozen-gemini.md) evaluates StyleDistance and
Qwen3-Embedding-0.6B with logistic regression and nested prompt-grouped validation.
The [four-version CPU specialist](docs/cpu-gemini-specialist.md) tests task specialization
and a nested StyleDistance blend, including end-to-end scores on all 2,699 rows.
The [CPU–Qwen blend](docs/qwen-blend.md) validates frozen Qwen3-8B-Base features on all
18 classes, a nested score blend, its audit, and the offline submission notebooks.
The [Claude–GLM specialist](docs/claude-glm.md) redistributes the blend's probability on the
five Claude and two GLM classes, using frozen Qwen plus StyleDistance features.
The [18-class StyleDistance head](docs/styledistance-head.md) tests whether that signal also
helps the other classes, blended after the specialist.
The [Gemini pair specialist](docs/gemini-pair.md) tests whether training on Gemini 3.7 and
3.8 Flash alone recovers the system's errors between the two.
The [adapted StyleDistance experiment](docs/styledistance-adapt.md) fine-tunes StyleDistance's
top four layers on the Claude and GLM passages and mixes it into the Claude–GLM specialist.

## Local setup

```sh
make setup
make doctor
make verify-local
make smoke
```

`uv` must be installed. The setup uses a project-local managed Python and environment;
it does not alter the system Python. Exact direct and transitive dependencies live in
`uv.lock`. The default install has CPU tools and tests; PyTorch/model weights are optional.
`make neural-setup` adds the pinned local neural stack. CUDA-only bitsandbytes is a
separate `gpu` extra and is not installed on macOS.

The smoke dataset is entirely synthetic and tests execution and artifact contracts.
Its score does not estimate competition performance.

## Competition data

Accept the rules yourself using your Kaggle account, then download the official files
into `data/raw/`. Do not put API tokens in chat, tracked files, or notebooks.
The download helper uses an existing Kaggle credential configuration and never accepts rules:

```sh
uv run --extra kaggle python scripts/download_data.py
make check-data
make cv
make train
make predict
```

You may also use Kaggle's Download All button and extract `train.csv`, `test.csv`, and
`sample_submission.parquet` directly into `data/raw/`. The hidden test is available only
in the scoring rerun. Visible test rows are examples; all code accepts arbitrary row counts.

## Kaggle notebooks

Generated notebooks embed this package's source so a scored run does not need internet
or a separate source dataset. Regenerate with `make notebooks` after code changes.
See [the exact upload and model-input steps](docs/kaggle.md).

Two online preparation notebooks create hash-locked dependency wheels and download the
pinned public model snapshot. The CPU and GPU notebooks run in isolated environments
using those offline inputs, preventing conflicts with Kaggle's preinstalled packages.
No model weights, competition files, paid services, or account actions are needed for local
setup and the synthetic smoke run.

## Artifacts and validation

The baseline writes fold manifests, OOF predictions, balanced accuracy, class recalls,
confusion matrices, fitted models, and metadata. Train/validation prompts stay disjoint;
each fold fits its own vocabulary, scalers, and classifier. Duplicate IDs are rejected;
identical passages are retained and reported for ambiguity and possible leakage.
Neural artifact metadata also records the immutable
backbone revision, quantization, truncation, readout layers, and token limit.

Raw data, model files, caches, credentials, and generated predictions are ignored by Git.
Load serialized artifacts only from this project or a trusted source: joblib uses pickle.

## Status

The first official-data CPU baseline has completed prompt-grouped validation: 0.363368
locally and 0.359667 on Kaggle using the same fold assignments. See
[the first-run record](docs/first-run.md) for environments and submission details.
The first Kaggle submission completed successfully with public balanced accuracy **0.360**.
On the fixed 570-row Gemini-version diagnostic, CPU accuracy is 51.05%; frozen
StyleDistance reaches 44.39%, frozen Qwen3-Embedding-0.6B 32.63%, and the earlier
fine-tuned DeBERTa recipe 29.12%. These conditional diagnostic scores are separate
from the full competition score. A four-version CPU specialist reaches 51.93% and its
nested StyleDistance blend 51.75%. Both are within prompt-resampling noise of the
reference, and end-to-end grouped balanced accuracy moves only from 0.359667 to 0.361519
and 0.361148. A training probe shows that the DeBERTa loop fits a small subset and
reproduces the flat loss without AMP: the recipe underfits rather than failing to train.
Frozen Qwen3-8B-Base features with an 18-class head reach 0.4386 grouped balanced
accuracy. The nested CPU–Qwen blend (`w = 0.25`) reaches 0.4541, +9.45 points over the
recorded CPU reference across all five folds. Extraction takes 509 s for 2,699 passages
on one T4 at 5.25 GB of peak memory. Its offline submission (57685450) scored **0.421**
public balanced accuracy, against 0.360 for the CPU baseline; the cross-validation gap is
discussed in [the blend record](docs/qwen-blend.md#submission-result). A
[topic-grouped check](docs/qwen-blend.md#topic-grouped-validation) finds Qwen's 7.5-point
lead over the CPU unchanged when whole subject areas are held out. That is no sign that topic
shift explains the gap, though the check cannot rule it out. [QLoRA adaptation](docs/lora.md)
does not improve on the current system. The adapted model alone reaches 0.4334, no better than
its adapters-disabled control (0.4379). Its nested blend with the current system reaches 0.4545
(+0.04 points, 95% interval [−0.81, +0.89]), so the current system stays.
A [Claude–GLM specialist](docs/claude-glm.md) on frozen Qwen plus StyleDistance features
raises grouped balanced accuracy to 0.4701 (+1.60 points, 95% interval [+0.78, +2.41]). It
improves every fold and all seven of its classes, and its Qwen-only control gains nothing.
With whole subject areas held out, it adds +0.74 points. Its offline submission (57721458)
scored **0.429** public balanced accuracy, +0.8 points over the blend: about the gain that
the topic-held-out check predicted. Adding frozen LUAR-MUD or LUSR features to that specialist
does not help (−0.04 points, 95% interval [−0.70, +0.63]), so neither is promoted. Nor does
an 18-class Qwen + StyleDistance head blended after the specialist (−0.11 points, 95%
interval [−0.67, +0.41]): it gains a little on GPT-5.6 Terra and loses more elsewhere.
A binary Gemini 3.7 / 3.8 Flash specialist changes nothing (0.00 points, 95% interval
[−0.37, +0.41]). Neither CPU nor frozen-Qwen features separate the pair better than the
system's own forced choice (54–61% against 59.7%).
Adapting StyleDistance's top four layers on the Claude and GLM passages, mixed into the
Claude–GLM specialist, does not help either (−0.19 points, 95% interval [−0.41, +0.04]).
On held-out passages the adapted embeddings separate those seven classes no better than
the frozen ones.
A [seven-marker Gemini 3.7 / 3.8 study](docs/gemini-marker-study.md) (word lists, semicolons
and colons) fails its prompt gate (−0.19 points, 95% interval [−0.52, +0.19]); length plus markers
separate the pair at 59.3%, against the incumbent's forced 59.7%.
A [fixed-C seven-marker scoring model](docs/gemini-binary-scoring.md) gets 175 of the 300 pair
passages right on held-out folds, against the incumbent's forced 179.
A [five-class GPT specialist](docs/gpt-specialist.md) mixed into the incumbent's GPT mass gains
+0.26 points (95% interval [−0.30, +0.81]), short of the +0.5 promotion rule. It raises Luna, Sol and
Terra recall by 2–4 points each, at a cost to Qwen3.8 Max.
The written final deadline is January 13, 2027, while live metadata previously showed
January 14; plan for the earlier date until the host resolves the discrepancy.
