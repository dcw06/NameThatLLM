# Backbone decision for Name That LLM

Reviewed 2026-10-26. **Start the competitive GPU track with `Qwen/Qwen3-8B-Base`, using the documented truncated classifier recipe.** Keep `microsoft/deberta-v3-large` as the cheaper neural comparator and Qwen3-4B-Base as the memory fallback. This is an evidence-based starting decision; no backbone has been trained or compared on this project's folds yet.

## Evidence and tradeoffs

This competition rewards attribution across unseen prompts, rather than chat quality. General reasoning benchmarks do not establish which representation best separates sibling author models. The strongest directly inspected recipe here trains Qwen3 LoRA classifiers on public response corpora, then adapts them to the official data. Its author reports five-fold prompt-grouped balanced accuracy of 0.483 before adaptation and 0.569 afterward for 8B. A 14B run also reports 0.569; combining them with other components reports 0.593 CV and 0.592 public LB. These are participant-reported results, not independently reproduced measurements. [Two-stage recipe](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool)

| Candidate | Verified architecture; approximate full two-byte weight memory | Role and limitation |
| --- | --- | --- |
| Qwen3-8B-Base | 8.19B parameters, 36 decoder layers, width 4096; 15.3 GiB | Best initial competitive choice: concrete grouped-CV recipe, established Transformers support, manageable truncation. |
| Qwen3-14B-Base | 14.77B, 40 layers, width 5120; 27.5 GiB | Later diversity candidate. Reported standalone parity with 8B does not justify its larger initial cost. |
| DeBERTa-v3-large | Approximately 435M including embeddings, 24 encoder layers, width 1024; 0.8 GiB | Fast frozen-feature or fine-tuned comparator; substantially more iteration room. No isolated competition score verified here. |
| Qwen3.5-9B | 9.65B including vision components, 32 hybrid text layers, width 4096; 18.0 GiB | Later frozen-token feature experiment. This exact repository is a multimodal, post-trained checkpoint with extra architecture and dependency complexity. |
| Qwen3-4B-Base | 4.02B, 36 decoder layers, width 2560; 7.5 GiB | Useful fallback for one 16 GB GPU; no isolated competition advantage verified. |

Counts and architectures come from the pinned [8B](https://huggingface.co/Qwen/Qwen3-8B-Base/blob/49e3418fbbbca6ecbdf9608b4d22e5a407081db4/config.json), [14B](https://huggingface.co/Qwen/Qwen3-14B-Base/blob/0b0bd3732e2c374d483664439ea334928b65f304/config.json), [DeBERTa](https://huggingface.co/microsoft/deberta-v3-large/blob/64a8c8eab3e352a784c658aef62be1662607476f/README.md), [3.5](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json), and [4B](https://huggingface.co/Qwen/Qwen3-4B-Base/blob/906bfd4b4dc7f14ee4320094d8b41684abff8539/config.json) sources and Hugging Face parameter metadata. Memory estimates are parameter-count calculations, excluding activations, adapters, optimizer state, temporary buffers and framework overhead; they are not measured peaks.

## The checkpoint and the classifier are different artifacts

The competition recipe retains **24 of the 8B checkpoint's 36 layers**, removes the vocabulary prediction head, pools masked token states at eight depths, and learns an 18-class head. Its rank-16 LoRA and separate prompt-centered head are additional trained artifacts. The 14B variant retains 27 of 40 layers. A stock full-depth `AutoModelForSequenceClassification` is a different experiment. Layer indices, pooling, padding, normalization, tokenizer, adapter targets and checkpoint revision must match training when loading an adapter. Do not substitute the instruction-tuned `Qwen3-8B` repository for `Qwen3-8B-Base`.

For 8B, truncation and removal of the language-model head reduce estimated two-byte base weights to about 9.8 GiB. That improves the prospect of a 16 GB run but does not guarantee training fit. Construct/load the reduced model without first placing the complete causal LM on the GPU. Use masked pooling, disable generation caches, start with microbatch one and gradient checkpointing, and measure peak allocated/reserved memory. T4×2 provides two separate devices; ordinary replication does not combine their memory.

## Dependencies, precision and offline execution

Qwen3's model card requires Transformers 4.51.0 or newer. Use a tested, pinned Transformers/PyTorch/PEFT/Accelerate combination rather than automatically upgrading. DeBERTa uses the `deberta-v2` implementation and SentencePiece; its verified upstream revision contains `.bin` weights, requiring compatible PyTorch loading support. Qwen3.5's pinned card recommends newer/source Transformers support, making it a separate environment qualification task. [Qwen3 requirements](https://huggingface.co/Qwen/Qwen3-8B-Base/blob/49e3418fbbbca6ecbdf9608b4d22e5a407081db4/README.md), [Qwen3.5 card](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md)

Select precision from the actual GPU's support; do not assume BF16 acceleration on free T4/P100 hardware. NF4 QLoRA can reduce memory, but quantization changes representations and may change attribution accuracy. Revalidate it against the same folds; adapter merging and deployment must preserve the chosen quantization path. Bitsandbytes supports NF4 on Pascal-or-newer NVIDIA hardware, whereas LLM.int8 requires Turing-or-newer. Its CUDA workflow is separate from local Apple MPS execution. [Quantization documentation](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)

Package weights, tokenizer, adapters and wheels before the internet-disabled scoring run. The public two-stage author estimates roughly one hour for its complete ensemble on T4×2; this is not a benchmark for our implementation. Measure an approximately 4,000-passage workload before trusting the competition's nine-hour ceiling.

## Immutable source revisions and promotion criteria

Public Hugging Face API metadata verified these ungated revisions; all Qwen entries declare Apache-2.0 and DeBERTa declares MIT. Preserve license/attribution files with derived artifacts.

| Repository | Revision |
| --- | --- |
| `Qwen/Qwen3-8B-Base` | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| `Qwen/Qwen3-14B-Base` | `0b0bd3732e2c374d483664439ea334928b65f304` |
| `microsoft/deberta-v3-large` | `64a8c8eab3e352a784c658aef62be1662607476f` |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| `Qwen/Qwen3-4B-Base` | `906bfd4b4dc7f14ee4320094d8b41684abff8539` |

First compare frozen 8B and DeBERTa features on identical prompt-grouped folds. Then evaluate 8B LoRA, followed by vetted public-data pretraining and official-data adaptation. Deduplicate augmentation across validation prompts, inspect per-class recall and paired out-of-fold errors, and promote 14B or Qwen3.5 only for reproducible gains or useful ensemble diversity. Keep postprocessing as a separate ablation: balanced accuracy does not imply equal hidden-test class counts.
