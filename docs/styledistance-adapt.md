# Adapted StyleDistance for the Claude–GLM specialist

**Question.** Does adapting StyleDistance improve the complete seven-class Claude–GLM
specialist, under the existing integration rule?

## Reference

The reference is the submitted system: the CPU–Qwen blend, redistributed on the seven
Claude and GLM classes by the Qwen + StyleDistance specialist.

| | Score |
| --- | ---: |
| Prompt-grouped CV | 47.01% |
| Topic-grouped CV | 42.30% |
| Public leaderboard | 0.429 |

`ntl.style_adapt lock` rebuilds its out-of-fold probabilities from the recorded blend, the
specialist and each fold's `s`. It checks that they reproduce the recorded predictions,
then records hashes of the probabilities, predictions, fold assignments and recipes, for
both prompt and topic grouping. The rejected Gemini pair candidate scores the same overall
but is a different system, and is not the reference.

**Data.**
- Training covers the 1,049 Claude and GLM passages of each training partition, whatever
  the incumbent predicted.
- Evaluation covers all 2,699 passages and all 18 classes.

## Design

This section was written before any training or evaluation ran.

### Training recipe

| Setting | Fixed choice |
| --- | --- |
| Starting checkpoint | StyleDistance `b7df5f0b0480773c097ba3121d83ca32b71015ca` |
| Trainable backbone | Transformer layers 9–12; embeddings, layers 1–8 and the unused pooler frozen |
| Readout | Masked mean including special tokens, then L2 normalization |
| Temporary head | Linear 768 → 7, Xavier-uniform weights, zero bias |
| Loss | Unweighted mean cross-entropy on the normalized embeddings |
| Optimizer | AdamW, betas (0.9, 0.999), epsilon 1e-8 |
| Learning rates | Encoder 1e-5; temporary head 1e-3 |
| Weight decay | 0.01, excluding biases and LayerNorm parameters |
| Batching | Microbatch 4, accumulated to 16; each epoch's incomplete last batch kept |
| Gradient clipping | Global norm 1.0, after accumulation |
| Schedule | Linear warmup over 10 optimizer updates, then constant |
| Checkpoints | End of epochs 3 and 8, from one continuous run |
| Numerics | FP32, eager attention, seed 2026, deterministic CPU algorithms |
| Tokenization | Existing tokenizer, right padding, at most 512 tokens |

**Implementation choices the recipe leaves open, fixed here:**
- **Loss per update.** Each update's loss is the sum of its examples' cross-entropy
  divided by its actual number of examples.
- **Data order.** Each epoch draws a fresh permutation from a generator seeded with 2026,
  and the last update of an epoch takes whatever remains. Neither the order nor the
  schedule depends on how many epochs are run. A three-epoch run therefore reproduces
  the first three epochs of an eight-epoch run, which is checked.
- **Training and evaluation modes.** The embeddings and layers 1–8 stay in evaluation
  mode throughout. Layers 9–12 and the head train, so dropout is active only there.
  Embedding extraction is entirely in evaluation mode, in batches of 16.
- **Fresh start per partition.** Every encoder starts from the original checkpoint, with
  a fresh head, optimizer and random state, all seeded with 2026.

### Partitions

The original five outer prompt folds and their two inner prompt folds
(`artifacts/lora/plan.csv`) are kept.

| Encoder | Trained on | Embeds |
| --- | --- | --- |
| Inner, 10 of them | Claude and GLM passages of an inner training partition (about 420 from 60 prompts) | Those training passages, and every passage of the inner validation fold |
| Outer, only where selected | Claude and GLM passages of an outer training partition (about 840 from 120 prompts) | Those training passages, and every passage of the outer fold |

An adapted embedding belongs to exactly one encoder, partition and checkpoint. Each cache
records:
- the hashes of its training rows and their prompts;
- the row grouping and the configuration;
- the checkpoint's trained-weight hash and epoch;
- the actual row counts and number of optimizer updates.

Evaluation refuses a cache whose training or validation rows differ from the partition
it is used for. Every row the encoder did not embed is NaN, so a misused row cannot
silently enter a fit. The cached frozen Qwen features and the incumbent's inner
predictions are reused, and are checked against the same row grouping.

### Classifier and integration

The temporary head is discarded. The candidate specialist is the deployed one, with the
adapted StyleDistance block in place of the frozen one:
- Qwen (8,192 dimensions) plus adapted StyleDistance (768 dimensions);
- a StandardScaler fit on the training partition;
- StyleDistance scaled by √(8192/768);
- logistic regression with `C = 0.005`, the same solver and iteration limit, and the
  canonical class order;
- fit on the training partition's seven-class rows; probabilities for every validation
  passage.

With `p_B` the CPU–Qwen probabilities, `q_F` the frozen specialist, `q_A` the adapted
specialist and `M` the blend's mass on the seven classes:
- `q_mix = (1 − a) q_F + a q_A`;
- `p′_S = (1 − s) p_B,S + s M q_mix`;
- the other 11 classes are unchanged.

### Selection

There are exactly five options:
- the unchanged incumbent;
- epoch 3 at `a = ¼`;
- epoch 8 at `a = ¼`;
- epoch 3 at `a = ½`;
- epoch 8 at `a = ½`.

For each outer fold:
1. The inner folds choose `s` as the incumbent does. The incumbent is refit within each
   inner training partition.
2. The same inner folds then choose one option, by mean inner balanced accuracy over all
   18 classes. Ties are broken with the shared 1e-12 tolerance, in the order listed above:
   the incumbent, then the smaller `a`, then the shorter training.
3. The selected option is refit on the outer training partition. If the incumbent wins,
   no outer encoder is trained.

### Checks before results are accepted

- `a = 0` reproduces the locked incumbent's predictions exactly, and its probabilities
  within 1e-12, on all 2,699 passages.
- Zero and nonzero integration agree with an independently written formula.
- Only layers 9–12 and the head change. Every frozen tensor is bit-identical to the
  checkpoint after training.
- Gradients are finite at every update. Embeddings are finite unit vectors.
- The untrained encoder's readout reproduces the cached frozen StyleDistance features.
- Saved weights, reloaded, reproduce the cached embeddings and the specialist's
  predictions.
- No adapted cache is used outside its own training partition.

**Before any partition is trained:**
- **A training-only fit check.** A small balanced subset (8 passages per class) is
  trained with the same optimizer and loop for 30 epochs, to confirm the loop can fit
  training data.
- **One-epoch profiling** on a real inner partition.

Structural failures are debugged. Training curves diagnose underfitting or overfitting,
but the recipe is not changed after any outer result is examined.

### Before training: review and probe

**Review.** An independent review of the code, before any partition was trained, found
no defect in the training loop, the nesting or the caches. It found four gaps, all closed
before training:
- **The topic gate.** `evaluate` now applies it whenever rows are grouped by topic.
- **The reload check.** Each partition re-embeds four training and four validation
  passages from its saved weights. `evaluate` reloads every outer encoder it uses,
  re-embeds all of that encoder's passages, and requires identical predictions and
  probabilities within 1e-6.
- **Report tests.** These now compare two systems that actually differ.
- **Checkpoint tests.** A checkpoint is now shown to leave the rest of the run unchanged.

All but one of 20 code mutations then fail the tests. The survivor marks the unused
pooler trainable, which changes nothing because the pooler receives no gradient.

**Probe** (`artifacts/styledistance-adapt/probe.json`; CPU, 8 threads, deterministic):
- **Readout.** The untrained encoder reproduces the cached frozen features within
  4.2 × 10⁻⁷.
- **Training-only fit check.** On 56 passages over 30 epochs (120 updates), training loss
  falls steadily from 1.95 to 1.19 and training accuracy rises from 9% to 79%. The loop
  fits, but slowly: the linear head sees unit-length embeddings, so its logits grow only
  as its weights do. This is a property of the fixed recipe and is recorded, not changed.
- **One-epoch profile.** An inner partition (419 passages, 27 updates) takes 42.5 s per
  epoch. Embedding takes 0.05 s per passage.

### Evaluation and promotion

**Reported:**
- overall balanced accuracy, and precision and recall for every class;
- passages gained and lost;
- the targeted Claude–GLM confusions;
- training curves: per-epoch training loss and accuracy, plus the temporary head's
  held-out loss and accuracy at each checkpoint;
- the selected settings.

Fable's precision and recall are reported as diagnostics. The promotion rule does not
require them to improve.

**Promotion rule.** The candidate is promoted only if:
- its unrounded gain is at least 0.005;
- its 95% paired prompt-bootstrap interval lies entirely above zero (10,000 whole-prompt
  resamples, seed 2026).

**Topic check.** Only a passing candidate repeats the complete nested procedure with topic
groups throughout, against the exact topic incumbent behind 42.30%. It must gain overall
and change by zero or more in at least three of the five folds. The bootstrap resamples
whole topics.

**Final fit.** Only after both gates pass:
- The existing full-data inner procedure chooses the checkpoint and `a`, reproducing the
  incumbent recipe. If it selects `a = 0`, the incumbent is kept.
- Otherwise the specialist is fit on all 1,049 passages. The saved artifacts are:
  - both specialist heads and their scalers;
  - the adapted encoder weights and the checkpoint identity;
  - the class order, `a` and `s`, and provenance.
- Reloaded predictions are verified before the offline 4,000-row benchmark.

**Scope of the conclusion.** The result is evidence about this fixed recipe and
restricted blend. Its interval is conditional on the fixed seed and on reused
validation folds.

## Result: original prompt folds

```sh
# Ten inner encoders (F = 0..4, J = 0, 1), each one eight-epoch run checkpointed at 3 and 8
PYTHONPATH=src .venv/bin/python -m ntl.style_adapt train --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv --model-path models/styledistance --threads 8 \
  --root artifacts/styledistance-adapt/encoders --outer-fold F --inner-fold J \
  --epochs 8 --save-epochs 3 8
PYTHONPATH=src .venv/bin/python -m ntl.style_adapt select --train data/raw/train.csv \
  --plan artifacts/lora/plan.csv \
  --qwen-features artifacts/qwen-validation-results/features/qwen3-8b-base-train.npz \
  --styledistance-features artifacts/claude-glm/styledistance-train.npz \
  --root artifacts/styledistance-adapt/encoders \
  --current-inner artifacts/claude-glm/current-inner/fold{0,1,2,3,4}.parquet \
  --output artifacts/styledistance-adapt/selection-prompts.json
# Outer encoders only where selected: folds 0 and 4 at epoch 3, fold 1 at epoch 8
PYTHONPATH=src .venv/bin/python -m ntl.style_adapt train ... --outer-fold F --epochs E --save-epochs E
PYTHONPATH=src .venv/bin/python -m ntl.style_adapt evaluate ... \
  --current-oof artifacts/qwen-blend/oof.parquet \
  --reference artifacts/styledistance-adapt/reference-prompts.json \
  --model-path models/styledistance --threads 8 --output artifacts/styledistance-adapt/prompts
```

### Checks

Every check passed:
- **The incumbent.** At `a = 0`, the evaluation reproduces the locked incumbent on all
  2,699 passages: identical predictions and a maximum probability difference of 0.
- **The specialist's `s`.** The inner folds choose ¼, ¼, ½, ¼, ¼, the incumbent's values.
- **Integration.** Both the zero and the selected mixtures match the independent formula
  exactly.
- **Parameters.** In all 13 encoders, every frozen tensor is bit-identical to the
  checkpoint and the trainable tensors changed. Each made 27 updates per epoch on an
  inner partition, or 53 on an outer partition.
- **Reloading.** Each partition's saved weights re-embed their sample passages exactly.
  The evaluation reloaded all three outer encoders and re-embedded all 1,379 of each one's
  passages. Embeddings and candidate probabilities were identical, and so were the
  predictions.
- **Prefix.** A separate three-epoch run of outer0-inner0 reproduces the eight-epoch run
  bit for bit: weight hashes and losses for epochs 1–3, the saved weight file and the
  embeddings. Its first-epoch weights also match the probe's.

### Competition result

| System | Grouped balanced accuracy | Change | 95% interval |
| --- | ---: | ---: | --- |
| Incumbent: blend + Claude–GLM specialist | 0.4701 | — | — |
| With adapted StyleDistance mixed in | 0.4682 | −0.19 | [−0.41, +0.04] |

97% of resamples put the change at or below zero. The candidate gains 4 passages and
loses 9.

**Selection.** Three folds chose a small adapted share. Its inner margins did not carry
over to the held-out folds:

| Fold | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Selected | Epoch 3, `a` ¼ | Epoch 8, `a` ¼ | Incumbent | Incumbent | Epoch 3, `a` ¼ |
| Best adapted option's inner margin (points) | +0.19 | +0.14 | −0.19 | 0.00 | +0.19 |
| Held-out change (points) | 0.00 | −0.37 | 0.00 | 0.00 | −0.56 |

**Ties.**
- In fold 3, epoch 3 at `a = ¼` tied the incumbent exactly, so the incumbent was kept.
- In fold 1, epoch 8 at `a = ¼` and `a = ½` tied within the tolerance, so the smaller `a`
  was taken.
- In fold 2, every adapted option scored below the incumbent.
- No fold selected `a = ½`.

**The 20 passages that changed** all lie in folds 0, 1 and 4:

| Class | Recall | Precision | Gained / lost |
| --- | --- | --- | --- |
| Claude Fable 5 | 20.1% → 19.5% | 28.6% → 27.6% | +0 / −1 |
| Claude Opus 4.6 | 65.3% → 64.0% | 54.1% → 53.3% | +0 / −2 |
| Claude Opus 4.8 | 39.3% → 38.7% | 44.7% → 43.6% | +0 / −1 |
| Claude Opus 5 | 76.7% → 76.0% | 59.9% → 60.3% | +1 / −2 |
| Claude Sonnet 5 | 51.3% → 50.0% | 52.7% → 51.4% | +0 / −2 |
| GLM 5.3 | 26.7% → 27.3% | 29.2% → 29.7% | +1 / −0 |
| GLM 5.3 Flash | 32.0% → 33.3% | 28.9% → 29.8% | +2 / −0 |
| Gemini 3.6 Flash | 49.3% → 48.7% | 58.7% → 58.4% | +0 / −1 |

Recall is unchanged for the other ten classes. Precision moves only for GPT-5.6 Terra
(36.9% → 37.1%) and Qwen3.8 Max (53.8% → 53.1%):
- a Kimi K3 passage moved from Terra to Sonnet 5;
- a wrongly predicted Fable passage and a wrongly predicted Sonnet 5 passage both moved
  to Qwen3.8 Max.

**Targeted confusions:**

| True → predicted | Incumbent | Candidate |
| --- | ---: | ---: |
| Fable → GLM 5.3 | 19 | 20 |
| Fable → GLM 5.3 Flash | 28 | 28 |
| GLM 5.3 → Fable | 21 | 21 |
| GLM 5.3 Flash → Fable | 16 | 16 |
| GLM 5.3 → GLM 5.3 Flash | 28 | 28 |
| GLM 5.3 Flash → GLM 5.3 | 30 | 28 |

**Fable diagnostics.** Fable's precision falls from 28.6% to 27.6% and its recall from
20.1% to 19.5% (30 to 29 of 149).

### Training curves

These are ranges over the ten inner encoders. Loss starts at ln 7 ≈ 1.946.

| | Epoch 1 | Epoch 3 | Epoch 8 |
| --- | --- | --- | --- |
| Training loss (running) | 1.943–1.950 | 1.868–1.919 | 1.655–1.748 |
| Training accuracy (running) | 10.7–17.1% | 21.0–26.3% | 32.4–41.9% |
| Temporary head, training accuracy | — | 23.3–28.6% | 36.9–47.4% |
| Temporary head, held-out accuracy | — | 18.6–26.5% | 26.0–33.4% |
| Temporary head, held-out loss | — | 1.855–1.903 | 1.737–1.827 |

The outer encoders' temporary heads reach 28.6% held-out accuracy (folds 0 and 1) and
25.2% (fold 4), where chance is 14.3%.

**What the curves show.**
- The recipe underfits. After 216 updates the loss has fallen only from 1.95 to about
  1.7.
- The temporary head's gap between training and held-out accuracy widens with training,
  from 3.6 points on average at epoch 3 to 13.3 at epoch 8.
- Mean gradient norms before clipping grow from 0.7–0.8 in epoch 1 to 5–10 in epoch 8,
  well above the clipping threshold of 1.0.

**Time.** All of this ran on the CPU with 8 threads.
- Each inner encoder took 9.3–11.3 minutes, including embedding and the reload check.
- The outer encoders took 6.6 minutes at three epochs and 15.6 at eight. Fold 0 took
  9.6 minutes because it shared the CPU with the prefix run.
- The evaluation, with its full reload check, took 7.8 minutes.

### Diagnostic: is the adapted embedding a better style signal?

This diagnostic is not part of the promotion rule. It measures seven-class balanced
accuracy on the held-out true Claude and GLM passages of each partition with an adapted
encoder (about 420 per inner partition, 210 per outer fold). Each classifier uses the
deployed recipe.

| Classifier | Frozen | Adapted, epoch 3 | Adapted, epoch 8 |
| --- | ---: | ---: | ---: |
| StyleDistance alone (inner, mean of 10) | 32.2% | 31.4% | 32.0% |
| Qwen + StyleDistance specialist (inner) | 41.0% | 40.0% | 37.5% |
| Specialist mixed at `a = ¼` (inner) | 41.0% | 41.0% | 41.1% |

**Adapted against frozen, per inner partition:**
- StyleDistance alone is higher in 4 of 10 partitions at epoch 3 and 5 of 10 at epoch 8.
- The full specialist is higher in 1 of 10 at epoch 3 and none at epoch 8.

**The three selected outer encoders:**
- StyleDistance alone: 33.2% → 33.7%;
- the specialist: 46.8% → 43.0%;
- the selected ¼ mixture: 46.7%.

**Adaptation did not produce a better style signal.**
- On passages it did not train on, the adapted StyleDistance alone separates the seven
  classes no better than the frozen one, at either checkpoint.
- Replacing the frozen block in the specialist costs 1.1 points at epoch 3 and 3.5 at
  epoch 8.
- The ¼ mixture only recovers the frozen level.

**A likely contributor, not tested.** The deployed classifier is fit on the same passages
the encoder was trained on. Adapted embeddings separate those passages better than new
ones: at epoch 8 the temporary head reaches 37–47% on its training passages but only
26–33% on held-out ones. The classifier may therefore weight the adapted block more than
its held-out value supports. That would fit three observations:
- the specialist's accuracy cost grows from epoch 3 to epoch 8;
- StyleDistance alone, held out, stays level;
- the deployed StyleDistance-only classifier fits its own training passages better on
  adapted embeddings than on frozen ones. The review checked three partitions: 72.0%,
  65.0% and 63.4% against 58.5%, 57.1% and 54.5%.

Fitting the classifier on cross-fitted embeddings would test this, but that is a
different design and a new experiment.

### Review

An independent review recomputed the run from the saved artifacts, using its own code.
It reproduced exactly:
- every metric and the bootstrap interval;
- all five folds' inner `s` and option scores;
- the outer candidate's probabilities;
- every partition's training and validation rows, with no outer-fold passage in any
  encoder's training set;
- every cache, weight and configuration hash.

It found no defect in the module.

**Two further findings:**
- **Solver tolerance.** The selection does not depend on how far the logistic
  regression converges. Refit at tolerance 1e-10, every fold makes the same choice,
  including the fold-3 tie, and the outer predictions are identical. At the default
  tolerance, however, a small numerical change, such as standardizing in float64,
  can break the fold-3 tie toward epoch 3 at `a = ¼`.
- **The bootstrap's fraction at or below zero.** The shared bootstrap's reported
  fraction was slightly low. Resamples whose change is exactly zero could come out a
  little above zero in floating point, about 10⁻¹⁹. The exact fraction is 97.35%,
  against the 96.9% recorded in `comparison.json`. The interval and the decision
  are unaffected. The bootstrap now counts such ties as zero. Rechecking every
  recorded comparison changed no interval or decision, only two quoted fractions: the
  [Gemini pair](gemini-pair.md) and [18-class head](styledistance-head.md) records.

**A later code review** found two defects that did not affect this run. Both are now fixed
and tested.
- **The topic gate.** It compared raw float differences. An exact overall tie could read
  as a gain of about 10⁻¹⁷ and pass. A real gain could fail when tied folds read as tiny
  losses. Overall and per-fold changes within the shared 1e-12 tolerance now count as
  exact ties. The same rule now applies to the blend's count of folds won, whose recorded
  margins are all at least 2.9 points.
- **Checkpoint diagnostics on an accelerator.** The temporary head's checkpoint scores
  built a CPU tensor for a head on `--device`. They now follow the device, which was
  checked by training a partition on MPS. CUDA was not available to test. On CUDA,
  PyTorch's deterministic mode also requires `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

### Decision

**Not promoted.** The change is −0.19 points, against a required gain of at least +0.5
with an interval above zero. As declared:
- the topic-grouped check was not run;
- no final fit was made;
- nothing was packaged or submitted.

The blend plus Claude–GLM specialist (47.01% CV, public 0.429) stays the submitted system.

**What this answers.** Under this fixed recipe, adapting StyleDistance's top four layers
on the Claude and GLM passages does not improve the complete seven-class specialist. The
adapted embeddings carry no more held-out seven-class signal than the frozen ones. At
the fixed learning rate and schedule, eight epochs leave the encoder underfit on
training data. The result says nothing about other recipes, such as a larger head
learning rate, a temperature on the normalized embeddings, or more epochs. Choosing
among those would need new nested folds.
