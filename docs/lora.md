# QLoRA adaptation of the frozen Qwen3-8B-Base recipe

**Question.** Does adapting Qwen3-8B-Base improve on the validated CPU + frozen-Qwen
system (45.41% grouped balanced accuracy, public 0.421)? Everything else stays fixed:
- tokenization and the 512-token limit;
- the 24 retained blocks;
- masked-mean pooling of blocks 7 and 22.

## Design

**Head.** The head is the frozen pipeline itself:
- a standardization fitted on the training partition;
- a linear layer initialised from the `C = 0.005` logistic fit;
- training against the same objective, mean cross-entropy plus `||W||² / (2 C n)`.

**Adapters.** Rank-16 LoRA adapters (`alpha = 32`, dropout 0.05) on all seven
projection modules of blocks 1–22 train jointly with the head. Blocks 23–24 come after
the last pooled block and cannot affect the features. The NF4 base weights stay frozen
(QLoRA, float16 compute with gradient scaling, gradient checkpointing). Each step uses
micro-batches of 4 and accumulates 4, an effective batch of 16.

**Control.** The identical loop with adapters disabled: only the head trains, on the
cached frozen features.

**Selection.**
- **Inner folds.** Inside every outer training partition, two inner prompt folds choose
  the training setting. Duration 0 is the frozen head itself.
- **Outer refit.** Outer folds are refit once with the chosen setting and scored once.
- **Blend weight.** On the same inner folds, the current system is refit with its own
  nested weight selection, and the blend `(1 - v) * current + v * adapted` takes `v` from
  `{0, 0.25, 0.5, 0.75, 1}`. `v = 0` reproduces the current system exactly.
- **Reporting.** All 2,699 rows are scored, with paired prompt-bootstrap intervals
  against the current system.

## Qwen-only leaderboard diagnostic

Changing only the blend weight of the submitted notebook to 1 gives public **0.399**
(submission 57688014). The blend scores 0.421 and the CPU baseline 0.360. The CPU adds
2.2 public points, at least as much as in cross-validation. The cross-validation
shortfall on the leaderboard comes from the Qwen component: Qwen alone loses 3.9 points,
the CPU none. See [the blend record](qwen-blend.md#submission-result).

## Training check (Kaggle T4, 72 training-partition passages)

| Measurement | Value |
| --- | --- |
| Trainable adapter parameters | 26.7 M |
| Peak allocated GPU memory | 4.66 GiB |
| Training time | 0.52 s per passage per epoch (0.44 s with length bucketing at fold scale) |
| Scoring time | 0.15 s per passage |
| Initial features vs. frozen archive | Mean relative difference 0.07% (float16 batching) |
| Subset cross-entropy over 6 epochs (lr 1e-4) | 0.144 → 0.006; no float16 overflow |

## Run 1: learning rate 1e-4, whole epochs 0–3

| Outer fold | Inner balanced accuracy at 0 / 1 / 2 / 3 epochs | Chosen | Outer: adapted | Outer: frozen head |
| --- | --- | ---: | ---: | ---: |
| 0 | **.389** / .381 / .375 / .366 | 0 | .444 | .444 |
| 1 | .385 / **.395** / .373 / .354 | 1 | .430 | .435 |
| 2 | **.387** / .362 / .375 / .368 | 0 | .452 | .452 |
| 3 | **.379** / .372 / .363 / .376 | 0 | .437 | .437 |
| 4 | **.390** / .385 / .356 / .369 | 0 | .424 | .424 |

The training objective falls steadily (about 0.58 → 0.36 over three epochs on 1,080
passages), but accuracy on held-out prompts falls after the first epoch. At this
learning rate, the 26.7 M adapter parameters overfit a training split of about 1,080
passages. The inner folds therefore reject training in four of five folds. The adapted
model scores **0.4375** out-of-fold, against 0.4386 for the frozen head and 0.4541 for the
current system. The matching adapters-disabled control (same grid) scores 0.4390.
Run 1 is not a candidate.

## Run 2: learning rate {1e-5, 3e-5} × duration {0, ¼, ½, 1, 2} epochs

This pre-specified, gentler grid is chosen inside the same inner folds. The learning rate
is constant after a 10-step warmup, so the checkpoint at `d` epochs equals training for
exactly `d` epochs. Ties choose the shorter duration, then the smaller learning rate.
The matching control (adapters disabled, same grid) scores **0.4323**. On near-flat inner
curves, choosing among nine settings costs about half a point against the frozen head
(0.4386).

| Outer fold | Chosen setting | Inner: chosen / frozen | Outer: adapted | Outer: frozen head | Outer: control | `v` | Outer: blend | Outer: current |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | lr 1e-5, ¼ epoch | .391 / .389 | .448 | .444 | .443 | 0.25 | .470 | .476 |
| 1 | lr 3e-5, ½ epoch | .386 / .385 | .419 | .435 | .433 | 0 | .426 | .426 |
| 2 | lr 3e-5, ¼ epoch | .392 / .387 | .457 | .452 | .435 | 0.25 | .478 | .487 |
| 3 | lr 3e-5, 1 epoch | .390 / .379 | .415 | .437 | .430 | 0.25 | .447 | .430 |
| 4 | lr 3e-5, 1 epoch | .394 / .389 | .428 | .424 | .420 | 0 | .452 | .452 |

Every fold chose some training, but no inner curve is more than about one point above
the frozen head. On held-out prompts, the adapted model gains 0.4–0.5 points in three folds
and loses 1.6 and 2.2 in the other two. Each fold took 1.5–2 hours on one T4 (about
8.6 GPU-hours in total).

**All 2,699 rows, paired prompt bootstrap (150 prompts, 10,000 resamples):**

| System | Grouped balanced accuracy | Difference from current | 95% interval |
| --- | ---: | ---: | --- |
| Current CPU + frozen Qwen | **0.4541** | — | — |
| Adapted Qwen alone | 0.4334 | −2.07 | [−3.51, −0.63] |
| Adapted Qwen + current, inner `v` | 0.4545 | +0.04 | [−0.81, +0.89] |
| Control alone | 0.4323 | −2.18 | [−3.52, −0.89] |
| Control + current, inner `v` | 0.4530 | −0.11 | [−0.70, +0.48] |

The adapted model is +0.11 points over its control (95% interval [−0.86, +1.11]): training
the adapters adds nothing measurable over training the head alone. The inner folds give the
adapted model weight 0.25 in three folds and zero in two. The resulting blend agrees with
the current system on 91% of rows and moves no class recall by more than 5 points
(each class has about 150 passages). The net effect is zero.

## Decision

Neither run produces a candidate:
- Adapted Qwen alone is 2 points below the current system.
- The blend with the current system is +0.04 points, inside the noise.

The current system (45.41%, public 0.421) remains the validated winner and is already
submitted, so there is no new submission. Two later steps apply only to a system that
beats 45.41%, so neither was run:
- the eight-topic robustness check;
- the final fit, notebook benchmark and submission.

The code for both is ready: `ntl.lora_final`, and `--mode final|submission` in
`scripts/prepare_lora_runs.py`.

**What this does not rule out.** Both runs share the following:
- one adapter shape: rank 16, all projections, blocks 1–22;
- a head initialized from the strongly regularized `C = 0.005` probe and kept under its
  penalty;
- about 1,080 passages per inner training split.

The flat inner curves suggest the binding constraint is the amount of labelled text per
prompt split, not the adapter shape. Even so, a larger-data or different-objective variant
remains untested.
