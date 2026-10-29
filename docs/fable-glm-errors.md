# Fable / GLM error inspection

**Question.** Are the incumbent's Fable and GLM errors narrow decisions that an adjustment
to the existing scores could recover, or do they need additional information?

**What was inspected.** The incumbent's own locked out-of-fold probabilities (blend plus
Claude–GLM specialist, 47.01% prompt-grouped CV). Nothing is trained.

```sh
PYTHONPATH=src .venv/bin/python -m ntl.error_inspection --train data/raw/train.csv \
  --reference artifacts/styledistance-adapt/reference-prompts.json \
  --output-dir artifacts/fable-glm-errors
```

**What is recorded for every passage:**
- the true label's rank (1 when correct);
- the gap between the predicted and true labels' probabilities;
- the top probability;
- the length, counted with the marker study's tokenizer.

**Outputs.** `passages.csv` and `summary.json` hold the measurements. `report.md` gives the
narrowest, median and most confident error of each direction, with excerpts. All three stay
in `artifacts/fable-glm-errors/`, because the excerpts are competition text.

**Context.** Across all 1,430 errors in the dataset:
- 36% have the true label second;
- the median gap is 0.14;
- errors have a median top probability of 0.27, against 0.33 for correct predictions.

## Results

| Confusion | Errors | True label 2nd / 3rd / 4th or lower | Gap, median (IQR) | Gap < 0.05 | Top probability, median | Words, median (errors / correct) |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Fable → GLM Flash | 28 | 9 / 7 / 12 | 0.12 (0.08–0.26) | 3 | 0.27 | 113 / 109 |
| GLM Flash → Fable | 16 | 3 / 5 / 8 | 0.16 (0.12–0.26) | 0 | 0.28 | **69** / 128 |
| GLM → GLM Flash | 28 | 9 / 8 / 11 | 0.18 (0.13–0.27) | 3 | 0.31 | 106 / 109 |
| GLM Flash → GLM | 30 | 12 / 6 / 12 | 0.16 (0.05–0.28) | 8 | 0.30 | 115 / 128 |
| GLM → Claude Opus 5 | 25 | **15** / 4 / 6 | 0.12 (0.08–0.18) | 5 | 0.27 | 109 / 109 |

### Fable ↔ GLM Flash: is the correct label usually second?

**No.** It is second in only 12 of the 44 errors, and fourth or lower in 20. The median gap
is 0.12 to 0.16 against a predicted probability near 0.27, and only 3 errors are within 0.05.
These are not near-ties between the two labels.

Two further observations:
- **Only one prompt has a mutual swap,** with its Fable passage called GLM Flash and its GLM
  Flash passage called Fable.
- **Fable's errors are spread widely.** Of 149 Fable passages, 28 go to GLM Flash, 19 to
  GLM 5.3, 17 to Opus 5, 15 each to Opus 4.8 and Sonnet 5, and 13 to Opus 4.6. Only 30 are
  correct.

**Short GLM Flash passages lean to Fable.** GLM Flash → Fable errors are short: median 69
words, against 128 for correctly classified GLM Flash (Mann–Whitney p = 0.04, descriptive).
Under 80 words, 10 of 52 GLM Flash passages are called Fable; at 80 words or more, 6 of 98.

Examples are in the report: narrowest `62bbcbb0ff71` and `506cc037e03d`; most confident
`aa9dd257b403` (true rank 5, gap 0.42) and `dcd0ee22087a` (48 words, true rank 6).

### GLM ↔ GLM Flash: how large is the gap between the versions?

**Usually large.** When one GLM version is called the other, the predicted version gets
about 0.30 and the true one about 0.11–0.12, a median gap of 0.16–0.18. Only 11 of 58 errors
are within 0.05, eight of them GLM Flash passages called GLM 5.3. The true version is second
in 21 of the 58 errors.

**Examples:**
- **Near-exact tie:** `2b0cbe7653d2`, at 0.299 each.
- **Most confident error:** `fde38289c3ce`, with GLM Flash at 0.63 and GLM 5.3 fourth
  at 0.05.

### GLM → Claude Opus 5: confident mistakes or narrow decisions?

**Low-confidence mistakes, though not knife-edge ones.**
- **Low confidence.** In 18 of the 25 errors, Opus 5 gets under 0.30, and only one exceeds
  0.4. Correct Opus 5 predictions have a median top probability of 0.44.
- **The true label is close.** GLM 5.3 is second in 15 of the 25 errors, the highest share
  of any group.
- **But not a near-tie.** The median gap is still 0.12, and 5 errors are within 0.05.

**Examples:**
- **Narrowest:** `17bef92191b6`, gap 0.007.
- **Most confident:** `25f758b824df` (62 words), with Opus 5 at 0.61 against GLM 5.3 at 0.10.

### Would an adjustment to the existing scores help?

**Correct predictions here are just as fragile.** The incumbent's correct calls in these
classes have low top probabilities (medians 0.25–0.28). Within 0.05 of the runner-up are:
- 11 of 30 correct Fable predictions;
- 13 of 40 correct GLM 5.3 predictions;
- 15 of 48 correct GLM Flash predictions.

Moving a decision boundary therefore breaks about as many correct predictions as it fixes.

**Scaling one class's probability by a constant.** This uses every label, so it is an
in-sample upper bound:

| Class scaled | ×1.1 | ×1.25 | ×1.5 | ×2 |
| --- | ---: | ---: | ---: | ---: |
| Fable | +0.04 (3 / 2) | −0.15 (8 / 12) | −0.14 (21 / 25) | −0.62 (36 / 53) |
| GLM 5.3 | −0.15 (5 / 9) | −0.45 (11 / 23) | −0.59 (16 / 32) | −1.34 (33 / 69) |
| GLM Flash | +0.04 (4 / 3) | −0.19 (11 / 16) | −0.74 (14 / 34) | −1.45 (27 / 66) |

Each cell is the balanced-accuracy change in points, then passages gained / lost.

Even chosen with hindsight, the best rescaling gains +0.04 points, about one passage net.
Larger factors take passages from every neighbouring Claude and GLM class.

## Conclusion

Most of these errors are not narrow decisions that the existing scores almost get right.
Across the three groups:
- the true label is second in only 48 of the 127 errors;
- the median gaps (0.12–0.18) exceed the margins of the incumbent's correct predictions in
  the same classes;
- rescaling the scores recovers at most about one passage, even in sample.

A further adjustment to the existing probabilities, such as another class weight or mixing
step, is therefore unlikely to help. These errors need additional information that the
current features do not carry.

**Two leads.** Both are descriptive, found by looking at all the labels, so neither is a
validated result:
- **Length:** short GLM Flash passages lean towards Fable.
- **The closest call:** GLM → Opus 5 is the confusion where the correct label is most often
  second.
