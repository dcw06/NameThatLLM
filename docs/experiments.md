# Experiment protocol

1. Validate the original CSVs without normalizing punctuation, case, or whitespace.
   Keep passage identifiers as strings. Never use an identifier as a predictive feature.
2. Freeze deterministic five-fold splits by `prompt_id`, with all labels represented in
   every validation fold. Preserve the content fingerprint and exact fold assignments.
   Cross-source near-duplicates and regenerated prompts require additional grouping.
3. Run the CPU baseline first. Save one out-of-fold prediction for each official row,
   the canonical label order, confusion matrix, per-model recall, and group-level scores.
4. Compare frozen Qwen features and a future DeBERTa experiment using the same folds.
   Choose by balanced accuracy, complementary errors, runtime, and peak GPU memory.
   Refit feature normalization and heads inside each training fold.
5. Audit token lengths with the pinned tokenizer before tuning truncation. A 512-token
   limit is an initial compute setting; 250 words can exceed it in code/math prose.
   Record truncation rates and performance by length. Do not silently strip punctuation.
6. Audit every external source for exact model identity, licenses, prompt provenance,
   API/proxy settings, exact and near duplicates, and official validation-prompt overlap.
   Keep all generated variants of a prompt on one side of a validation boundary.
7. Treat NF4 vs FP16, layer truncation, pooling, external adaptation, centering, and
   predicted-class rebalancing as separate ablations. Unequal hidden class counts are
   explicitly documented; balanced accuracy does not imply equal predicted label mass.
8. Tune small model blends on held-out predictions with a separate selection split or
   nested validation. Reusing the same 2,699 labels for many blend searches can overfit.
   Estimate uncertainty by resampling prompt groups rather than independent passages.
9. Before scoring, test a clean offline run and a roughly 4,000-row inference workload.
   Measure wall time and peak allocated GPU memory. Two T4 devices do not act as a
   single 32 GB device; batch size is measured per device.
10. Select final submissions using reliable validation evidence. Public score changes
    can reflect a different class mixture; keep a log of every submitted experiment.

Run records should include Git revision, data hash, fold hash, model repository/revision,
dependency versions, seeds, precision, token cap, hardware, time, and all class recalls.
Avoid comparing a public notebook's score to local CV as if they used identical data.

The frozen-neural baseline is a diagnostic. The planned competitive path is audited
external data followed by official-data adaptation with LoRA, then selective blending.
No paid generation or external data usage is automatically approved by this plan.

Sources: [official starter](https://www.kaggle.com/code/martynaplomecka/starter-notebook-for-name-that-llm-competition),
[prompt matching failure](https://www.kaggle.com/competitions/name-that-llm/discussion/751802),
[two-stage adaptation](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool),
[rules](https://www.kaggle.com/competitions/name-that-llm/rules).
