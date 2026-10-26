# Run on Kaggle's free compute

The project starts with a CPU reference and a frozen Qwen3-8B-Base diagnostic.
Use free Kaggle T4 GPUs for the neural experiment. No cloud account, paid API,
or external training corpus is required for these notebooks. The frozen model's
competition score is unmeasured; it establishes a reproducible starting point.

The generated notebooks embed `src/ntl` directly. You do not need to upload the
project as another input. Regenerate them after changing the source:

```sh
python scripts/build_kaggle_notebooks.py
```

Check that all four exports match current source and that their code cells
compile with `python scripts/build_kaggle_notebooks.py --check`.

## Prepare the offline dependency inputs

Import
[`notebooks/kaggle_prepare_dependencies.ipynb`](../notebooks/kaggle_prepare_dependencies.ipynb)
into a Kaggle **CPU notebook with Internet on**. Save & Run All downloads two
hash-locked Linux wheel bundles: `cpu-wheelhouse` for the baseline, and
`neural-wheelhouse` for the frozen-model experiment. Set `PREPARE_NEURAL = False`
if you only need the smaller CPU bundle. Downloading GPU libraries does not
allocate a GPU. The complete neural bundle can require several GB.

Create a **private Kaggle dataset** from the completed notebook's wheelhouse
folders. Keep each folder's `wheel_manifest.json`, `requirements.lock`, and all
its wheel files together. Attach this dataset to the offline submission
notebooks. Each automatically selects the matching CPU or neural profile;
if several matching bundles are attached, set `WHEEL_DIR` explicitly.

The notebooks require the matching bundle, verify its lock and platform, and
install it into an isolated environment under `/kaggle/working/ntl_env` with
`pip --no-index --require-hashes`. Training, inference, and output validation
run as subprocesses in that environment. They do not replace packages inside
Kaggle's already-running notebook kernel. CPU pins come from `requirements-cpu.lock`;
the complete neural pins come from `requirements-kaggle.lock`. The environment
checks use the exact project versions, including NumPy 2.3.5, scikit-learn 1.7.2,
PyArrow 21.0.0, and the neural stack's PyTorch 2.8.0 and Transformers 4.57.1.
Kaggle's preinstalled package versions can differ. Recreate the bundles if the
Kaggle Python version or Linux image changes; prepare them on Kaggle instead of
uploading macOS wheels.

## First run: CPU baseline

1. Read the [competition rules](https://www.kaggle.com/competitions/name-that-llm/rules)
   in your own account and accept them if you choose to participate.
2. Create a Kaggle notebook and import
   [`notebooks/kaggle_cpu_baseline.ipynb`](../notebooks/kaggle_cpu_baseline.ipynb).
3. Attach the official **Name That LLM** competition data and the prepared
   dependency dataset. Use **Accelerator: None** and **Internet: off**.
4. Leave `RUN_GROUPED_CV = False` for an initial smoke test, or set it to `True`
   to save prompt-grouped validation results. Save & Run All.
5. Review the completed version and submit its `submission.parquet` output
   through Kaggle's competition flow.

The training/inference notebooks discover only the known official competition
mounts, avoiding unrelated attached files named `train.csv`. If both mounts are
present, set `COMPETITION_DIR` explicitly in the data cell. The output validator
checks all IDs, row order, nulls, duplicate IDs, and allowed model labels.

## Prepare the neural model input once

The selected backbone is **`Qwen/Qwen3-8B-Base`**, pinned to commit
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. Its original configuration has 36
decoder layers and hidden width 4096. The initial feature experiment uses the
first 24 layers; the original snapshot remains intact. See
[model selection](model-selection.md) for the choice and its limits.

1. Import
   [`notebooks/kaggle_prepare_qwen_model.ipynb`](../notebooks/kaggle_prepare_qwen_model.ipynb)
   into a separate Kaggle notebook. Select **Accelerator: None** and
   **Internet: on**. Preparation uses no GPU.
2. Save & Run All. This downloads the public snapshot at the pinned commit,
   including its tokenizer, model weights, configuration, license, and
   `snapshot_manifest.json`. The full weights require roughly 16 GB of storage;
   allow additional scratch space.
3. From the completed notebook's Output tab, create a **private Kaggle dataset**
   containing the `qwen3-8b-base` folder. Keep `snapshot_manifest.json` beside
   `config.json` and preserve every weight shard.

Model preparation intentionally uses the network. The training and scoring
notebooks consume the resulting local snapshot with network access disabled.
The helper never accepts competition rules or creates a dataset for you.

The upstream [model card](https://huggingface.co/Qwen/Qwen3-8B-Base) declares
Apache-2.0. Preserve its license and attribution. Model weights, project code,
competition data, and any later external datasets retain their separate terms.
No external generated corpus is used in the initial workflow.

## Run frozen features on a free T4

1. Import
   [`notebooks/kaggle_qwen_frozen.ipynb`](../notebooks/kaggle_qwen_frozen.ipynb).
2. Attach the official competition data, your prepared model dataset, and the
   dependency dataset containing the neural wheelhouse.
   Select **GPU T4 x2** and **Internet: off**. The initial feature extractor can
   use one T4; selecting two does not imply that inference automatically uses
   both cards.
3. The model cell discovers the pinned snapshot by its manifest. If several
   snapshots are attached, set `MODEL_DIR` to the directory containing
   `snapshot_manifest.json` and `config.json`.
4. Keep the initial batch size of 2, maximum sequence length of 512, and
   `nf4` quantization. This requires `bitsandbytes` and `accelerate`. The notebook
   checks CUDA and runs a tiny NF4 forward probe before loading the real model.
   It stops on unsupported dependencies or kernels; it does not silently
   choose another model. This probe executes on your Kaggle GPU and has not
   been measured on a T4 from this local setup.
5. For the first execution leave `RUN_GROUPED_CV = False`. For an experiment
   intended to compare accuracy, set it to `True` and inspect the saved
   prompt-grouped results before drawing conclusions.
6. Save & Run All, inspect the final schema-validation cell, and submit that
   saved version's `submission.parquet`.

The initial classifier uses frozen neural features and a small supervised
readout. It does not train a LoRA adapter. The feature and pooling settings are
experimental choices, not a claimed best architecture or measured competition
score. A final model fit uses all official training rows; its fitting accuracy
would not establish generalization.

If dependencies are absent or versions differ, recreate the matching prepared
wheel bundle. The notebook stops with an actionable error when it cannot
establish the expected isolated environment. Keep package
versions and the snapshot manifest with experiment reports. A local model
snapshot is required: the scoring notebook never downloads weights.

If NF4 is unavailable, investigate the dependency error first. You can
deliberately use `--quantization none` for a separate memory-tested experiment;
it increases memory use and must be verified on the selected GPU. Reduce batch
size for activation-memory pressure. Keep a recorded configuration for each
change rather than relying on an automatic fallback.

## What a successful sample run establishes

The interactive `test.csv` contains only example rows. Kaggle replaces it with
about 4,000 hidden rows for scoring. All predictions are computed from the
current test file, so sample-row counts and IDs are never hard-coded. The
notebooks write exactly `submission.parquet`, with columns `id` and `model`.

A completed example run checks loading, fitting, inference, and output format.
It does not measure hidden accuracy or establish full-test runtime. Keep a
substantial margin below the competition's nine-hour CPU/GPU limit. Review
the runtime status and release the free GPU allocation when finished; stopping
a cell alone does not necessarily release the session. Your account's available
GPU quota is shown by Kaggle.

For validation, keep entire `prompt_id` groups together. Fit learned transforms
and classifiers only inside each training fold. Score balanced accuracy and
inspect per-model recall, particularly confusion between sibling models. If
you later add external passages, exclude host-prompt overlaps from validation
training and audit source identity, licenses, duplicates, and endpoint versions.

The official [overview](https://www.kaggle.com/competitions/name-that-llm/overview)
defines the offline notebook constraints and scoring; the
[data page](https://www.kaggle.com/competitions/name-that-llm/data) describes the
competition inputs. The notebook reads and writes local artifacts only during
training and scoring. It does not publish data, submit predictions, or start
paid services automatically.
