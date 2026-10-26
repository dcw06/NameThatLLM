# First official-data CPU run

Run on October 26, 2026. Source commit:
`7a476831891f7fdb7504b14ad55aadaaa1872a14`.

The first submitted model is the CPU reference: case-preserving character and word
TF-IDF plus stylometry, followed by class-weighted logistic regression. It does not
use the selected Qwen backbone, extra training data, or a GPU.

| Measurement | Balanced accuracy |
| --- | ---: |
| Local five-fold prompt-grouped CV | 0.363368 |
| Kaggle five-fold prompt-grouped CV | 0.359667 |
| Kaggle public leaderboard, submission 57676158 | 0.360 |

Both runs use the exact same 2,699 rows and fold assignments, with 150 prompt groups,
five validation folds, and seed 2026. There are no exact duplicate passages. Each
validation fold holds out 30 complete prompts; learned transforms are fitted only on
its training partition. The official training-content SHA-256 fingerprint is
`d76a05125fad49d450e61ec43e78ac5fee9d6a291607386a73ce1046511a301a`.

The environments differ: local Python 3.12.13 / NumPy 2.3.5 / scikit-learn 1.7.2,
versus Kaggle Python 3.13.15 / NumPy 2.1.3 / scikit-learn 1.6.1. Record these results
separately; the first-submission notebook reports its environment and fits and consumes
the model within that same environment. The isolated-wheel notebooks remain available
for subsequent experiments requiring exact package pins.

The private notebook is
[daichongwei06/name-that-llm-cpu-first-submission](https://www.kaggle.com/code/daichongwei06/name-that-llm-cpu-first-submission),
version **1**. Its saved run completed and its three example predictions passed schema,
label, and ID checks. Kaggle submission **57676158** was created from that exact version;
the three-row example file was not uploaded as a stand-alone prediction submission.

Submission **57676158** completed successfully with public balanced accuracy **0.360**.
The private score is unavailable until competition results are released. Check the
existing submission without creating another:

```sh
.venv/bin/kaggle competitions submission 57676158
```

Local run files are in `artifacts/first-real-cv/`, downloaded Kaggle reports are in
`artifacts/kaggle-version-1/`, and submission state is recorded in
`artifacts/first_submission/run_state.json`. Those files, competition data, and
credentials are excluded from Git. This run used one submission slot and no GPU quota.
