# Kaggle API tokens and the first submission

The installed client is locked in `uv.lock`. To add it to the local environment:

```sh
uv --cache-dir .cache/uv sync --locked --extra dev --extra neural --extra kaggle
```

Create an API token in [Kaggle settings](https://www.kaggle.com/settings/api).
Store the token in `~/.kaggle/access_token`, readable only by your user (`chmod 600`).
The Kaggle client reads that file automatically. Never put a token in source, notebook
metadata, a command argument, or a Git remote URL. Tokens do not belong in chat.
The project needs no token inside a Kaggle notebook.

Accept the [competition rules](https://www.kaggle.com/competitions/name-that-llm/rules)
in your own browser. Authentication alone does not accept the rules. Download and validate:

```sh
.venv/bin/python scripts/download_data.py
.venv/bin/python -m ntl data-check --train data/raw/train.csv --test data/raw/test.csv
.venv/bin/python -m ntl baseline-cv --train data/raw/train.csv --output-dir artifacts/cpu-cv
```

For an initial submission without preparing model weights or dependency wheel datasets,
generate the private CPU notebook. Replace `YOUR_USERNAME` with your Kaggle username:

```sh
.venv/bin/python scripts/prepare_kaggle_submission.py --username YOUR_USERNAME
.venv/bin/kaggle kernels push -p artifacts/first_submission -t 32400
.venv/bin/kaggle kernels status YOUR_USERNAME/name-that-llm-cpu-first-submission
```

The push saves and executes a notebook; it does not submit predictions for scoring.
This first notebook uses Kaggle's installed CPU dependencies and records their versions.
It fits and consumes its model in that same run, runs prompt-grouped CV, and writes
`submission.parquet`. The other generated notebooks retain the isolated, locked dependency
workflow documented in [kaggle.md](kaggle.md).

After that saved notebook version completes successfully, submit its output, specifying
the actual version number returned by the push:

```sh
.venv/bin/kaggle competitions submit name-that-llm \
  -k YOUR_USERNAME/name-that-llm-cpu-first-submission \
  -v VERSION_NUMBER -f submission.parquet \
  -m "CPU baseline with prompt-grouped validation"
.venv/bin/kaggle competitions submissions name-that-llm
```

The competition reruns the notebook on hidden test data. Uploading the three-row local
example file directly would not submit predictions for the hidden test. Do not repeat
the submit command while scoring is pending: it uses another daily submission slot.

Official sources: [authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication),
[code competition submission](https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md#tutorial-how-to-submit-to-a-code-competition).
