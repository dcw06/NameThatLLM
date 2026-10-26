UV := uv --cache-dir .cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.tools/python
export PYTHONPATH := $(CURDIR)/src
TRAIN ?= data/raw/train.csv
TEST ?= data/raw/test.csv

.PHONY: setup verify-local smoke doctor check-data cv train predict notebooks neural-setup
setup:
	$(UV) sync --locked --extra dev
verify-local:
	$(UV) run --locked --no-sync --extra dev ruff check src tests scripts
	$(UV) run --locked --no-sync --extra dev python -m pytest -W error::RuntimeWarning
	$(UV) run --locked --no-sync --extra dev python scripts/build_kaggle_notebooks.py --check
smoke:
	$(UV) run --locked ntl smoke --output-dir artifacts/smoke
doctor:
	$(UV) run --locked ntl doctor
check-data:
	$(UV) run --locked ntl data-check --train $(TRAIN) --test $(TEST)
cv:
	$(UV) run --locked ntl baseline-cv --train $(TRAIN) --output-dir artifacts/cpu-cv
train:
	$(UV) run --locked ntl baseline-train --train $(TRAIN) --output-dir artifacts/cpu-model
predict:
	$(UV) run --locked ntl predict --model artifacts/cpu-model/model.joblib --test $(TEST) --output artifacts/submission.parquet
notebooks:
	$(UV) run --locked --extra dev python scripts/build_kaggle_notebooks.py
neural-setup:
	$(UV) sync --locked --extra dev --extra neural
