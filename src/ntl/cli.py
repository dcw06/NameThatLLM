"""Command-line entry points; no credentials, network calls or submissions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from .constants import DEFAULT_SEED


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--char-max-features", type=int, default=50000)
    parser.add_argument("--word-max-features", type=int, default=30000)
    parser.add_argument("--C", type=float, default=4.0)
    parser.add_argument("--max-iter", type=int, default=1000)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Name That LLM: offline data checks and CPU baseline")
    subcommands = result.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="Report interpreter and dependency versions")
    check = subcommands.add_parser("data-check", help="Validate official data schemas without altering text")
    check.add_argument("--train", required=True, type=Path)
    check.add_argument("--test", type=Path)
    cv = subcommands.add_parser("baseline-cv", help="Evaluate with prompt-grouped folds and save OOF predictions")
    _model_arguments(cv)
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--fold-manifest", type=Path)
    training = subcommands.add_parser("baseline-train", help="Fit on all labeled data and save an inference artifact")
    _model_arguments(training)
    predict = subcommands.add_parser("predict", help="Write a validated submission.parquet from a trusted local model")
    predict.add_argument("--model", required=True, type=Path)
    predict.add_argument("--test", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    smoke = subcommands.add_parser("smoke", help="Exercise the pipeline on clearly synthetic fixtures")
    smoke.add_argument("--output-dir", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            versions = {}
            for package in ("pandas", "numpy", "scipy", "scikit-learn", "joblib", "pyarrow"):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = None
            print(json.dumps({"python": platform.python_version(), "executable": sys.executable,
                              "dependencies": versions}, indent=2))
            return int(any(value is None for value in versions.values()))
        if args.command == "data-check":
            from .data import data_quality_report, fingerprint_frame, read_test, read_train
            train = read_train(args.train)
            summary = {"training_rows": len(train), "prompts": int(train["prompt_id"].nunique()),
                       "class_counts": train["model"].value_counts().sort_index().to_dict(),
                       "train_fingerprint": fingerprint_frame(train),
                       "data_quality": data_quality_report(train)}
            if args.test:
                test = read_test(args.test)
                if set(train["id"]) & set(test["id"]):
                    raise ValueError("Train and test passage ids overlap")
                summary["test_rows"] = len(test)
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "smoke":
            from .smoke import run_smoke
            summary = run_smoke(args.output_dir)
        elif args.command in {"baseline-cv", "baseline-train"}:
            from .data import read_train
            from .modeling import BaselineConfig, run_cross_validation, train_baseline
            config = BaselineConfig(seed=args.seed, char_max_features=args.char_max_features,
                                    word_max_features=args.word_max_features, C=args.C, max_iter=args.max_iter)
            train = read_train(args.train)
            if args.command == "baseline-cv":
                summary = run_cross_validation(train, args.output_dir, n_splits=args.folds,
                                               config=config, fold_manifest=args.fold_manifest)
            else:
                summary = {"model_path": str(train_baseline(train, args.output_dir, config=config))}
        else:
            from .data import read_test
            from .modeling import predict_baseline
            submission = predict_baseline(args.model, read_test(args.test), args.output)
            summary = {"submission_path": str(args.output), "rows": len(submission)}
        print(json.dumps(summary, indent=2))
        return 0
    except (ValueError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
