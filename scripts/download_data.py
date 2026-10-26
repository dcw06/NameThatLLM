"""Download official data using existing Kaggle auth; never join or accept rules."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "Configure Kaggle credentials locally, accept the competition rules on Kaggle, "
            "then run: uv run --extra kaggle python scripts/download_data.py. "
            "Alternatively download the files through Kaggle's website."
        ) from exc
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected = {"train.csv", "test.csv", "sample_submission.parquet"}
    if any((output / name).exists() for name in expected):
        raise SystemExit("Official data already exists in the destination; refusing to overwrite it.")
    api = KaggleApi()
    try:
        api.authenticate()
        api.competition_download_files("name-that-llm", path=str(output), quiet=False)
    except Exception as exc:
        # Avoid printing response bodies or credential-bearing exception strings.
        raise SystemExit(
            "Kaggle download did not complete. Check local credentials and rules acceptance "
            "on Kaggle. No competition rules were accepted by this helper."
        ) from exc
    archive = output / "name-that-llm.zip"
    with zipfile.ZipFile(archive) as bundle:
        members = {member.filename: member for member in bundle.infolist()}
        if not expected.issubset(members):
            raise SystemExit("Downloaded archive is missing expected official files.")
        for name in sorted(expected):
            member = members[name]
            if member.is_dir() or member.file_size > 200_000_000:
                raise SystemExit("Unexpected official file structure or size; inspect archive first.")
        for name in sorted(expected):
            # Only these fixed, flat names are extracted, preventing path traversal.
            with bundle.open(members[name]) as source, (output / name).open("xb") as target:
                target.write(source.read())
    print(f"Downloaded official files to {output}. Next: make check-data")


if __name__ == "__main__":
    main()
