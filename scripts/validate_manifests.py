"""Validate dataset manifest CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from crop_grading.data.manifest import ManifestSummary, validate_manifest, validate_manifest_directory
from crop_grading.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate crop grading dataset manifests.")
    parser.add_argument(
        "manifest",
        nargs="?",
        help="Optional CSV path. If omitted, validates train/val/test files from config.",
    )
    parser.add_argument(
        "--no-file-check",
        action="store_true",
        help="Validate labels and schema without checking image files exist.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Return success when standard manifests do not exist yet.",
    )
    args = parser.parse_args()

    config = load_config(ROOT_DIR / "configs/default.yaml")
    if args.manifest:
        summaries = [
            validate_manifest(
                args.manifest,
                project_root=ROOT_DIR,
                check_files=not args.no_file_check,
            )
        ]
    else:
        summaries = validate_manifest_directory(
            ROOT_DIR / config.data.metadata_dir,
            project_root=ROOT_DIR,
            check_files=not args.no_file_check,
        )

    has_errors = False
    for summary in summaries:
        if _allowed_missing_manifest(summary, args.allow_missing):
            status = "missing allowed"
        else:
            status = "valid" if summary.is_valid else "invalid"
        print(f"{summary.file_path}: {status} ({summary.rows} rows)")
        if summary.crop_counts:
            print(f"  crops: {summary.crop_counts}")
        if summary.grade_counts:
            print(f"  grades: {summary.grade_counts}")
        if summary.split_counts:
            print(f"  splits: {summary.split_counts}")
        for issue in summary.issues:
            print(f"  {issue.format()}")
        if not _allowed_missing_manifest(summary, args.allow_missing):
            has_errors = has_errors or not summary.is_valid

    return 1 if has_errors else 0


def _allowed_missing_manifest(summary: ManifestSummary, allow_missing: bool) -> bool:
    if not allow_missing:
        return False
    return all(issue.message == "manifest file does not exist" for issue in summary.issues)


if __name__ == "__main__":
    raise SystemExit(main())
