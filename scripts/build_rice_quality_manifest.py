"""Build train/val/test manifests for the rice quality dataset."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from crop_grading.data.manifest import validate_manifest

QUALITY_TO_GRADE = {
    "full": "A",
    "mixed": "C",
    "broken": "D",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_NAME = "rice_quality_broken_full_mixed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create rice quality train/val/test manifests.")
    parser.add_argument(
        "--raw-dir",
        default="data/raw/Rice/rice_quality",
        help="Folder containing broken/full/mixed subfolders.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/metadata",
        help="Folder where train_labels.csv, val_labels.csv, and test_labels.csv are written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    raw_dir = ROOT_DIR / args.raw_dir
    metadata_dir = ROOT_DIR / args.metadata_dir
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(raw_dir)
    if not rows:
        print(f"No images found under {raw_dir}")
        return 1

    split_rows = split_by_grade(rows, args.seed, args.train_ratio, args.val_ratio)
    for split, split_data in split_rows.items():
        output_path = metadata_dir / f"{split}_labels.csv"
        write_manifest(output_path, split_data)
        summary = validate_manifest(output_path, project_root=ROOT_DIR)
        print(f"{output_path}: {summary.rows} rows, valid={summary.is_valid}")
        if not summary.is_valid:
            for issue in summary.issues:
                print(f"  {issue.format()}")
            return 1

    print("Rice quality manifests created.")
    print("Grade B is intentionally absent because this source has no B-quality class.")
    return 0


def collect_rows(raw_dir: Path) -> list[dict[str, str]]:
    rows = []
    for quality_label, grade_label in QUALITY_TO_GRADE.items():
        quality_dir = raw_dir / quality_label
        if not quality_dir.is_dir():
            raise FileNotFoundError(f"Expected folder not found: {quality_dir}")

        for image_path in sorted(quality_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "image_path": image_path.relative_to(ROOT_DIR).as_posix(),
                        "crop_label": "rice",
                        "grade_label": grade_label,
                        "source": SOURCE_NAME,
                        "split": "",
                    }
                )
    return rows


def split_by_grade(
    rows: list[dict[str, str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    split_rows = {"train": [], "val": [], "test": []}

    for grade in sorted({row["grade_label"] for row in rows}):
        grade_rows = [row for row in rows if row["grade_label"] == grade]
        rng.shuffle(grade_rows)

        train_end = int(len(grade_rows) * train_ratio)
        val_end = train_end + int(len(grade_rows) * val_ratio)
        grade_splits = {
            "train": grade_rows[:train_end],
            "val": grade_rows[train_end:val_end],
            "test": grade_rows[val_end:],
        }

        for split, split_data in grade_splits.items():
            for row in split_data:
                row = dict(row)
                row["split"] = split
                split_rows[split].append(row)

    for split_data in split_rows.values():
        split_data.sort(key=lambda row: row["image_path"])
    return split_rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "crop_label", "grade_label", "source", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
