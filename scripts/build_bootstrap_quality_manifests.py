"""Build bootstrap train/val/test manifests from confirmed quality datasets."""

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CONFIRMED_SOURCES = (
    {
        "crop_label": "rice",
        "root": "data/raw/Rice/rice_quality",
        "source": "rice_quality_broken_full_mixed",
        "mapping": {
            "full": "A",
            "mixed": "C",
            "broken": "D",
        },
    },
    {
        "crop_label": "mango",
        "root": "data/raw/mango/Grading_dataset",
        "source": "pakistani_mango_grading",
        "mapping": {
            "Extra_Class": "A",
            "Class_I": "B",
            "Class_II": "C",
        },
    },
)

GRAINSET_SOURCES = (
    {
        "crop_label": "wheat",
        "root": "data/raw/wheat",
        "source": "grainset_wheat_quality",
        "mapping": {
            "0_NOR": "A",
            "5_BN": "B",
            "6_BP": "B",
            "4_AP": "C",
            "2_SD": "C",
            "1_F_and_S": "D",
            "3_MY": "D",
        },
        "impurity_class": "7_IM",
    },
    {
        "crop_label": "maize",
        "root": "data/raw/maize",
        "source": "grainset_maize_quality",
        "mapping": {
            "0_NOR": "A",
            "5_BN": "B",
            "6_HD": "B",
            "4_AP": "C",
            "2_SD": "C",
            "1_F_and_S": "D",
            "3_MY": "D",
        },
        "impurity_class": "7_IM",
    },
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create manifests from quality datasets with confirmed grade mappings."
    )
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--include-impurities",
        action="store_true",
        help="Include GrainSet 7_IM impurity images as grade D.",
    )
    args = parser.parse_args()

    rows = []
    included: list[str] = []
    for source in CONFIRMED_SOURCES:
        if not (ROOT_DIR / source["root"]).is_dir():
            print(f"[skip] {source['crop_label']}: missing {source['root']}")
            continue
        rows.extend(collect_source_rows(source))
        included.append(source["crop_label"])
    for source in GRAINSET_SOURCES:
        if not (ROOT_DIR / source["root"] / "train").is_dir():
            print(
                f"[skip] {source['crop_label']}: missing {source['root']}/train "
                "(GrainSet images not downloaded yet)"
            )
            continue
        rows.extend(collect_grainset_rows(source, include_impurities=args.include_impurities))
        included.append(source["crop_label"])

    if not rows:
        print("No usable source datasets found under data/raw/. Nothing to build.")
        return 1

    split_rows = split_by_crop_and_grade(rows, args.seed, args.train_ratio, args.val_ratio)
    metadata_dir = ROOT_DIR / args.metadata_dir
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for split, split_data in split_rows.items():
        output_path = metadata_dir / f"{split}_labels.csv"
        write_manifest(output_path, split_data)
        summary = validate_manifest(output_path, project_root=ROOT_DIR)
        print(f"{output_path}: {summary.rows} rows, valid={summary.is_valid}")
        print(f"  crops: {summary.crop_counts}")
        print(f"  grades: {summary.grade_counts}")
        if not summary.is_valid:
            for issue in summary.issues:
                print(f"  {issue.format()}")
            return 1

    print("Bootstrap quality manifests created from confirmed mappings.")
    print(f"Included crops: {', '.join(included)}.")
    if not args.include_impurities:
        print("Excluded: GrainSet 7_IM impurity images.")
    return 0


def collect_source_rows(source: dict) -> list[dict[str, str]]:
    root = ROOT_DIR / source["root"]
    rows = []
    for class_name, grade_label in source["mapping"].items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected folder not found: {class_dir}")
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "image_path": image_path.relative_to(ROOT_DIR).as_posix(),
                        "crop_label": source["crop_label"],
                        "grade_label": grade_label,
                        "source": source["source"],
                        "split": "",
                    }
                )
    return rows


def collect_grainset_rows(source: dict, *, include_impurities: bool) -> list[dict[str, str]]:
    mapping = dict(source["mapping"])
    if include_impurities:
        mapping[source["impurity_class"]] = "D"

    root = ROOT_DIR / source["root"]
    rows = []
    for raw_split in ("train", "test"):
        split_dir = root / raw_split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Expected folder not found: {split_dir}")
        for class_name, grade_label in mapping.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected folder not found: {class_dir}")
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append(
                        {
                            "image_path": image_path.relative_to(ROOT_DIR).as_posix(),
                            "crop_label": source["crop_label"],
                            "grade_label": grade_label,
                            "source": source["source"],
                            "split": "",
                        }
                    )
    return rows


def split_by_crop_and_grade(
    rows: list[dict[str, str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    split_rows = {"train": [], "val": [], "test": []}
    keys = sorted({(row["crop_label"], row["grade_label"]) for row in rows})

    for crop_label, grade_label in keys:
        group = [
            row for row in rows if row["crop_label"] == crop_label and row["grade_label"] == grade_label
        ]
        rng.shuffle(group)

        train_end = int(len(group) * train_ratio)
        val_end = train_end + int(len(group) * val_ratio)
        for split, split_data in {
            "train": group[:train_end],
            "val": group[train_end:val_end],
            "test": group[val_end:],
        }.items():
            for row in split_data:
                row = dict(row)
                row["split"] = split
                split_rows[split].append(row)

    for split_data in split_rows.values():
        split_data.sort(key=lambda row: (row["crop_label"], row["grade_label"], row["image_path"]))
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
