"""Validation helpers for dataset split manifests."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from crop_grading.constants import CROP_CLASSES, GRADE_CLASSES

REQUIRED_COLUMNS = ("image_path", "crop_label", "grade_label", "source", "split")
VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ManifestIssue:
    """A single manifest validation issue."""

    severity: str
    message: str
    row_number: int | None = None
    file_path: Path | None = None

    def format(self) -> str:
        location = ""
        if self.file_path is not None:
            location = str(self.file_path)
        if self.row_number is not None:
            location = f"{location}:{self.row_number}" if location else f"row {self.row_number}"
        return f"[{self.severity}] {location} {self.message}".strip()


@dataclass(frozen=True)
class ManifestSummary:
    """Summary of one validated manifest."""

    file_path: Path
    rows: int
    split_counts: dict[str, int] = field(default_factory=dict)
    crop_counts: dict[str, int] = field(default_factory=dict)
    grade_counts: dict[str, int] = field(default_factory=dict)
    issues: tuple[ManifestIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def validate_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path = ".",
    check_files: bool = True,
) -> ManifestSummary:
    """Validate one train/validation/test CSV manifest."""
    path = Path(manifest_path)
    root = Path(project_root)
    issues: list[ManifestIssue] = []
    split_counts = {split: 0 for split in VALID_SPLITS}
    crop_counts = {crop: 0 for crop in CROP_CLASSES}
    grade_counts = {grade: 0 for grade in GRADE_CLASSES}
    seen_images: set[str] = set()
    rows = 0

    if not path.exists():
        return ManifestSummary(
            file_path=path,
            rows=0,
            issues=(ManifestIssue("error", "manifest file does not exist", file_path=path),),
        )

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            issues.append(
                ManifestIssue(
                    "error",
                    f"missing required columns: {', '.join(missing_columns)}",
                    file_path=path,
                )
            )
            return ManifestSummary(file_path=path, rows=0, issues=tuple(issues))

        for row_number, row in enumerate(reader, start=2):
            rows += 1
            image_path = (row.get("image_path") or "").strip()
            crop_label = (row.get("crop_label") or "").strip().lower()
            grade_label = (row.get("grade_label") or "").strip().upper()
            source = (row.get("source") or "").strip()
            split = (row.get("split") or "").strip().lower()

            if not image_path:
                issues.append(_row_error(path, row_number, "image_path is required"))
            elif image_path in seen_images:
                issues.append(_row_error(path, row_number, f"duplicate image_path: {image_path}"))
            else:
                seen_images.add(image_path)
                if check_files and not _resolve_image_path(root, image_path).is_file():
                    issues.append(_row_error(path, row_number, f"image file not found: {image_path}"))

            if crop_label not in CROP_CLASSES:
                issues.append(_row_error(path, row_number, f"invalid crop_label: {crop_label}"))
            else:
                crop_counts[crop_label] += 1

            if grade_label not in GRADE_CLASSES:
                issues.append(_row_error(path, row_number, f"invalid grade_label: {grade_label}"))
            else:
                grade_counts[grade_label] += 1

            if split not in VALID_SPLITS:
                issues.append(_row_error(path, row_number, f"invalid split: {split}"))
            else:
                split_counts[split] += 1

            if not source:
                issues.append(_row_error(path, row_number, "source is required"))

    if rows == 0:
        issues.append(ManifestIssue("error", "manifest contains no rows", file_path=path))

    return ManifestSummary(
        file_path=path,
        rows=rows,
        split_counts={key: value for key, value in split_counts.items() if value},
        crop_counts={key: value for key, value in crop_counts.items() if value},
        grade_counts={key: value for key, value in grade_counts.items() if value},
        issues=tuple(issues),
    )


def validate_manifest_directory(
    metadata_dir: str | Path,
    *,
    project_root: str | Path = ".",
    check_files: bool = True,
) -> list[ManifestSummary]:
    """Validate the standard train, validation, and test manifest files."""
    metadata_path = Path(metadata_dir)
    manifest_names = ("train_labels.csv", "val_labels.csv", "test_labels.csv")
    return [
        validate_manifest(metadata_path / name, project_root=project_root, check_files=check_files)
        for name in manifest_names
    ]


def find_cross_split_duplicates(metadata_dir: str | Path) -> dict[str, list[str]]:
    """Find image paths that appear in more than one standard split manifest."""
    metadata_path = Path(metadata_dir)
    split_files = {
        "train": metadata_path / "train_labels.csv",
        "val": metadata_path / "val_labels.csv",
        "test": metadata_path / "test_labels.csv",
    }
    path_to_splits: dict[str, list[str]] = {}

    for split, manifest_path in split_files.items():
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_path = (row.get("image_path") or "").strip()
                if image_path:
                    path_to_splits.setdefault(image_path, []).append(split)

    return {
        image_path: splits
        for image_path, splits in path_to_splits.items()
        if len(set(splits)) > 1
    }


def _row_error(path: Path, row_number: int, message: str) -> ManifestIssue:
    return ManifestIssue("error", message, row_number=row_number, file_path=path)


def _resolve_image_path(project_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return project_root / path
