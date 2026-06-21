"""PyTorch dataset backed by crop grading manifest CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from crop_grading.constants import CROP_LABEL_TO_INDEX, GRADE_LABEL_TO_INDEX
from crop_grading.data.manifest import REQUIRED_COLUMNS, validate_manifest


@dataclass(frozen=True)
class CropGradeSample:
    """One manifest sample with both text labels and integer targets."""

    image_path: Path
    crop_label: str
    crop_index: int
    grade_label: str
    grade_index: int
    source: str
    split: str


class CropGradeDataset(Dataset):
    """Load crop images and multi-task labels from a manifest CSV."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        project_root: str | Path = ".",
        transform: Callable | None = None,
        validate: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root)
        self.transform = transform

        if validate:
            summary = validate_manifest(self.manifest_path, project_root=self.project_root)
            if not summary.is_valid:
                formatted_issues = "\n".join(issue.format() for issue in summary.issues)
                raise ValueError(f"Invalid manifest {self.manifest_path}:\n{formatted_issues}")

        self.samples = self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return {
            "image": image,
            "crop_label": sample.crop_index,
            "grade_label": sample.grade_index,
            "crop_name": sample.crop_label,
            "grade_name": sample.grade_label,
            "image_path": str(sample.image_path),
            "source": sample.source,
            "split": sample.split,
        }

    def _load_samples(self) -> list[CropGradeSample]:
        samples = []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

            for row in reader:
                image_path = self._resolve_path(row["image_path"].strip())
                crop_label = row["crop_label"].strip().lower()
                grade_label = row["grade_label"].strip().upper()
                samples.append(
                    CropGradeSample(
                        image_path=image_path,
                        crop_label=crop_label,
                        crop_index=CROP_LABEL_TO_INDEX[crop_label],
                        grade_label=grade_label,
                        grade_index=GRADE_LABEL_TO_INDEX[grade_label],
                        source=row["source"].strip(),
                        split=row["split"].strip().lower(),
                    )
                )
        return samples

    def _resolve_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.project_root / path
