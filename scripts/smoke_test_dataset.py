"""Smoke test the manifest-backed PyTorch dataset and DataLoader."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from torch.utils.data import DataLoader

from crop_grading.constants import CROP_INDEX_TO_LABEL, GRADE_INDEX_TO_LABEL
from crop_grading.data.dataset import CropGradeDataset
from crop_grading.data.transforms import build_eval_transforms
from crop_grading.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Load one dataset batch from a manifest.")
    parser.add_argument("--manifest", default="data/metadata/train_labels.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config(ROOT_DIR / "configs/default.yaml")
    dataset = CropGradeDataset(
        ROOT_DIR / args.manifest,
        project_root=ROOT_DIR,
        transform=build_eval_transforms(config.data.image_size),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    crop_counts = _count_labels(batch["crop_label"].tolist(), CROP_INDEX_TO_LABEL)
    grade_counts = _count_labels(batch["grade_label"].tolist(), GRADE_INDEX_TO_LABEL)

    print(f"Dataset rows: {len(dataset)}")
    print(f"Batch image shape: {tuple(batch['image'].shape)}")
    print(f"Batch crop labels: {crop_counts}")
    print(f"Batch grade labels: {grade_counts}")
    print("Dataset smoke test passed.")
    return 0


def _count_labels(indexes: list[int], label_map: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in indexes:
        label = label_map[index]
        counts[label] = counts.get(label, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
