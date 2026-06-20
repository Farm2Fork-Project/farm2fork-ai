"""Class weighting helpers for imbalanced crop grading datasets."""

from __future__ import annotations

import torch
from torch.utils.data import Subset

from crop_grading.data.dataset import CropGradeDataset


def compute_class_weights_from_subset(
    dataset: CropGradeDataset,
    subset: Subset,
    *,
    label_type: str,
    num_classes: int,
) -> torch.Tensor:
    """Compute normalized inverse-frequency class weights for a subset."""
    if label_type not in {"crop", "grade"}:
        raise ValueError("label_type must be 'crop' or 'grade'")

    counts = torch.zeros(num_classes, dtype=torch.float32)
    for index in subset.indices:
        sample = dataset.samples[index]
        class_index = sample.crop_index if label_type == "crop" else sample.grade_index
        counts[class_index] += 1

    weights = torch.zeros_like(counts)
    nonzero = counts > 0
    weights[nonzero] = 1.0 / counts[nonzero]
    if weights.sum() > 0:
        weights = weights / weights.sum() * nonzero.sum().float()
    return weights
