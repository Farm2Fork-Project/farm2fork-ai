"""Dataset sampling helpers."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Subset, WeightedRandomSampler

from crop_grading.data.dataset import CropGradeDataset


def sample_subset_indexes(
    dataset: CropGradeDataset,
    *,
    max_samples: int | None,
    seed: int,
) -> list[int]:
    """Sample indexes across crop/grade groups to keep small runs diverse."""
    if max_samples is None or max_samples >= len(dataset):
        return list(range(len(dataset)))

    grouped: dict[tuple[str, str], list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        grouped.setdefault((sample.crop_label, sample.grade_label), []).append(index)

    rng = random.Random(seed)
    selected = []
    group_quota = max(1, max_samples // max(1, len(grouped)))
    for indexes in grouped.values():
        indexes = list(indexes)
        rng.shuffle(indexes)
        selected.extend(indexes[:group_quota])

    if len(selected) < max_samples:
        selected_set = set(selected)
        remaining = [index for index in range(len(dataset)) if index not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)])

    rng.shuffle(selected)
    return selected[:max_samples]


def build_balanced_sampler(
    dataset: CropGradeDataset,
    subset: Subset,
    *,
    balance_by: str = "crop_grade",
    seed: int = 42,
) -> WeightedRandomSampler:
    """Build a replacement sampler weighted by inverse group frequency."""
    if balance_by not in {"crop_grade", "grade", "crop"}:
        raise ValueError("balance_by must be one of: crop_grade, grade, crop")

    group_counts: dict[tuple[str, ...], int] = {}
    groups = []
    for index in subset.indices:
        sample = dataset.samples[index]
        if balance_by == "crop_grade":
            group = (sample.crop_label, sample.grade_label)
        elif balance_by == "grade":
            group = (sample.grade_label,)
        else:
            group = (sample.crop_label,)
        groups.append(group)
        group_counts[group] = group_counts.get(group, 0) + 1

    weights = torch.tensor([1.0 / group_counts[group] for group in groups], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
