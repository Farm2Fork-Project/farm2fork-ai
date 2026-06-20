"""Dataset sampling helpers."""

from __future__ import annotations

import random

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
