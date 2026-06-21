"""Small metric helpers for training and smoke tests."""

from __future__ import annotations

import torch


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy for a batch."""
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def adjacent_accuracy(predictions: torch.Tensor, targets: torch.Tensor, *, max_distance: int = 1) -> float:
    """Compute grade accuracy where neighboring grade mistakes count as acceptable."""
    return ((predictions - targets).abs() <= max_distance).float().mean().item()


def adjacent_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_distance: int = 1,
) -> float:
    """Compute adjacent accuracy from classification logits."""
    predictions = logits.argmax(dim=1)
    return adjacent_accuracy(predictions, targets, max_distance=max_distance)


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    """Create a row=true, column=predicted confusion matrix."""
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(targets.view(-1), predictions.view(-1)):
        matrix[target.long(), prediction.long()] += 1
    return matrix


class RunningAverage:
    """Weighted running average for epoch metrics."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n

    @property
    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count
