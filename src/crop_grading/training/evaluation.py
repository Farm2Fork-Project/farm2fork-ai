"""Evaluation utilities for trained crop grading checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from crop_grading.constants import CROP_CLASSES, GRADE_CLASSES
from crop_grading.training.metrics import adjacent_accuracy, confusion_matrix


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate evaluation output."""

    crop_accuracy: float
    grade_accuracy: float
    adjacent_grade_accuracy: float
    crop_confusion: torch.Tensor
    grade_confusion: torch.Tensor
    total_samples: int


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    show_progress: bool = True,
) -> EvaluationResult:
    """Evaluate crop and grade predictions for all batches in a loader."""
    model.eval()
    crop_predictions = []
    crop_targets = []
    grade_predictions = []
    grade_targets = []

    progress = tqdm(loader, desc="evaluate", leave=False, disable=not show_progress)
    for batch in progress:
        images = batch["image"].to(device)
        crop_target = batch["crop_label"].to(device)
        grade_target = batch["grade_label"].to(device)
        crop_logits, grade_logits = model(images)

        crop_predictions.append(crop_logits.argmax(dim=1).cpu())
        crop_targets.append(crop_target.cpu())
        grade_predictions.append(grade_logits.argmax(dim=1).cpu())
        grade_targets.append(grade_target.cpu())
        progress.set_postfix(samples=sum(tensor.numel() for tensor in grade_targets))

    crop_pred = torch.cat(crop_predictions)
    crop_true = torch.cat(crop_targets)
    grade_pred = torch.cat(grade_predictions)
    grade_true = torch.cat(grade_targets)

    crop_accuracy = (crop_pred == crop_true).float().mean().item()
    grade_accuracy = (grade_pred == grade_true).float().mean().item()
    adjacent_grade_acc = adjacent_accuracy(grade_pred, grade_true)

    return EvaluationResult(
        crop_accuracy=crop_accuracy,
        grade_accuracy=grade_accuracy,
        adjacent_grade_accuracy=adjacent_grade_acc,
        crop_confusion=confusion_matrix(crop_pred, crop_true, num_classes=len(CROP_CLASSES)),
        grade_confusion=confusion_matrix(grade_pred, grade_true, num_classes=len(GRADE_CLASSES)),
        total_samples=len(grade_true),
    )


def format_confusion_matrix(matrix: torch.Tensor, labels: tuple[str, ...]) -> str:
    """Format a confusion matrix as a compact markdown-like table."""
    header = ["true\\pred", *labels]
    rows = [" | ".join(header)]
    rows.append(" | ".join(["---"] * len(header)))
    for label, values in zip(labels, matrix.tolist()):
        rows.append(" | ".join([label, *[str(value) for value in values]]))
    return "\n".join(rows)
