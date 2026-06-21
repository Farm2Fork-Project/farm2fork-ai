"""Loss functions for crop grading training."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Weighted crop classification and grade classification cross entropy."""

    def __init__(
        self,
        *,
        crop_weight: float = 0.4,
        grade_weight: float = 0.6,
        crop_class_weights: torch.Tensor | None = None,
        grade_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if crop_weight < 0 or grade_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if crop_weight + grade_weight <= 0:
            raise ValueError("at least one loss weight must be positive")

        self.crop_weight = crop_weight
        self.grade_weight = grade_weight
        self.register_buffer("crop_class_weights", crop_class_weights)
        self.register_buffer("grade_class_weights", grade_class_weights)

    def forward(
        self,
        crop_logits: torch.Tensor,
        grade_logits: torch.Tensor,
        crop_targets: torch.Tensor,
        grade_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        crop_loss = F.cross_entropy(
            crop_logits,
            crop_targets,
            weight=self.crop_class_weights,
        )
        grade_loss = F.cross_entropy(
            grade_logits,
            grade_targets,
            weight=self.grade_class_weights,
        )
        total_loss = (self.crop_weight * crop_loss) + (self.grade_weight * grade_loss)
        return {
            "loss": total_loss,
            "crop_loss": crop_loss,
            "grade_loss": grade_loss,
        }
