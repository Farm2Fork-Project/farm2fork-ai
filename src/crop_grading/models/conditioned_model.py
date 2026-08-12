"""Crop-conditioned, grade-only model.

The crop is known at inference (the farmer selects it), so instead of predicting the crop
we *condition* grading on it: a shared backbone extracts general quality features and a learned
crop embedding tells the grade head which crop's quality rubric to apply. This shares statistical
strength across crops (helping small crops like mango) while allowing crop-specific decisions.
No crop-ID head — grade output only.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("timm is required. Install with `pip install -e .[ml]`.") from exc

from crop_grading.constants import CROP_CLASSES, GRADE_CLASSES


class CropConditionedGradeModel(nn.Module):
    """Shared backbone + crop embedding → grade head."""

    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b0",
        num_crops: int = len(CROP_CLASSES),
        num_grades: int = len(GRADE_CLASSES),
        crop_embed_dim: int = 16,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feature_dim = self.backbone.num_features
        self.crop_embed = nn.Embedding(num_crops, crop_embed_dim)
        self.grade_head = nn.Sequential(
            nn.Linear(feature_dim + crop_embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_grades),
        )

    def forward(self, images: torch.Tensor, crop_index: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        crop_vec = self.crop_embed(crop_index)
        return self.grade_head(torch.cat([features, crop_vec], dim=1))

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
