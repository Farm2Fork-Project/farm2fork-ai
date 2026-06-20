"""Multi-task crop identification and quality grading model."""

from __future__ import annotations

import torch
from torch import nn

try:
    import timm
except ImportError as exc:  # pragma: no cover - exercised only without ml deps installed
    raise ImportError("timm is required for CropGradingModel. Install with `pip install -e .[ml]`.") from exc

from crop_grading.constants import CROP_CLASSES, GRADE_CLASSES


class ClassificationHead(nn.Module):
    """Small MLP classification head used for each task."""

    def __init__(self, feature_dim: int, output_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class CropGradingModel(nn.Module):
    """Shared EfficientNet backbone with crop ID and quality grade heads."""

    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b3",
        num_crops: int = len(CROP_CLASSES),
        num_grades: int = len(GRADE_CLASSES),
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features
        self.crop_head = ClassificationHead(feature_dim, num_crops, dropout_rate)
        self.grade_head = ClassificationHead(feature_dim, num_grades, dropout_rate)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        crop_logits = self.crop_head(features)
        grade_logits = self.grade_head(features)
        return crop_logits, grade_logits

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters for warm-up training."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone_top(self, fraction: float = 0.3) -> None:
        """Unfreeze approximately the top fraction of backbone child modules."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be > 0 and <= 1")

        self.freeze_backbone()
        layers = list(self.backbone.children())
        if not layers:
            return

        unfreeze_from = max(0, int(len(layers) * (1.0 - fraction)))
        for layer in layers[unfreeze_from:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
