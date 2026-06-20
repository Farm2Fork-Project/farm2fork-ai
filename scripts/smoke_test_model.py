"""Smoke test model forward pass and multi-task loss."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required. Install ML dependencies with: pip install -e .[ml]") from exc

from crop_grading.models.multitask_model import CropGradingModel
from crop_grading.training.losses import MultiTaskLoss
from crop_grading.utils.config import load_config


def main() -> int:
    config = load_config(ROOT_DIR / "configs/default.yaml")
    model = CropGradingModel(
        backbone_name=config.model.backbone,
        num_crops=config.model.num_crops,
        num_grades=config.model.num_grades,
        dropout_rate=config.model.dropout_rate,
        pretrained=False,
    )
    model.eval()

    batch_size = 2
    images = torch.randn(batch_size, 3, config.data.image_size, config.data.image_size)
    crop_targets = torch.tensor([0, 1], dtype=torch.long)
    grade_targets = torch.tensor([0, 2], dtype=torch.long)

    with torch.no_grad():
        crop_logits, grade_logits = model(images)

    loss_fn = MultiTaskLoss(
        crop_weight=config.training.crop_loss_weight,
        grade_weight=config.training.grade_loss_weight,
    )
    losses = loss_fn(crop_logits, grade_logits, crop_targets, grade_targets)

    print(f"Crop logits shape: {tuple(crop_logits.shape)}")
    print(f"Grade logits shape: {tuple(grade_logits.shape)}")
    print(f"Loss: {losses['loss'].item():.4f}")
    print("Model smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
