import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from crop_grading.models.multitask_model import CropGradingModel
from crop_grading.training.losses import MultiTaskLoss


def test_crop_grading_model_forward_shapes() -> None:
    model = CropGradingModel(
        backbone_name="efficientnet_b0",
        num_crops=6,
        num_grades=4,
        pretrained=False,
    )
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        crop_logits, grade_logits = model(images)

    assert tuple(crop_logits.shape) == (2, 6)
    assert tuple(grade_logits.shape) == (2, 4)


def test_multitask_loss_returns_components() -> None:
    loss_fn = MultiTaskLoss(crop_weight=0.4, grade_weight=0.6)
    crop_logits = torch.randn(3, 6)
    grade_logits = torch.randn(3, 4)
    crop_targets = torch.tensor([0, 1, 2])
    grade_targets = torch.tensor([0, 1, 3])

    losses = loss_fn(crop_logits, grade_logits, crop_targets, grade_targets)

    assert set(losses) == {"loss", "crop_loss", "grade_loss"}
    assert losses["loss"].item() > 0
