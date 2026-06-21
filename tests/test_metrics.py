import pytest

torch = pytest.importorskip("torch")

from crop_grading.training.metrics import accuracy_from_logits


def test_accuracy_from_logits() -> None:
    logits = torch.tensor([[0.1, 0.9], [2.0, 1.0], [0.2, 0.8]])
    targets = torch.tensor([1, 1, 1])

    assert accuracy_from_logits(logits, targets) == pytest.approx(2 / 3)
