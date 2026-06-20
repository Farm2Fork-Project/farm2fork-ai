import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from crop_grading.training.evaluation import evaluate_model, format_confusion_matrix
from crop_grading.training.metrics import adjacent_accuracy, confusion_matrix


class FixedModel(torch.nn.Module):
    def forward(self, images):
        batch_size = images.shape[0]
        crop_logits = torch.zeros(batch_size, 6)
        grade_logits = torch.zeros(batch_size, 4)
        crop_logits[:, 1] = 1.0
        grade_logits[:, 2] = 1.0
        return crop_logits, grade_logits


class DictDataset(TensorDataset):
    def __getitem__(self, index):
        image, crop_label, grade_label = super().__getitem__(index)
        return {
            "image": image,
            "crop_label": crop_label,
            "grade_label": grade_label,
        }


def test_adjacent_accuracy_and_confusion_matrix() -> None:
    predictions = torch.tensor([0, 1, 3])
    targets = torch.tensor([0, 2, 1])

    assert adjacent_accuracy(predictions, targets) == pytest.approx(2 / 3)
    matrix = confusion_matrix(predictions, targets, num_classes=4)
    assert matrix[0, 0] == 1
    assert matrix[2, 1] == 1
    assert matrix[1, 3] == 1


def test_evaluate_model_returns_aggregate_metrics() -> None:
    images = torch.randn(3, 3, 2, 2)
    crop_labels = torch.tensor([1, 1, 0])
    grade_labels = torch.tensor([2, 1, 2])
    loader = DataLoader(DictDataset(images, crop_labels, grade_labels), batch_size=2)

    result = evaluate_model(FixedModel(), loader, device=torch.device("cpu"), show_progress=False)

    assert result.total_samples == 3
    assert result.crop_accuracy == pytest.approx(2 / 3)
    assert result.grade_accuracy == pytest.approx(2 / 3)
    assert result.adjacent_grade_accuracy == pytest.approx(1.0)
    assert "true\\pred" in format_confusion_matrix(result.grade_confusion, ("A", "B", "C", "D"))
