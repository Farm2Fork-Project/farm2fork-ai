import csv
from pathlib import Path

from PIL import Image
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import Subset

from crop_grading.data.dataset import CropGradeDataset
from crop_grading.training.class_weights import compute_class_weights_from_subset


def test_compute_grade_class_weights_from_subset(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(4):
        Image.new("RGB", (8, 8)).save(image_dir / f"{index}.jpg")

    manifest = tmp_path / "labels.csv"
    rows = [
        ("images/0.jpg", "rice", "A"),
        ("images/1.jpg", "rice", "A"),
        ("images/2.jpg", "rice", "A"),
        ("images/3.jpg", "rice", "D"),
    ]
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "crop_label", "grade_label", "source", "split"])
        for image_path, crop_label, grade_label in rows:
            writer.writerow([image_path, crop_label, grade_label, "unit_test", "train"])

    dataset = CropGradeDataset(manifest, project_root=tmp_path, transform=None)
    subset = Subset(dataset, [0, 1, 2, 3])
    weights = compute_class_weights_from_subset(dataset, subset, label_type="grade", num_classes=4)

    assert weights[0] < weights[3]
    assert weights[1] == 0
    assert weights[2] == 0
