import csv
from pathlib import Path

from PIL import Image
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import Subset, WeightedRandomSampler

from crop_grading.data.dataset import CropGradeDataset
from crop_grading.data.sampling import build_balanced_sampler, sample_subset_indexes


def make_dataset(tmp_path: Path) -> CropGradeDataset:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = [
        ("0.jpg", "rice", "A"),
        ("1.jpg", "rice", "A"),
        ("2.jpg", "rice", "A"),
        ("3.jpg", "maize", "D"),
    ]
    for image_name, _, _ in rows:
        Image.new("RGB", (8, 8)).save(image_dir / image_name)

    manifest = tmp_path / "labels.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "crop_label", "grade_label", "source", "split"])
        for image_name, crop_label, grade_label in rows:
            writer.writerow([f"images/{image_name}", crop_label, grade_label, "unit_test", "train"])
    return CropGradeDataset(manifest, project_root=tmp_path, transform=None)


def test_sample_subset_indexes_limits_rows(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)

    indexes = sample_subset_indexes(dataset, max_samples=2, seed=42)

    assert len(indexes) == 2
    assert all(0 <= index < len(dataset) for index in indexes)


def test_build_balanced_sampler_weights_inverse_group_frequency(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    subset = Subset(dataset, [0, 1, 2, 3])

    sampler = build_balanced_sampler(dataset, subset, balance_by="crop_grade", seed=42)

    assert isinstance(sampler, WeightedRandomSampler)
    weights = sampler.weights.tolist()
    assert weights[0] == pytest.approx(1 / 3)
    assert weights[1] == pytest.approx(1 / 3)
    assert weights[2] == pytest.approx(1 / 3)
    assert weights[3] == pytest.approx(1.0)
