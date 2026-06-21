import csv
from pathlib import Path

from PIL import Image
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from crop_grading.constants import CROP_LABEL_TO_INDEX, GRADE_LABEL_TO_INDEX
from crop_grading.data.dataset import CropGradeDataset
from crop_grading.data.transforms import build_eval_transforms


def test_crop_grade_dataset_loads_image_and_labels(tmp_path: Path) -> None:
    image_path = tmp_path / "data" / "raw" / "mango" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), color=(200, 150, 40)).save(image_path)

    manifest_path = tmp_path / "labels.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "crop_label", "grade_label", "source", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "data/raw/mango/sample.jpg",
                "crop_label": "mango",
                "grade_label": "B",
                "source": "unit_test",
                "split": "train",
            }
        )

    dataset = CropGradeDataset(
        manifest_path,
        project_root=tmp_path,
        transform=build_eval_transforms(image_size=32),
    )
    item = dataset[0]

    assert len(dataset) == 1
    assert tuple(item["image"].shape) == (3, 32, 32)
    assert item["crop_label"] == CROP_LABEL_TO_INDEX["mango"]
    assert item["grade_label"] == GRADE_LABEL_TO_INDEX["B"]
    assert item["crop_name"] == "mango"
    assert item["grade_name"] == "B"
