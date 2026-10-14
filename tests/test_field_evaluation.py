from pathlib import Path

import pytest
from PIL import Image

from scripts.evaluate_field import discover, filename_grade, metrics


def test_extensionless_image_is_included_and_corrupt_file_reported(tmp_path):
    crop = tmp_path / "maize"
    crop.mkdir()
    Image.new("RGB", (20, 20), "yellow").save(crop / "A1", format="JPEG")
    (crop / "B.jpeg").write_bytes(b"invalid image")
    rows, excluded = discover(tmp_path)
    assert len(rows) == 1
    assert rows[0]["image_path"] == "maize/A1"
    assert rows[0]["grade_label"] == "A"
    assert len(excluded) == 1
    assert excluded[0]["image_path"] == "maize/B.jpeg"


def test_grade_parser_rejects_unlabeled_or_ambiguous_names():
    assert filename_grade(Path("C2.webp")) == "C"
    with pytest.raises(ValueError):
        filename_grade(Path("Apple.jpeg"))


def test_metrics_keep_absent_grades_and_high_confidence_errors_visible():
    rows = [
        {"true_grade": "A", "predicted_grade": "A", "confidence": 0.9, "correct": True},
        {"true_grade": "B", "predicted_grade": "D", "confidence": 0.95, "correct": False},
        {"true_grade": "C", "predicted_grade": "B", "confidence": 0.6, "correct": False},
    ]
    result = metrics(rows, 0.7)
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["adjacent_accuracy"] == pytest.approx(2 / 3)
    assert result["true_grade_support"]["D"] == 0
    assert result["severe_errors_distance_at_least_two"] == 1
    assert result["accepted_wrong_n"] == 1
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["accepted_accuracy"] == 0.5
