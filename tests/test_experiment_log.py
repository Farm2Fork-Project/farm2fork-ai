import csv
from pathlib import Path

from crop_grading.utils.experiment_log import append_experiment_log


def test_append_experiment_log_creates_and_appends_rows(tmp_path: Path) -> None:
    path = tmp_path / "outputs" / "experiments" / "runs.csv"

    append_experiment_log(path, {"name": "run1", "score": 0.5})
    append_experiment_log(path, {"name": "run2", "score": 0.75})

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {"name": "run1", "score": "0.500000"},
        {"name": "run2", "score": "0.750000"},
    ]
