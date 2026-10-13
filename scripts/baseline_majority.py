"""Majority-class baseline.

Predicts the single most frequent crop and grade (learned from the train manifest)
for every test row. This is the sanity floor: under class imbalance a constant
prediction can score high accuracy, which is exactly why we also report macro-F1.

Usage:
    python scripts/baseline_majority.py --config configs/rice_mango_15ep.yaml
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from crop_grading.utils.config import load_config

try:
    from sklearn.metrics import f1_score
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn required. Install with: pip install -e .[ml]") from exc


def read_labels(path: Path) -> tuple[list[str], list[str]]:
    crops, grades = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            crops.append((row.get("crop_label") or "").strip().lower())
            grades.append((row.get("grade_label") or "").strip().upper())
    return crops, grades


def report(name: str, y_true: list[str], y_pred: list[str]) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    print(f"{name}: accuracy={acc:.4f}  macro-F1={macro:.4f}  (predicting constant '{y_pred[0]}')")


def main() -> int:
    parser = argparse.ArgumentParser(description="Majority-class baseline.")
    parser.add_argument("--config", default="configs/rice_mango_15ep.yaml")
    args = parser.parse_args()

    config = load_config(ROOT_DIR / args.config)
    train_crops, train_grades = read_labels(ROOT_DIR / config.data.train_manifest)
    test_crops, test_grades = read_labels(ROOT_DIR / config.data.test_manifest)

    majority_crop = Counter(train_crops).most_common(1)[0][0]
    majority_grade = Counter(train_grades).most_common(1)[0][0]

    print(f"Train majority crop='{majority_crop}', grade='{majority_grade}'")
    print(f"Test rows: {len(test_crops)}")
    report("CROP ", test_crops, [majority_crop] * len(test_crops))
    report("GRADE", test_grades, [majority_grade] * len(test_grades))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
