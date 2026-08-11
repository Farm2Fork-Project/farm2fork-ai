"""Evaluate a trained crop grading checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import torch
    from torch.utils.data import DataLoader, Subset
except ImportError as exc:
    raise SystemExit("PyTorch is required. Install ML dependencies with: pip install -e .[ml]") from exc

from crop_grading.constants import CROP_CLASSES, GRADE_CLASSES
from crop_grading.data.dataset import CropGradeDataset
from crop_grading.data.sampling import sample_subset_indexes
from crop_grading.data.transforms import build_eval_transforms
from crop_grading.models.multitask_model import CropGradingModel
from crop_grading.training.evaluation import evaluate_model, format_confusion_matrix
from crop_grading.utils.config import load_config
from crop_grading.utils.device import pick_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate crop grading checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(ROOT_DIR / args.config)
    manifest = config.data.test_manifest
    checkpoint_arg = config.inference.checkpoint_path
    backbone = config.model.backbone
    batch_size = config.inference.batch_size
    device = pick_device()

    dataset = CropGradeDataset(
        ROOT_DIR / manifest,
        project_root=ROOT_DIR,
        transform=build_eval_transforms(config.data.image_size),
    )
    subset = Subset(
        dataset,
        sample_subset_indexes(
            dataset,
            max_samples=config.inference.max_samples,
            seed=config.project.seed,
        ),
    )
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    model = CropGradingModel(
        backbone_name=backbone,
        num_crops=config.model.num_crops,
        num_grades=config.model.num_grades,
        dropout_rate=config.model.dropout_rate,
        pretrained=False,
    ).to(device)

    checkpoint_path = ROOT_DIR / checkpoint_arg
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    result = evaluate_model(
        model,
        loader,
        device=device,
        show_progress=config.inference.show_progress,
    )
    print(f"Checkpoint: {checkpoint_path.relative_to(ROOT_DIR)}")
    print(f"Device: {device}")
    print(f"Samples: {result.total_samples}")
    print(f"Crop accuracy: {result.crop_accuracy:.4f}")
    print(f"Grade accuracy: {result.grade_accuracy:.4f}")
    print(f"Adjacent grade accuracy: {result.adjacent_grade_accuracy:.4f}")
    print("\nCrop confusion matrix")
    print(format_confusion_matrix(result.crop_confusion, CROP_CLASSES))
    print("\nGrade confusion matrix")
    print(format_confusion_matrix(result.grade_confusion, GRADE_CLASSES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
