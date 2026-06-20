"""Run a tiny training sanity check on a subset of the dataset."""

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

from crop_grading.data.dataset import CropGradeDataset
from crop_grading.data.sampling import sample_subset_indexes
from crop_grading.data.transforms import build_train_transforms
from crop_grading.models.multitask_model import CropGradingModel
from crop_grading.training.losses import MultiTaskLoss
from crop_grading.training.metrics import accuracy_from_logits
from crop_grading.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train for a few batches to verify the pipeline.")
    parser.add_argument("--manifest", default="data/metadata/train_labels.csv")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    config = load_config(ROOT_DIR / "configs/default.yaml")
    torch.manual_seed(config.project.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CropGradeDataset(
        ROOT_DIR / args.manifest,
        project_root=ROOT_DIR,
        transform=build_train_transforms(config.data.image_size),
    )
    subset_indexes = sample_subset_indexes(dataset, max_samples=args.max_samples, seed=config.project.seed)
    subset = Subset(dataset, subset_indexes)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = CropGradingModel(
        backbone_name=args.backbone,
        num_crops=config.model.num_crops,
        num_grades=config.model.num_grades,
        dropout_rate=config.model.dropout_rate,
        pretrained=False,
    ).to(device)
    loss_fn = MultiTaskLoss(
        crop_weight=config.training.crop_loss_weight,
        grade_weight=config.training.grade_loss_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.training.weight_decay)

    model.train()
    print(f"Device: {device}")
    print(f"Dataset rows: {len(dataset)} | subset rows: {len(subset)}")

    last_loss = None
    for step, batch in enumerate(loader, start=1):
        if step > args.steps:
            break

        images = batch["image"].to(device)
        crop_targets = batch["crop_label"].to(device)
        grade_targets = batch["grade_label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        crop_logits, grade_logits = model(images)
        losses = loss_fn(crop_logits, grade_logits, crop_targets, grade_targets)
        losses["loss"].backward()
        optimizer.step()

        crop_acc = accuracy_from_logits(crop_logits.detach(), crop_targets)
        grade_acc = accuracy_from_logits(grade_logits.detach(), grade_targets)
        last_loss = losses["loss"].item()
        print(
            f"step={step} "
            f"loss={last_loss:.4f} "
            f"crop_loss={losses['crop_loss'].item():.4f} "
            f"grade_loss={losses['grade_loss'].item():.4f} "
            f"crop_acc={crop_acc:.3f} "
            f"grade_acc={grade_acc:.3f}"
        )

    if last_loss is None:
        raise RuntimeError("No batches were loaded. Check manifest and batch size.")

    print("Tiny training sanity check passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
