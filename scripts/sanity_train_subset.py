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
    from tqdm.auto import tqdm
except ImportError as exc:
    raise SystemExit("PyTorch is required. Install ML dependencies with: pip install -e .[ml]") from exc

from crop_grading.data.dataset import CropGradeDataset
from crop_grading.data.sampling import sample_subset_indexes
from crop_grading.data.transforms import build_train_transforms
from crop_grading.models.multitask_model import CropGradingModel
from crop_grading.training.losses import MultiTaskLoss
from crop_grading.training.metrics import accuracy_from_logits
from crop_grading.utils.config import load_config
from crop_grading.utils.device import pick_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Train for a few batches to verify the pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    config = load_config(ROOT_DIR / args.config)
    manifest = args.manifest or config.data.train_manifest
    backbone = args.backbone or config.model.backbone
    batch_size = args.batch_size or config.training.batch_size
    learning_rate = args.lr or config.training.learning_rate
    torch.manual_seed(config.project.seed)
    device = pick_device()

    dataset = CropGradeDataset(
        ROOT_DIR / manifest,
        project_root=ROOT_DIR,
        transform=build_train_transforms(config.data.image_size),
    )
    subset_indexes = sample_subset_indexes(dataset, max_samples=args.max_samples, seed=config.project.seed)
    subset = Subset(dataset, subset_indexes)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )

    model = CropGradingModel(
        backbone_name=backbone,
        num_crops=config.model.num_crops,
        num_grades=config.model.num_grades,
        dropout_rate=config.model.dropout_rate,
        pretrained=False,
    ).to(device)
    loss_fn = MultiTaskLoss(
        crop_weight=config.training.crop_loss_weight,
        grade_weight=config.training.grade_loss_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.training.weight_decay,
    )

    model.train()
    print(f"Device: {device}")
    print(f"Dataset rows: {len(dataset)} | subset rows: {len(subset)}")

    last_loss = None
    progress = tqdm(loader, desc="sanity train", total=min(args.steps, len(loader)), leave=False)
    for step, batch in enumerate(progress, start=1):
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
        progress.set_postfix(
            step=step,
            loss=f"{last_loss:.4f}",
            crop_acc=f"{crop_acc:.3f}",
            grade_acc=f"{grade_acc:.3f}",
        )

    if last_loss is None:
        raise RuntimeError("No batches were loaded. Check manifest and batch size.")

    print("Tiny training sanity check passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
