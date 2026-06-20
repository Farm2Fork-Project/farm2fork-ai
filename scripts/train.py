"""Train the crop grading model."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
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
from crop_grading.data.transforms import build_eval_transforms, build_train_transforms
from crop_grading.models.multitask_model import CropGradingModel
from crop_grading.training.class_weights import compute_class_weights_from_subset
from crop_grading.training.losses import MultiTaskLoss
from crop_grading.training.trainer import Trainer
from crop_grading.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train crop quality grading model.")
    parser.add_argument("--train-manifest", default="data/metadata/train_labels.csv")
    parser.add_argument("--val-manifest", default="data/metadata/val_labels.csv")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained backbone weights.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    parser.add_argument(
        "--class-weights",
        choices=("none", "grade", "both"),
        default="grade",
        help="Apply inverse-frequency class weights computed from the training subset.",
    )
    args = parser.parse_args()

    config = load_config(ROOT_DIR / "configs/default.yaml")
    torch.manual_seed(config.project.seed)
    random.seed(config.project.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CropGradeDataset(
        ROOT_DIR / args.train_manifest,
        project_root=ROOT_DIR,
        transform=build_train_transforms(config.data.image_size),
    )
    val_dataset = CropGradeDataset(
        ROOT_DIR / args.val_manifest,
        project_root=ROOT_DIR,
        transform=build_eval_transforms(config.data.image_size),
    )

    train_subset = Subset(
        train_dataset,
        sample_subset_indexes(
            train_dataset,
            max_samples=args.max_train_samples,
            seed=config.project.seed,
        ),
    )
    val_subset = Subset(
        val_dataset,
        sample_subset_indexes(
            val_dataset,
            max_samples=args.max_val_samples,
            seed=config.project.seed,
        ),
    )

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CropGradingModel(
        backbone_name=args.backbone,
        num_crops=config.model.num_crops,
        num_grades=config.model.num_grades,
        dropout_rate=config.model.dropout_rate,
        pretrained=args.pretrained,
    ).to(device)
    crop_class_weights = None
    grade_class_weights = None
    if args.class_weights in {"grade", "both"}:
        grade_class_weights = compute_class_weights_from_subset(
            train_dataset,
            train_subset,
            label_type="grade",
            num_classes=config.model.num_grades,
        ).to(device)
    if args.class_weights == "both":
        crop_class_weights = compute_class_weights_from_subset(
            train_dataset,
            train_subset,
            label_type="crop",
            num_classes=config.model.num_crops,
        ).to(device)

    loss_fn = MultiTaskLoss(
        crop_weight=config.training.crop_loss_weight,
        grade_weight=config.training.grade_loss_weight,
        crop_class_weights=crop_class_weights,
        grade_class_weights=grade_class_weights,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.training.weight_decay)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=ROOT_DIR / args.checkpoint_dir,
        show_progress=not args.no_progress,
    )

    print(f"Device: {device}")
    print(f"Backbone: {args.backbone} | pretrained={args.pretrained}")
    print(f"Train rows: {len(train_subset)} | Val rows: {len(val_subset)}")
    print(f"Class weights: {args.class_weights}")
    if grade_class_weights is not None:
        print(f"Grade weights: {[round(value, 3) for value in grade_class_weights.cpu().tolist()]}")

    best_grade_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(epoch=epoch)
        val_metrics = trainer.validate(epoch=epoch)
        is_best = val_metrics.grade_accuracy > best_grade_accuracy
        if is_best:
            best_grade_accuracy = val_metrics.grade_accuracy
        checkpoint_path = trainer.save_checkpoint(epoch=epoch, metrics=val_metrics, is_best=is_best)

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_crop_acc={train_metrics.crop_accuracy:.3f} "
            f"train_grade_acc={train_metrics.grade_accuracy:.3f} "
            f"train_adj_grade_acc={train_metrics.adjacent_grade_accuracy:.3f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_crop_acc={val_metrics.crop_accuracy:.3f} "
            f"val_grade_acc={val_metrics.grade_accuracy:.3f} "
            f"val_adj_grade_acc={val_metrics.adjacent_grade_accuracy:.3f} "
            f"checkpoint={checkpoint_path.relative_to(ROOT_DIR)}"
        )

    print("Training run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
