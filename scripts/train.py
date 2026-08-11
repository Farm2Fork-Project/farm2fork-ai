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
from crop_grading.data.sampling import build_balanced_sampler, sample_subset_indexes
from crop_grading.data.transforms import build_eval_transforms, build_train_transforms
from crop_grading.models.multitask_model import CropGradingModel
from crop_grading.training.class_weights import compute_class_weights_from_subset
from crop_grading.training.losses import MultiTaskLoss
from crop_grading.training.trainer import Trainer
from crop_grading.utils.config import load_config
from crop_grading.utils.device import pick_device
from crop_grading.utils.experiment_log import append_experiment_log, utc_timestamp


def main() -> int:
    parser = argparse.ArgumentParser(description="Train crop quality grading model.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(ROOT_DIR / args.config)
    train_manifest = config.data.train_manifest
    val_manifest = config.data.val_manifest
    backbone = config.model.backbone
    epochs = config.training.epochs
    batch_size = config.training.batch_size
    learning_rate = config.training.learning_rate
    pretrained = config.model.pretrained
    class_weights = config.training.class_weights
    selection_metric = config.training.selection_metric
    sampler = config.training.sampler
    balance_by = config.training.balance_by
    checkpoint_dir = config.training.checkpoint_dir
    experiment_log = config.training.experiment_log
    patience = config.training.patience
    min_delta = config.training.min_delta

    torch.manual_seed(config.project.seed)
    random.seed(config.project.seed)
    device = pick_device()

    train_dataset = CropGradeDataset(
        ROOT_DIR / train_manifest,
        project_root=ROOT_DIR,
        transform=build_train_transforms(config.data.image_size),
    )
    val_dataset = CropGradeDataset(
        ROOT_DIR / val_manifest,
        project_root=ROOT_DIR,
        transform=build_eval_transforms(config.data.image_size),
    )

    train_subset = Subset(
        train_dataset,
        sample_subset_indexes(
            train_dataset,
            max_samples=config.training.max_train_samples,
            seed=config.project.seed,
        ),
    )
    val_subset = Subset(
        val_dataset,
        sample_subset_indexes(
            val_dataset,
            max_samples=config.training.max_val_samples,
            seed=config.project.seed,
        ),
    )

    train_sampler = None
    train_shuffle = True
    if sampler == "balanced":
        train_sampler = build_balanced_sampler(
            train_dataset,
            train_subset,
            balance_by=balance_by,
            seed=config.project.seed,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    model = CropGradingModel(
        backbone_name=backbone,
        num_crops=config.model.num_crops,
        num_grades=config.model.num_grades,
        dropout_rate=config.model.dropout_rate,
        pretrained=pretrained,
    ).to(device)
    crop_class_weights = None
    grade_class_weights = None
    if class_weights in {"grade", "both"}:
        grade_class_weights = compute_class_weights_from_subset(
            train_dataset,
            train_subset,
            label_type="grade",
            num_classes=config.model.num_grades,
        ).to(device)
    if class_weights == "both":
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.training.weight_decay,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=ROOT_DIR / checkpoint_dir,
        show_progress=config.training.show_progress,
    )

    print(f"Device: {device}")
    print(f"Backbone: {backbone} | pretrained={pretrained}")
    print(f"Train rows: {len(train_subset)} | Val rows: {len(val_subset)}")
    print(f"Class weights: {class_weights}")
    print(f"Sampler: {sampler}" + (f" ({balance_by})" if sampler == "balanced" else ""))
    print(f"Early stopping: patience={patience}, min_delta={min_delta}")
    if grade_class_weights is not None:
        print(f"Grade weights: {[round(value, 3) for value in grade_class_weights.cpu().tolist()]}")

    best_score = None
    best_epoch = None
    best_val_metrics = None
    final_train_metrics = None
    final_val_metrics = None
    epochs_without_improvement = 0
    completed_epochs = 0
    for epoch in range(1, epochs + 1):
        completed_epochs = epoch
        train_metrics = trainer.train_epoch(epoch=epoch)
        val_metrics = trainer.validate(epoch=epoch)
        final_train_metrics = train_metrics
        final_val_metrics = val_metrics
        selection_score = _selection_score(val_metrics, selection_metric)
        is_best = best_score is None or _is_better_score(
            selection_score,
            best_score,
            selection_metric,
            min_delta=min_delta,
        )
        if is_best:
            best_score = selection_score
            best_epoch = epoch
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
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
            f"selection_{selection_metric}={selection_score:.3f} "
            f"checkpoint={checkpoint_path.relative_to(ROOT_DIR)}"
        )

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no {selection_metric} improvement for {patience} epochs."
            )
            break

    append_experiment_log(
        ROOT_DIR / experiment_log,
        {
            "timestamp_utc": utc_timestamp(),
            "backbone": backbone,
            "pretrained": pretrained,
            "epochs": epochs,
            "completed_epochs": completed_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_rows": len(train_subset),
            "val_rows": len(val_subset),
            "max_train_samples": config.training.max_train_samples,
            "max_val_samples": config.training.max_val_samples,
            "class_weights": class_weights,
            "sampler": sampler,
            "balance_by": balance_by if sampler == "balanced" else "",
            "selection_metric": selection_metric,
            "patience": patience,
            "min_delta": min_delta,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_val_loss": best_val_metrics.loss if best_val_metrics else None,
            "best_val_crop_acc": best_val_metrics.crop_accuracy if best_val_metrics else None,
            "best_val_grade_acc": best_val_metrics.grade_accuracy if best_val_metrics else None,
            "best_val_adj_grade_acc": (
                best_val_metrics.adjacent_grade_accuracy if best_val_metrics else None
            ),
            "final_train_loss": final_train_metrics.loss if final_train_metrics else None,
            "final_train_crop_acc": final_train_metrics.crop_accuracy if final_train_metrics else None,
            "final_train_grade_acc": final_train_metrics.grade_accuracy if final_train_metrics else None,
            "final_train_adj_grade_acc": (
                final_train_metrics.adjacent_grade_accuracy if final_train_metrics else None
            ),
            "final_val_loss": final_val_metrics.loss if final_val_metrics else None,
            "final_val_crop_acc": final_val_metrics.crop_accuracy if final_val_metrics else None,
            "final_val_grade_acc": final_val_metrics.grade_accuracy if final_val_metrics else None,
            "final_val_adj_grade_acc": (
                final_val_metrics.adjacent_grade_accuracy if final_val_metrics else None
            ),
            "checkpoint_dir": checkpoint_dir,
        },
    )

    print(f"Training run complete. Logged to {experiment_log}")
    return 0


def _selection_score(metrics, selection_metric: str) -> float:
    if selection_metric == "combined":
        return metrics.selection_score()
    if selection_metric == "grade":
        return metrics.grade_accuracy
    if selection_metric == "adjacent":
        return metrics.adjacent_grade_accuracy
    if selection_metric == "crop":
        return metrics.crop_accuracy
    if selection_metric == "loss":
        return metrics.loss
    raise ValueError(f"Unknown selection metric: {selection_metric}")


def _is_better_score(
    current: float,
    best: float,
    selection_metric: str,
    *,
    min_delta: float = 0.0,
) -> bool:
    if selection_metric == "loss":
        return (best - current) > min_delta
    return (current - best) > min_delta


if __name__ == "__main__":
    raise SystemExit(main())
