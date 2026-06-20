"""Reusable training loop for the crop grading model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crop_grading.training.losses import MultiTaskLoss
from crop_grading.training.metrics import RunningAverage, accuracy_from_logits


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics collected for one training or validation epoch."""

    loss: float
    crop_loss: float
    grade_loss: float
    crop_accuracy: float
    grade_accuracy: float


class Trainer:
    """Owns epoch-level train/validate/checkpoint behavior."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: MultiTaskLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str | Path,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_grade_accuracy = -1.0

    def train_epoch(self) -> EpochMetrics:
        self.model.train()
        return self._run_epoch(train=True)

    @torch.no_grad()
    def validate(self) -> EpochMetrics:
        self.model.eval()
        return self._run_epoch(train=False)

    def save_checkpoint(
        self,
        *,
        epoch: int,
        metrics: EpochMetrics,
        is_best: bool,
    ) -> Path:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics.__dict__,
        }
        latest_path = self.checkpoint_dir / "latest_model.pth"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            return best_path
        return latest_path

    def _run_epoch(self, *, train: bool) -> EpochMetrics:
        loss_avg = RunningAverage()
        crop_loss_avg = RunningAverage()
        grade_loss_avg = RunningAverage()
        crop_acc_avg = RunningAverage()
        grade_acc_avg = RunningAverage()

        loader = self.train_loader if train else self.val_loader
        for batch in loader:
            images = batch["image"].to(self.device)
            crop_targets = batch["crop_label"].to(self.device)
            grade_targets = batch["grade_label"].to(self.device)
            batch_size = images.size(0)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            crop_logits, grade_logits = self.model(images)
            losses = self.loss_fn(crop_logits, grade_logits, crop_targets, grade_targets)

            if train:
                losses["loss"].backward()
                self.optimizer.step()

            loss_avg.update(losses["loss"].item(), batch_size)
            crop_loss_avg.update(losses["crop_loss"].item(), batch_size)
            grade_loss_avg.update(losses["grade_loss"].item(), batch_size)
            crop_acc_avg.update(accuracy_from_logits(crop_logits.detach(), crop_targets), batch_size)
            grade_acc_avg.update(accuracy_from_logits(grade_logits.detach(), grade_targets), batch_size)

        return EpochMetrics(
            loss=loss_avg.value,
            crop_loss=crop_loss_avg.value,
            grade_loss=grade_loss_avg.value,
            crop_accuracy=crop_acc_avg.value,
            grade_accuracy=grade_acc_avg.value,
        )
