import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from crop_grading.training.losses import MultiTaskLoss
from crop_grading.training.trainer import Trainer


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.crop_head = torch.nn.Linear(12, 6)
        self.grade_head = torch.nn.Linear(12, 4)

    def forward(self, images):
        features = self.flatten(images)
        return self.crop_head(features), self.grade_head(features)


class DictDataset(TensorDataset):
    def __getitem__(self, index):
        image, crop_label, grade_label = super().__getitem__(index)
        return {
            "image": image,
            "crop_label": crop_label,
            "grade_label": grade_label,
        }


def test_trainer_runs_one_epoch_and_saves_checkpoint(tmp_path):
    images = torch.randn(4, 3, 2, 2)
    crop_labels = torch.tensor([0, 1, 0, 1])
    grade_labels = torch.tensor([0, 1, 2, 3])
    loader = DataLoader(DictDataset(images, crop_labels, grade_labels), batch_size=2)

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        loss_fn=MultiTaskLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        show_progress=False,
    )

    train_metrics = trainer.train_epoch()
    val_metrics = trainer.validate()
    checkpoint = trainer.save_checkpoint(epoch=1, metrics=val_metrics, is_best=True)

    assert train_metrics.loss > 0
    assert train_metrics.adjacent_grade_accuracy >= 0
    assert train_metrics.selection_score() >= 0
    assert val_metrics.loss > 0
    assert checkpoint.exists()
    assert (tmp_path / "latest_model.pth").exists()
    assert (tmp_path / "best_model.pth").exists()
