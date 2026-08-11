"""Frozen-features baseline.

Extracts features from a *frozen* ImageNet-pretrained backbone (no fine-tuning) and
fits a logistic-regression classifier for grade (and crop). This quantifies how much
the fine-tuned multi-task model gains over off-the-shelf features.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
        python scripts/baseline_frozen_features.py --config configs/rice_mango_15ep.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import numpy as np
    import timm
    import torch
    from torch.utils.data import DataLoader
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install ML deps: pip install -e .[ml]") from exc

from crop_grading.data.dataset import CropGradeDataset
from crop_grading.data.transforms import build_eval_transforms
from crop_grading.utils.config import load_config
from crop_grading.utils.device import pick_device


@torch.no_grad()
def extract(backbone, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats, crops, grades = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        out = backbone(images).detach().cpu().numpy()
        feats.append(out)
        crops.append(batch["crop_label"].numpy())
        grades.append(batch["grade_label"].numpy())
    return np.concatenate(feats), np.concatenate(crops), np.concatenate(grades)


def fit_and_report(name: str, x_tr, y_tr, x_te, y_te) -> None:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    acc = accuracy_score(y_te, pred)
    macro = f1_score(y_te, pred, average="macro", zero_division=0)
    print(f"{name}: accuracy={acc:.4f}  macro-F1={macro:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen-features + logistic-regression baseline.")
    parser.add_argument("--config", default="configs/rice_mango_15ep.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = load_config(ROOT_DIR / args.config)
    device = pick_device()
    print(f"Device: {device} | backbone: {config.model.backbone} (frozen, ImageNet)")

    backbone = timm.create_model(
        config.model.backbone, pretrained=True, num_classes=0, global_pool="avg"
    ).to(device).eval()

    transform = build_eval_transforms(config.data.image_size)
    loaders = {}
    for split, manifest in (("train", config.data.train_manifest), ("test", config.data.test_manifest)):
        dataset = CropGradeDataset(ROOT_DIR / manifest, project_root=ROOT_DIR, transform=transform)
        loaders[split] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=config.data.num_workers
        )
        print(f"  {split}: {len(dataset)} images")

    print("Extracting frozen features...")
    x_tr, crop_tr, grade_tr = extract(backbone, loaders["train"], device)
    x_te, crop_te, grade_te = extract(backbone, loaders["test"], device)
    print(f"Feature dim: {x_tr.shape[1]}")

    fit_and_report("CROP  (LogReg on frozen feats)", x_tr, crop_tr, x_te, crop_te)
    fit_and_report("GRADE (LogReg on frozen feats)", x_tr, grade_tr, x_te, grade_te)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
