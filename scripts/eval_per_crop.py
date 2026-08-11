"""Per-crop grade metrics on the test set for a trained checkpoint.

Reports, per crop, the grade accuracy and macro-F1 (and n), plus overall. Useful because
pooled metrics hide per-crop behaviour and small crops (e.g. deduplicated rice, n=19) have
wide uncertainty.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
        python scripts/eval_per_crop.py --config configs/four_crops_dedup_15ep.yaml
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install ML deps: pip install -e .[ml]") from exc

from crop_grading.constants import CROP_INDEX_TO_LABEL, GRADE_INDEX_TO_LABEL
from crop_grading.data.dataset import CropGradeDataset
from crop_grading.data.transforms import build_eval_transforms
from crop_grading.models.multitask_model import CropGradingModel
from crop_grading.utils.config import load_config
from crop_grading.utils.device import pick_device


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="Per-crop grade metrics.")
    parser.add_argument("--config", default="configs/four_crops_dedup_15ep.yaml")
    args = parser.parse_args()

    config = load_config(ROOT_DIR / args.config)
    device = pick_device()

    model = CropGradingModel(
        backbone_name=config.model.backbone,
        num_crops=config.model.num_crops,
        num_grades=config.model.num_grades,
        dropout_rate=config.model.dropout_rate,
        pretrained=False,
    ).to(device)
    ckpt = torch.load(ROOT_DIR / config.inference.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    dataset = CropGradeDataset(
        ROOT_DIR / config.data.test_manifest,
        project_root=ROOT_DIR,
        transform=build_eval_transforms(config.data.image_size),
    )
    loader = DataLoader(dataset, batch_size=config.inference.batch_size, shuffle=False,
                        num_workers=config.data.num_workers)

    by_crop = defaultdict(lambda: {"gt": [], "pred": []})
    for batch in loader:
        images = batch["image"].to(device)
        crop_t = batch["crop_label"]
        grade_t = batch["grade_label"]
        _, grade_logits = model(images)
        grade_p = grade_logits.argmax(1).cpu()
        for i in range(len(crop_t)):
            crop = CROP_INDEX_TO_LABEL[int(crop_t[i])]
            by_crop[crop]["gt"].append(int(grade_t[i]))
            by_crop[crop]["pred"].append(int(grade_p[i]))

    print(f"Checkpoint: {config.inference.checkpoint_path}")
    print(f"{'crop':8s} {'n':>4s}  {'grade_acc':>9s}  {'grade_macroF1':>13s}  grades_present")
    all_gt, all_pred = [], []
    for crop in sorted(by_crop):
        gt = by_crop[crop]["gt"]; pred = by_crop[crop]["pred"]
        all_gt += gt; all_pred += pred
        acc = sum(g == p for g, p in zip(gt, pred)) / len(gt)
        mf1 = f1_score(gt, pred, average="macro", zero_division=0)
        present = "".join(sorted({GRADE_INDEX_TO_LABEL[g] for g in gt}))
        print(f"{crop:8s} {len(gt):>4d}  {acc:>9.4f}  {mf1:>13.4f}  {present}")
    acc = sum(g == p for g, p in zip(all_gt, all_pred)) / len(all_gt)
    mf1 = f1_score(all_gt, all_pred, average="macro", zero_division=0)
    print(f"{'OVERALL':8s} {len(all_gt):>4d}  {acc:>9.4f}  {mf1:>13.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
