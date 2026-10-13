"""Cross-source crop-ID probe.

Tests whether the crop classifier recognises a crop from a DIFFERENT image source than it was
trained on. The model's crop head was trained on rice_quality / mango_grading / GrainSet. Here we
evaluate it on the *variety* datasets (Koklu rice; Pakistani mango classification) — independent
acquisitions of the same crops. If crop-ID holds, it reflects robust crop recognition; if it
collapses, it supports the source-shortcut hypothesis (paper §5.4).

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/crosssource_cropid.py
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

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install ML deps: pip install -e .[ml]") from exc

from crop_grading.constants import CROP_INDEX_TO_LABEL
from crop_grading.data.dataset import CropGradeDataset
from crop_grading.data.transforms import build_eval_transforms
from crop_grading.models.multitask_model import CropGradingModel
from crop_grading.utils.config import load_config
from crop_grading.utils.device import pick_device

EXTS = {".jpg", ".jpeg", ".png"}
# (true crop, source root, images-per-subfolder)
SOURCES = [
    ("rice", "data/raw/Rice/Rice_Image_Dataset", 60),
    ("mango", "data/raw/mango/Classification_dataset", 40),
]


def collect(crop: str, root: str, per_folder: int) -> list[dict]:
    base = ROOT_DIR / root
    rows = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        imgs = sorted(p for p in sub.iterdir() if p.suffix.lower() in EXTS)[:per_folder]
        for img in imgs:
            rows.append({
                "image_path": img.relative_to(ROOT_DIR).as_posix(),
                "crop_label": crop, "grade_label": "A",
                "source": "crosssource_variety", "split": "test",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-source crop-ID probe.")
    parser.add_argument("--config", default="configs/four_crops_dedup_15ep.yaml")
    args = parser.parse_args()
    config = load_config(ROOT_DIR / args.config)
    device = pick_device()

    rows = []
    for crop, root, k in SOURCES:
        rows += collect(crop, root, k)
    manifest = ROOT_DIR / "data/metadata/_crosssource.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "crop_label", "grade_label", "source", "split"])
        w.writeheader(); w.writerows(rows)

    model = CropGradingModel(
        backbone_name=config.model.backbone, num_crops=config.model.num_crops,
        num_grades=config.model.num_grades, dropout_rate=config.model.dropout_rate, pretrained=False,
    ).to(device)
    ckpt = torch.load(ROOT_DIR / config.inference.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt)); model.eval()

    dataset = CropGradeDataset(manifest, project_root=ROOT_DIR,
                               transform=build_eval_transforms(config.data.image_size))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=config.data.num_workers)

    preds_by_true: dict[str, Counter] = {}
    with torch.no_grad():
        for batch in loader:
            crop_logits, _ = model(batch["image"].to(device))
            pred = crop_logits.argmax(1).cpu()
            for i in range(len(pred)):
                true = batch["crop_name"][i]
                preds_by_true.setdefault(true, Counter())[CROP_INDEX_TO_LABEL[int(pred[i])]] += 1

    print(f"Checkpoint: {config.inference.checkpoint_path}")
    print("Crop-ID on INDEPENDENT-source images (variety datasets):\n")
    for true in sorted(preds_by_true):
        c = preds_by_true[true]; n = sum(c.values())
        correct = c.get(true, 0)
        dist = ", ".join(f"{k}={v}" for k, v in c.most_common())
        print(f"  true={true:6s} n={n:4d}  →  correct(as {true})={correct} ({100*correct/n:.1f}%)")
        print(f"           predicted distribution: {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
