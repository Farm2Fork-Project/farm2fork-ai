"""Build the photo-similarity reference that lets the service refer unfamiliar photos.

The threshold is the chosen percentile of validation-image similarities, so it is
fixed without looking at the held-out test split or the supplied field photos.

    ./.venv/bin/python scripts/build_photo_reference.py \
        --checkpoint models/exports/mvp-2026-10-14/best_model.pth \
        --output reference/photo_reference.npz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from crop_grading.data.transforms import build_eval_transforms
from crop_grading.models.multitask_model import CropGradingModel
from crop_grading.service.photo_reference import save_reference, weights_fingerprint
from crop_grading.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def manifest_images(path: Path) -> list[Path]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [ROOT / row["image_path"] for row in csv.DictReader(handle)]


def embed(model: CropGradingModel, transform, images: list[Path]) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for image in images:
            tensor = transform(Image.open(image).convert("RGB")).unsqueeze(0)
            rows.append(F.normalize(model.backbone(tensor), dim=1)[0].numpy())
    return np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/grade_only_four_crops_dedup_15ep.yaml")
    parser.add_argument("--train", type=Path, default=ROOT / "data/metadata/dedup_train_labels.csv")
    parser.add_argument("--val", type=Path, default=ROOT / "data/metadata/dedup_val_labels.csv")
    parser.add_argument("--percentile", type=float, default=5.0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    if "crop_embed.weight" in state:
        raise ValueError("Photo references are built for the plain grade-only model.")
    model = CropGradingModel(
        backbone_name=config.model.backbone,
        num_crops=config.model.num_crops,
        num_grades=config.model.num_grades,
        dropout_rate=config.model.dropout_rate,
        pretrained=False,
    )
    model.load_state_dict(state)
    model.eval()
    torch.set_num_threads(4)
    transform = build_eval_transforms(config.data.image_size)

    train = embed(model, transform, manifest_images(args.train))
    val = embed(model, transform, manifest_images(args.val))
    # Use the float16 values that will be stored, so the threshold matches serving.
    bank = train.astype(np.float16).astype(np.float32)
    val_sims = np.sort(val @ bank.T, axis=1)[:, -args.k :].mean(axis=1)
    threshold = float(np.percentile(val_sims, args.percentile))

    save_reference(
        args.output,
        embeddings=train,
        k=args.k,
        min_similarity=threshold,
        fingerprint=weights_fingerprint(state),
        provenance={
            "built_at_pkt": datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "train_manifest": args.train.name,
            "train_images": int(train.shape[0]),
            "val_manifest": args.val.name,
            "val_images": int(val.shape[0]),
            "threshold_rule": f"validation p{args.percentile:g} of mean top-{args.k} cosine similarity",
        },
    )
    print(f"saved {args.output} train={train.shape[0]} threshold={threshold:.4f}")


if __name__ == "__main__":
    main()
