"""Run the trained grader on arbitrary images (e.g. real phone photos).

Doubles as (a) the reality-check harness for the domain-shift audit and (b) the core
inference component for the app. Reports predicted crop + grade with softmax confidences
and a low-confidence flag, matching the exact eval preprocessing used in training.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
        python scripts/predict.py --dir data/field_test/wheat --crop wheat
    python scripts/predict.py --image /path/to/photo.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install ML deps: pip install -e .[ml]") from exc

from crop_grading.constants import CROP_CLASSES, GRADE_CLASSES
from crop_grading.data.transforms import build_eval_transforms
from crop_grading.models.multitask_model import CropGradingModel
from crop_grading.utils.config import load_config
from crop_grading.utils.device import pick_device

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}


def collect_images(args) -> list[Path]:
    if args.image:
        return [Path(args.image)]
    root = Path(args.dir)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade arbitrary images with the trained model.")
    parser.add_argument("--config", default="configs/four_crops_dedup_15ep.yaml")
    parser.add_argument("--image", default=None, help="Single image path.")
    parser.add_argument("--dir", default=None, help="Folder of images (recursive).")
    parser.add_argument("--crop", default=None, help="Optional true crop label for match reporting.")
    parser.add_argument("--threshold", type=float, default=None, help="Low-confidence flag threshold.")
    args = parser.parse_args()
    if not args.image and not args.dir:
        raise SystemExit("Provide --image or --dir.")

    config = load_config(ROOT_DIR / args.config)
    threshold = args.threshold if args.threshold is not None else config.inference.confidence_threshold
    device = pick_device()

    model = CropGradingModel(
        backbone_name=config.model.backbone, num_crops=config.model.num_crops,
        num_grades=config.model.num_grades, dropout_rate=config.model.dropout_rate, pretrained=False,
    ).to(device)
    ckpt = torch.load(ROOT_DIR / config.inference.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    tf = build_eval_transforms(config.data.image_size)

    images = collect_images(args)
    if not images:
        raise SystemExit("No images found.")
    print(f"Model: {config.inference.checkpoint_path} | device {device} | conf threshold {threshold}")
    print(f"Grading {len(images)} image(s)"
          + (f" (expected crop: {args.crop})" if args.crop else "") + "\n")
    print(f"{'file':32s} {'pred_crop':10s} {'p':>5s}  {'grade':5s} {'p':>5s}  {'A/B/C/D probs':22s} flag")

    flagged = 0
    crop_match = 0
    with torch.no_grad():
        for path in images:
            try:
                with Image.open(path) as im:
                    x = tf(im.convert("RGB")).unsqueeze(0).to(device)
            except Exception as exc:  # noqa: BLE001
                print(f"{path.name[:32]:32s} <could not open: {exc}>")
                continue
            crop_logits, grade_logits = model(x)
            cp = F.softmax(crop_logits, dim=1)[0]
            gp = F.softmax(grade_logits, dim=1)[0]
            ci = int(cp.argmax()); gi = int(gp.argmax())
            crop = CROP_CLASSES[ci]; grade = GRADE_CLASSES[gi]
            gprobs = " ".join(f"{v:.2f}" for v in gp.tolist())
            low = float(gp[gi]) < threshold
            flag = "LOW-CONF" if low else ""
            if low:
                flagged += 1
            if args.crop and crop == args.crop.lower():
                crop_match += 1
            print(f"{path.name[:32]:32s} {crop:10s} {cp[ci]:5.2f}  {grade:5s} {gp[gi]:5.2f}  "
                  f"{gprobs:22s} {flag}")

    n = len(images)
    print(f"\nSummary: {n} images | {flagged} low-confidence (<{threshold})"
          + (f" | crop predicted '{args.crop}' for {crop_match}/{n}" if args.crop else ""))
    print("Note: on real phone photos, watch for (a) wrong/over-confident grades and "
          "(b) whether crop prediction collapses — that is the domain-shift signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
