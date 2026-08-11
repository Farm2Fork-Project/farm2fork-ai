"""Measure single-image inference latency and model size.

Reports batch=1 latency (the on-device / phone scenario) on CPU and, if available, MPS,
plus parameter count and checkpoint size. Latency includes host->device transfer + forward.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
        python scripts/measure_latency.py --config configs/four_crops_15ep.yaml --n 128
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install ML deps: pip install -e .[ml]") from exc

from crop_grading.data.dataset import CropGradeDataset
from crop_grading.data.transforms import build_eval_transforms
from crop_grading.models.multitask_model import CropGradingModel
from crop_grading.utils.config import load_config


def load_model(config) -> CropGradingModel:
    model = CropGradingModel(
        backbone_name=config.model.backbone,
        num_crops=config.model.num_crops,
        num_grades=config.model.num_grades,
        dropout_rate=config.model.dropout_rate,
        pretrained=False,
    )
    ckpt_path = ROOT_DIR / config.inference.checkpoint_path
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model, ckpt_path


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def bench(model: CropGradingModel, images, device: torch.device, warmup: int = 10) -> dict:
    model.to(device)
    # Warmup
    for i in range(min(warmup, len(images))):
        _ = model(images[i].unsqueeze(0).to(device))
    sync(device)
    times_ms = []
    for img in images:
        x = img.unsqueeze(0)
        start = time.perf_counter()
        _ = model(x.to(device))
        sync(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    times_ms.sort()
    p95 = times_ms[int(len(times_ms) * 0.95) - 1]
    mean = statistics.mean(times_ms)
    return {
        "mean": mean,
        "std": statistics.pstdev(times_ms),
        "median": statistics.median(times_ms),
        "p95": p95,
        "throughput": 1000.0 / mean,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure single-image inference latency.")
    parser.add_argument("--config", default="configs/four_crops_15ep.yaml")
    parser.add_argument("--n", type=int, default=128)
    args = parser.parse_args()

    config = load_config(ROOT_DIR / args.config)
    torch.manual_seed(config.project.seed)

    dataset = CropGradeDataset(
        ROOT_DIR / config.data.test_manifest,
        project_root=ROOT_DIR,
        transform=build_eval_transforms(config.data.image_size),
    )
    n = min(args.n, len(dataset))
    images = [dataset[i]["image"] for i in range(n)]

    model, ckpt_path = load_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"Model: {config.model.backbone} multi-task | params: {n_params/1e6:.2f}M | "
          f"checkpoint: {size_mb:.1f} MB")
    print(f"Timing single-image (batch=1) over {n} images @ {config.data.image_size}px\n")

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    for dev in devices:
        r = bench(model, images, torch.device(dev))
        print(f"{dev.upper():4s}  mean {r['mean']:6.1f} ms  median {r['median']:6.1f}  "
              f"p95 {r['p95']:6.1f}  std {r['std']:5.1f}  |  {r['throughput']:6.1f} img/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
