"""Train/test leakage check via perceptual hashing (pHash).

For every test image, find its nearest train image in pHash space. Near-duplicates that
straddle the split inflate test metrics. Reports leakage counts at several Hamming
thresholds and per crop, and writes a "clean" test manifest (test rows whose nearest train
image is farther than --threshold) for a leakage-corrected re-evaluation.

Usage:
    python scripts/check_leakage.py --config configs/four_crops_15ep.yaml --threshold 5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import numpy as np
    from PIL import Image
    from scipy.fft import dct
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install ML deps: pip install -e .[ml]") from exc

from crop_grading.utils.config import load_config


def phash64(path: Path) -> np.uint64:
    """64-bit DCT perceptual hash (imagehash-compatible)."""
    img = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
    px = np.asarray(img, dtype=np.float64)
    coeff = dct(dct(px, axis=0), axis=1)
    low = coeff[:8, :8]
    bits = (low > np.median(low)).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return np.uint64(value)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def hash_rows(rows: list[dict], label: str) -> np.ndarray:
    hashes = np.empty(len(rows), dtype=np.uint64)
    for i, row in enumerate(rows):
        hashes[i] = phash64(ROOT_DIR / row["image_path"])
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i+1}/{len(rows)} {label}")
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description="pHash train/test leakage check.")
    parser.add_argument("--config", default="configs/four_crops_15ep.yaml")
    parser.add_argument("--threshold", type=int, default=5, help="Hamming distance for near-dup.")
    parser.add_argument("--clean-out", default="data/metadata/test_clean.csv")
    args = parser.parse_args()

    config = load_config(ROOT_DIR / args.config)
    train_rows = read_rows(ROOT_DIR / config.data.train_manifest)
    test_rows = read_rows(ROOT_DIR / config.data.test_manifest)
    print(f"Hashing {len(train_rows)} train + {len(test_rows)} test images...")
    train_h = hash_rows(train_rows, "train")
    test_h = hash_rows(test_rows, "test")

    # Nearest train neighbour (Hamming) for each test image — vectorised.
    min_dist = np.empty(len(test_rows), dtype=np.int64)
    argmin = np.empty(len(test_rows), dtype=np.int64)
    for i, h in enumerate(test_h):
        d = np.bitwise_count(train_h ^ h)  # popcount per train hash
        argmin[i] = int(d.argmin())
        min_dist[i] = int(d[argmin[i]])

    n = len(test_rows)
    print(f"\n=== Train↔Test leakage (n_test={n}) ===")
    for t in (0, 1, 3, args.threshold, 8, 10):
        c = int((min_dist <= t).sum())
        print(f"  test imgs with a train match at Hamming ≤ {t:>2}: {c:4d}  ({100*c/n:5.1f}%)")

    # Per-crop breakdown at the chosen threshold
    print(f"\n=== Per-crop leakage at Hamming ≤ {args.threshold} ===")
    crops = sorted({r["crop_label"] for r in test_rows})
    for crop in crops:
        idx = [i for i, r in enumerate(test_rows) if r["crop_label"] == crop]
        leaked = sum(1 for i in idx if min_dist[i] <= args.threshold)
        print(f"  {crop:8s}: {leaked:4d}/{len(idx):4d} ({100*leaked/max(1,len(idx)):5.1f}%)")

    # A few example leaking pairs (exact / near)
    print("\n=== Example leaking pairs (test ⟵ nearest train) ===")
    order = np.argsort(min_dist)
    for i in order[:5]:
        tr = train_rows[argmin[i]]
        te = test_rows[i]
        print(f"  d={min_dist[i]:>2}  test[{te['crop_label']}/{te['grade_label']}] {Path(te['image_path']).name}"
              f"  ⟵  train[{tr['crop_label']}/{tr['grade_label']}] {Path(tr['image_path']).name}")

    # Write clean test manifest (nearest train farther than threshold)
    clean = [test_rows[i] for i in range(n) if min_dist[i] > args.threshold]
    out = ROOT_DIR / args.clean_out
    with out.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "crop_label", "grade_label", "source", "split"])
        writer.writeheader()
        writer.writerows(clean)
    print(f"\nClean test set: {len(clean)}/{n} rows (removed {n-len(clean)} with train near-dup ≤ {args.threshold})")
    print(f"Written to {args.clean_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
