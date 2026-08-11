"""Deduplicate near-duplicate images (pHash) BEFORE splitting, then write leakage-free
train/val/test manifests.

Clusters near-duplicates within each crop (union-find over Hamming <= threshold), keeps one
representative per cluster, then stratified-splits 70/15/15 by (crop, grade). Because every
near-duplicate collapses to a single image before the split, the resulting split cannot leak
at the chosen threshold. Prints a threshold sweep first so you can see how much each crop shrinks.

Usage:
    python scripts/dedup_and_split.py --threshold 5 --out-prefix dedup
"""

from __future__ import annotations

import argparse
import csv
import random
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

from crop_grading.data.manifest import validate_manifest

MANIFESTS = ("train_labels.csv", "val_labels.csv", "test_labels.csv")
FIELDS = ["image_path", "crop_label", "grade_label", "source", "split"]


def phash64(path: Path) -> np.uint64:
    img = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
    coeff = dct(dct(np.asarray(img, dtype=np.float64), axis=0), axis=1)
    low = coeff[:8, :8]
    bits = (low > np.median(low)).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return np.uint64(value)


def load_all_rows(metadata_dir: Path) -> list[dict]:
    rows = []
    for name in MANIFESTS:
        path = metadata_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows.extend(list(csv.DictReader(file)))
    return rows


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster_within_crop(hashes: np.ndarray, crops: list[str], threshold: int) -> DSU:
    dsu = DSU(len(hashes))
    for crop in sorted(set(crops)):
        idx = np.array([i for i, c in enumerate(crops) if c == crop])
        sub = hashes[idx]
        for a in range(len(idx)):
            d = np.bitwise_count(sub[a + 1:] ^ sub[a])
            for off in np.nonzero(d <= threshold)[0]:
                dsu.union(int(idx[a]), int(idx[a + 1 + off]))
    return dsu


def unique_count_per_crop(dsu: DSU, crops: list[str]) -> dict[str, int]:
    roots_by_crop: dict[str, set] = {}
    for i, c in enumerate(crops):
        roots_by_crop.setdefault(c, set()).add(dsu.find(i))
    return {c: len(r) for c, r in roots_by_crop.items()}


def stratified_split(rows: list[dict], seed: int, train_r: float, val_r: float) -> None:
    rng = random.Random(seed)
    keys = sorted({(r["crop_label"], r["grade_label"]) for r in rows})
    for row in rows:
        row["split"] = ""
    for crop, grade in keys:
        group = [r for r in rows if r["crop_label"] == crop and r["grade_label"] == grade]
        rng.shuffle(group)
        n = len(group)
        tr_end = int(n * train_r)
        va_end = tr_end + int(n * val_r)
        for r in group[:tr_end]:
            r["split"] = "train"
        for r in group[tr_end:va_end]:
            r["split"] = "val"
        for r in group[va_end:]:
            r["split"] = "test"


def write_split(rows: list[dict], metadata_dir: Path, prefix: str, split: str) -> Path:
    out = metadata_dir / f"{prefix}_{split}_labels.csv"
    with out.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([r for r in rows if r["split"] == split])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Dedup near-duplicates then split.")
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--out-prefix", default="dedup")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_dir = ROOT_DIR / args.metadata_dir
    rows = load_all_rows(metadata_dir)
    print(f"Loaded {len(rows)} labelled images from current manifests. Hashing...")
    hashes = np.empty(len(rows), dtype=np.uint64)
    for i, r in enumerate(rows):
        hashes[i] = phash64(ROOT_DIR / r["image_path"])
        if (i + 1) % 1500 == 0:
            print(f"  hashed {i+1}/{len(rows)}")
    crops = [r["crop_label"] for r in rows]

    orig = {}
    for c in crops:
        orig[c] = orig.get(c, 0) + 1
    print("\n=== Dedup sweep (unique images kept per crop) ===")
    print(f"{'threshold':>9} | " + " ".join(f"{c:>7}" for c in sorted(orig)))
    print(f"{'original':>9} | " + " ".join(f"{orig[c]:>7}" for c in sorted(orig)))
    for t in (0, 2, 3, 5):
        counts = unique_count_per_crop(cluster_within_crop(hashes, crops, t), crops)
        print(f"{t:>9} | " + " ".join(f"{counts.get(c,0):>7}" for c in sorted(orig)))

    # Dedup at chosen threshold: keep lexicographically-first path per cluster
    dsu = cluster_within_crop(hashes, crops, args.threshold)
    rep_by_root: dict[int, int] = {}
    for i in range(len(rows)):
        root = dsu.find(i)
        if root not in rep_by_root or rows[i]["image_path"] < rows[rep_by_root[root]]["image_path"]:
            rep_by_root[root] = i
    kept = [dict(rows[i]) for i in sorted(rep_by_root.values())]
    print(f"\nKept {len(kept)}/{len(rows)} images after dedup at Hamming <= {args.threshold}.")

    stratified_split(kept, args.seed, 0.70, 0.15)
    for split in ("train", "val", "test"):
        out = write_split(kept, metadata_dir, args.out_prefix, split)
        summary = validate_manifest(out, project_root=ROOT_DIR)
        print(f"{out.name}: {summary.rows} rows, valid={summary.is_valid} | "
              f"crops={summary.crop_counts} grades={summary.grade_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
