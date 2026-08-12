"""RiceCC dataset audit (4 checks) — run after the HF download completes.

1. Within-folder duplicate/near-duplicate rate (pHash Hamming 0/≤3/≤5 + cluster sizes).
2. Cross-folder near-duplicates (same image appearing across different variety folders).
3. Source-separability: frozen-feature linear probe over the 6 variety folders (absurdly high +
   instant ⇒ possible background/capture confound, per the crop-ID lesson).
4. broken_rice audit: is it the same imaging domain as the 6 varieties, or effectively a 7th source?

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
        python scripts/audit_ricecc.py --dir data/raw/ricecc --sample-per-folder 600
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import numpy as np
    import timm
    import torch
    from PIL import Image
    from scipy.fft import dct
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install ML deps: pip install -e .[ml]") from exc

from crop_grading.data.transforms import build_eval_transforms
from crop_grading.utils.device import pick_device

VARIETIES = ["GD", "NM", "PJX", "WC", "WN", "YB"]
BROKEN = "broken_rice"
EXTS = {".jpg", ".jpeg", ".png"}


def phash(path):
    im = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
    c = dct(dct(np.asarray(im, dtype=np.float64), axis=0), axis=1)[:8, :8]
    v = 0
    for b in (c > np.median(c)).flatten():
        v = (v << 1) | int(b)
    return np.uint64(v)


def list_imgs(d):
    return sorted(str(p) for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/ricecc")
    ap.add_argument("--sample-per-folder", type=int, default=600)
    ap.add_argument("--dedup-threshold", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(42)
    base = ROOT_DIR / args.dir

    folders = {f: list_imgs(base / f) for f in VARIETIES + [BROKEN] if (base / f).is_dir()}
    print("folders found:", {k: len(v) for k, v in folders.items()})
    if len(folders) < 3:
        raise SystemExit("Not enough folders present — is the download complete?")

    # ---- pHash everything (sampled if huge) for checks 1 & 2 ----
    paths, labels = [], []
    for f, fs in folders.items():
        take = fs if len(fs) <= 4000 else rng.sample(fs, 4000)
        paths += take
        labels += [f] * len(take)
    print(f"\nHashing {len(paths)} images for dedup checks...")
    H = np.array([phash(p) for p in paths], dtype=np.uint64)
    lab = np.array(labels)

    # Check 1: within-folder near-dup rate
    print("\n=== 1) WITHIN-FOLDER near-duplicates ===")
    for f in folders:
        idx = np.where(lab == f)[0]
        sub = H[idx]
        # count images having another in-folder image within threshold t
        dup0 = dup3 = dup5 = 0
        for a in range(len(sub)):
            d = np.bitwise_count(np.delete(sub, a) ^ sub[a])
            m = d.min() if len(d) else 99
            dup0 += m <= 0; dup3 += m <= 3; dup5 += m <= args.dedup_threshold
        n = len(sub)
        print(f"  {f:12s} n={n:5d} | has-dup ≤0:{100*dup0/n:5.1f}%  ≤3:{100*dup3/n:5.1f}%  "
              f"≤{args.dedup_threshold}:{100*dup5/n:5.1f}%")

    # Check 2: cross-folder near-dup (same image in two variety folders)
    print("\n=== 2) CROSS-FOLDER near-duplicates (labeling/preprocessing red flag) ===")
    cross = {t: 0 for t in (0, 3, 5)}
    for i in range(len(H)):
        d = np.bitwise_count(H ^ H[i])
        d[lab == lab[i]] = 99  # ignore same folder
        m = int(d.min())
        for t in cross:
            cross[t] += m <= t
    for t, c in cross.items():
        print(f"  images with a DIFFERENT-folder match ≤ {t}: {c} / {len(H)} ({100*c/len(H):.2f}%)")

    # ---- frozen features for checks 3 & 4 ----
    print("\n=== 3&4) frozen-feature analysis ===")
    device = pick_device()
    backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0,
                                 global_pool="avg").to(device).eval()
    tf = build_eval_transforms(224)

    feats, flabs = [], []
    with torch.no_grad():
        for f, fs in folders.items():
            take = rng.sample(fs, min(args.sample_per_folder, len(fs)))
            for p in take:
                try:
                    x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                except Exception:
                    continue
                feats.append(backbone(x).cpu().numpy()[0]); flabs.append(f)
    X = np.array(feats); y = np.array(flabs)

    # Check 3: 6-way variety source-separability
    vmask = np.isin(y, VARIETIES)
    Xv, yv = X[vmask], y[vmask]
    Xtr, Xte, ytr, yte = train_test_split(Xv, yv, test_size=0.3, random_state=42, stratify=yv)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"  3) frozen-feature 6-way VARIETY linear probe accuracy: {acc:.4f}  "
          f"(chance ≈ {1/len(VARIETIES):.2f}) — very high ⇒ inspect capture/background confound")

    # Check 4: broken_rice — same domain or 7th source?
    if BROKEN in folders:
        Xb = X[y == BROKEN]
        # mean nearest-neighbour cosine distance: broken→variety vs variety→variety
        def norm(a): return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        Xvn, Xbn = norm(Xv), norm(Xb)
        # broken → nearest variety
        sim_bv = (Xbn @ Xvn.T).max(axis=1)
        # variety → nearest other-variety (sample)
        s = min(400, len(Xvn)); ridx = rng.sample(range(len(Xvn)), s)
        sim_vv = []
        for i in ridx:
            sims = Xvn[i] @ Xvn.T; sims[i] = -1
            sim_vv.append(sims.max())
        print(f"  4) broken_rice domain check (cosine sim to nearest neighbour; higher = closer):")
        print(f"       broken→variety  mean {np.mean(sim_bv):.3f}")
        print(f"       variety→variety mean {np.mean(sim_vv):.3f}")
        # 7-way probe: is broken trivially its own class?
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        clf7 = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        cm = confusion_matrix(yte, clf7.predict(Xte), labels=VARIETIES + [BROKEN])
        brow = cm[-1]
        print(f"       broken_rice recall in 7-way probe: {brow[-1]/max(1,brow.sum()):.3f} "
              f"(≈1.0 ⇒ distinct condition/domain, not interleaved with varieties)")
    print("\nInterpretation: within-folder dups → redundancy; cross-folder dups → label issues; "
          "high variety probe → possible source shortcut; broken clearly separable → whole/broken is a "
          "domain split, so it CANNOT support 'does variety affect brokenness' per-variety.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
