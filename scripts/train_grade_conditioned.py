"""Train the crop-conditioned, grade-only model and evaluate it (overall + per crop).

The crop is a known input (farmer selects it); the model predicts grade only, conditioned on
the crop embedding. Reuses the same grade class weights + grade-balanced sampler as the main
pipeline. Compares directly against the crop-agnostic single-task grader (0.928 ± 0.013).

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
        python scripts/train_grade_conditioned.py --config configs/four_crops_dedup_15ep.yaml
"""

from __future__ import annotations

import argparse
import copy
import random
from collections import defaultdict
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from sklearn.metrics import f1_score
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install ML deps: pip install -e .[ml]") from exc

from crop_grading.constants import CROP_INDEX_TO_LABEL, GRADE_CLASSES
from crop_grading.data.dataset import CropGradeDataset
from crop_grading.data.sampling import build_balanced_sampler, sample_subset_indexes
from crop_grading.data.transforms import build_eval_transforms, build_train_transforms
from crop_grading.models.conditioned_model import CropConditionedGradeModel
from crop_grading.training.class_weights import compute_class_weights_from_subset
from crop_grading.utils.config import load_config
from crop_grading.utils.device import pick_device


def make_loader(ds, subset, batch_size, workers, sampler=None):
    return DataLoader(subset, batch_size=batch_size, shuffle=(sampler is None),
                      sampler=sampler, num_workers=workers)


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    gts, preds, crops = [], [], []
    for batch in loader:
        logits = model(batch["image"].to(device), batch["crop_label"].to(device))
        p = logits.argmax(1).cpu()
        gts += batch["grade_label"].tolist()
        preds += p.tolist()
        crops += batch["crop_label"].tolist()
    return gts, preds, crops


def acc_adj_mf1(gts, preds):
    acc = sum(g == p for g, p in zip(gts, preds)) / len(gts)
    adj = sum(abs(g - p) <= 1 for g, p in zip(gts, preds)) / len(gts)
    mf1 = f1_score(gts, preds, average="macro", zero_division=0)
    return acc, adj, mf1


def main() -> int:
    parser = argparse.ArgumentParser(description="Train crop-conditioned grade-only model.")
    parser.add_argument("--config", default="configs/four_crops_dedup_15ep.yaml")
    parser.add_argument("--crop-embed-dim", type=int, default=16)
    parser.add_argument("--checkpoint-dir", default="models/checkpoints/grade_cond_dedup")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT_DIR / args.config)
    seed = args.seed if args.seed is not None else cfg.project.seed
    epochs = args.epochs if args.epochs is not None else cfg.training.epochs
    torch.manual_seed(seed); random.seed(seed)
    device = pick_device()
    workers = cfg.data.num_workers

    train_ds = CropGradeDataset(ROOT_DIR / cfg.data.train_manifest, project_root=ROOT_DIR,
                                transform=build_train_transforms(cfg.data.image_size))
    val_ds = CropGradeDataset(ROOT_DIR / cfg.data.val_manifest, project_root=ROOT_DIR,
                              transform=build_eval_transforms(cfg.data.image_size))
    test_ds = CropGradeDataset(ROOT_DIR / cfg.data.test_manifest, project_root=ROOT_DIR,
                               transform=build_eval_transforms(cfg.data.image_size))
    train_sub = Subset(train_ds, sample_subset_indexes(train_ds, max_samples=None, seed=seed))
    val_sub = Subset(val_ds, sample_subset_indexes(val_ds, max_samples=None, seed=seed))

    sampler = None
    if cfg.training.sampler == "balanced":
        sampler = build_balanced_sampler(train_ds, train_sub, balance_by=cfg.training.balance_by, seed=seed)
    train_loader = make_loader(train_ds, train_sub, cfg.training.batch_size, workers, sampler)
    val_loader = make_loader(val_ds, val_sub, cfg.training.batch_size, workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=workers)

    grade_w = None
    if cfg.training.class_weights in {"grade", "both"}:
        grade_w = compute_class_weights_from_subset(train_ds, train_sub, label_type="grade",
                                                    num_classes=cfg.model.num_grades).to(device)

    model = CropConditionedGradeModel(
        backbone_name=cfg.model.backbone, num_crops=cfg.model.num_crops,
        num_grades=cfg.model.num_grades, crop_embed_dim=args.crop_embed_dim,
        dropout_rate=cfg.model.dropout_rate, pretrained=cfg.model.pretrained,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=grade_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate,
                                  weight_decay=cfg.training.weight_decay)

    print(f"Device {device} | conditioned grade-only | embed_dim {args.crop_embed_dim} | seed {seed}")
    print(f"Train {len(train_sub)} / Val {len(val_sub)} / Test {len(test_ds)} | "
          f"grade weights {[round(w,3) for w in (grade_w.cpu().tolist() if grade_w is not None else [])]}")

    best_acc, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        if epoch <= cfg.training.warmup_epochs:
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(device), batch["crop_label"].to(device))
            loss = loss_fn(logits, batch["grade_label"].to(device))
            loss.backward()
            optimizer.step()
        vg, vp, _ = collect(model, val_loader, device)
        vacc, vadj, vmf1 = acc_adj_mf1(vg, vp)
        improved = vacc > best_acc + cfg.training.min_delta
        if improved:
            best_acc, best_state, no_improve = vacc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        print(f"epoch={epoch} val_grade_acc={vacc:.4f} val_adj={vadj:.4f} val_macroF1={vmf1:.4f}"
              + ("  *best*" if improved else ""))
        if cfg.training.patience > 0 and no_improve >= cfg.training.patience:
            print(f"Early stopping at epoch {epoch}."); break

    ckpt_dir = ROOT_DIR / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state}, ckpt_dir / "best_model.pth")
    model.load_state_dict(best_state)

    # Final TEST evaluation
    gts, preds, crops = collect(model, test_loader, device)
    acc, adj, mf1 = acc_adj_mf1(gts, preds)
    print(f"\n=== TEST (n={len(gts)}) === grade_acc={acc:.4f}  adjacent={adj:.4f}  macro-F1={mf1:.4f}")
    print(f"{'crop':8s} {'n':>4s} {'grade_acc':>9s} {'macroF1':>8s}")
    by = defaultdict(lambda: {"g": [], "p": []})
    for g, p, c in zip(gts, preds, crops):
        by[CROP_INDEX_TO_LABEL[c]]["g"].append(g); by[CROP_INDEX_TO_LABEL[c]]["p"].append(p)
    for crop in sorted(by):
        cg, cp = by[crop]["g"], by[crop]["p"]
        cacc = sum(a == b for a, b in zip(cg, cp)) / len(cg)
        cmf1 = f1_score(cg, cp, average="macro", zero_division=0)
        print(f"{crop:8s} {len(cg):>4d} {cacc:>9.4f} {cmf1:>8.4f}")
    # grade confusion
    ng = cfg.model.num_grades
    conf = [[0] * ng for _ in range(ng)]
    for g, p in zip(gts, preds):
        conf[g][p] += 1
    print("grade confusion (rows=true " + "/".join(GRADE_CLASSES) + "):")
    for i, row in enumerate(conf):
        print(f"  {GRADE_CLASSES[i]}: {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
