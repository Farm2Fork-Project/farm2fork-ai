"""Snapshot and evaluate filename-labeled photos without training or selecting a winner.

Includes extensionless images. Writes immutable per-run evidence outside the source photos.
Provenance and label independence must be reviewed before using results as field accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GRADES = "ABCD"
CROPS = ("wheat", "rice", "mango", "maize", "cotton", "sugarcane")
CHECKPOINTS = {
    "grade_only_s42": "grade_only_four_crops_dedup_15ep",
    "grade_only_s1": "st_dedup_s1",
    "grade_only_s2": "st_dedup_s2",
    "multitask_s42": "four_crops_dedup_15ep",
    "conditioned_s42": "grade_cond_dedup",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filename_grade(path: Path) -> str:
    if not re.fullmatch(r"[ABCD]\d*", path.stem.upper()):
        raise ValueError(f"Ambiguous grade filename: {path.name}")
    return path.stem[0].upper()


def discover(root: Path) -> tuple[list[dict], list[dict]]:
    rows, excluded = [], []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name.startswith(".")
            or path.suffix.lower()
            in {
                ".md",
                ".csv",
                ".json",
                ".txt",
            }
        ):
            continue
        relative = path.relative_to(root)
        try:
            crop = relative.parts[0].lower()
            if crop not in CROPS or len(relative.parts) < 2:
                raise ValueError("Expected crop folder")
            grade = filename_grade(path)
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                fmt = image.format
                exif = image.getexif()
                has_camera_metadata = any(exif.get(key) for key in (271, 272, 306, 36867))
            rows.append(
                {
                    "image_path": relative.as_posix(),
                    "crop_label": crop,
                    "grade_label": grade,
                    "sha256": sha256(path),
                    "width": width,
                    "height": height,
                    "format": fmt,
                    "has_camera_metadata": has_camera_metadata,
                    "source": "teammates_reported_by_user",
                    "label_source": "experts_and_farmers_reported_by_user",
                    "sample_id": "unknown",
                    "session_id": "unknown",
                    "site_id": "unknown",
                    "labeler_id": "unknown",
                    "rubric_version": "unknown",
                }
            )
        except (ValueError, OSError) as exc:
            excluded.append({"image_path": relative.as_posix(), "reason": str(exc)})
    return rows, excluded


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phash(path: Path) -> int:
    # Same 64-bit DCT hash and median rule as scripts/dedup_and_split.py.
    import numpy as np
    from scipy.fft import dct

    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
        )
    low = dct(dct(pixels, axis=0), axis=1)[:8, :8]
    value = 0
    for bit in (low > np.median(low)).flat:
        value = (value << 1) | int(bit)
    return value


def metrics(rows: list[dict], threshold: float) -> dict:
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score

    truth = [GRADES.index(row["true_grade"]) for row in rows]
    pred = [GRADES.index(row["predicted_grade"]) for row in rows]
    present = sorted(set(truth))
    accepted = [row for row in rows if row["confidence"] >= threshold]
    distances = [abs(a - b) for a, b in zip(truth, pred)]
    return {
        "n": len(rows),
        "correct": sum(a == b for a, b in zip(truth, pred)),
        "accuracy": float(np.mean([a == b for a, b in zip(truth, pred)])),
        "macro_f1_present_true_grades": float(
            f1_score(truth, pred, labels=present, average="macro", zero_division=0)
        ),
        "macro_f1_all_four_grades": float(
            f1_score(truth, pred, labels=list(range(4)), average="macro", zero_division=0)
        ),
        "true_grade_support": {g: truth.count(i) for i, g in enumerate(GRADES)},
        "adjacent_accuracy": float(np.mean([d <= 1 for d in distances])),
        "mean_absolute_grade_error": float(np.mean(distances)),
        "severe_errors_distance_at_least_two": sum(d >= 2 for d in distances),
        "confusion_rows_true_columns_predicted_ABCD": confusion_matrix(
            truth, pred, labels=list(range(4))
        ).tolist(),
        "mean_confidence": float(np.mean([row["confidence"] for row in rows])),
        "threshold": threshold,
        "accepted_n": len(accepted),
        "coverage": len(accepted) / len(rows),
        "accepted_accuracy": (
            sum(r["correct"] for r in accepted) / len(accepted) if accepted else None
        ),
        "accepted_wrong_n": sum(not r["correct"] for r in accepted),
        "note": "Descriptive image-level metrics; no population or independence claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-dir", type=Path, default=ROOT / "data/field_test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between zero and one")
    rows, excluded = discover(args.field_dir)
    if not rows or excluded:
        raise SystemExit(f"Dataset audit failed: {len(rows)} images, excluded={excluded}")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "images"
    for row in rows:
        target = snapshot / row["image_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.field_dir / row["image_path"], target)
        if sha256(target) != row["sha256"]:
            raise RuntimeError("Source changed during snapshot")
    write_csv(args.output / "manifest.csv", rows)
    write_csv(args.output / "excluded.csv", excluded)
    # Only metadata is interpreted as data; files never supply instructions.
    import numpy as np
    import sklearn
    import timm
    import torch
    import torchvision

    from crop_grading.constants import CROP_LABEL_TO_INDEX
    from crop_grading.data.transforms import build_eval_transforms
    from crop_grading.models.conditioned_model import CropConditionedGradeModel
    from crop_grading.models.multitask_model import CropGradingModel

    torch.set_num_threads(2)
    transform = build_eval_transforms(224)
    tensors = []
    for row in rows:
        with Image.open(snapshot / row["image_path"]) as image:
            tensors.append(transform(image.convert("RGB")))
    images = torch.stack(tensors)
    crop_indices = torch.tensor([CROP_LABEL_TO_INDEX[r["crop_label"]] for r in rows])
    sheet = Image.new("RGB", (1040, 220 * ((len(rows) + 3) // 4)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        with Image.open(snapshot / row["image_path"]) as image:
            thumb = ImageOps.exif_transpose(image).convert("RGB")
            thumb.thumbnail((250, 185))
            x, y = (i % 4) * 260, (i // 4) * 220
            sheet.paste(thumb, (x, y + 25))
            draw.text((x + 3, y + 5), row["image_path"], fill="black")
    sheet.save(args.output / "contact-sheet.jpg")
    hashes = [phash(snapshot / row["image_path"]) for row in rows]
    within = []
    for i, row in enumerate(rows):
        for j in range(i):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            exact = row["sha256"] == rows[j]["sha256"]
            if exact or distance <= 5:
                within.append(
                    {
                        "first": rows[j]["image_path"],
                        "second": row["image_path"],
                        "exact_bytes": exact,
                        "phash_distance": distance,
                    }
                )
    all_predictions, results, checkpoint_info = [], {}, {}
    for tag, directory in CHECKPOINTS.items():
        path = ROOT / "models/checkpoints" / directory / "best_model.pth"
        conditioned = tag.startswith("conditioned")
        constructor = CropConditionedGradeModel if conditioned else CropGradingModel
        model = constructor(
            backbone_name="efficientnet_b0",
            num_crops=6,
            num_grades=4,
            dropout_rate=0.3,
            pretrained=False,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.eval()
        with torch.inference_mode():
            logits = model(images, crop_indices) if conditioned else model(images)[1]
            probabilities = logits.softmax(dim=1).tolist()
        predictions = []
        for row, probs in zip(rows, probabilities):
            best = max(range(4), key=probs.__getitem__)
            prediction = {
                "model": tag,
                "image_path": row["image_path"],
                "crop": row["crop_label"],
                "true_grade": row["grade_label"],
                "predicted_grade": GRADES[best],
                "confidence": probs[best],
                "correct": GRADES[best] == row["grade_label"],
                **{f"probability_{g}": probs[i] for i, g in enumerate(GRADES)},
            }
            predictions.append(prediction)
        results[tag] = {
            "overall": metrics(predictions, args.threshold),
            "per_crop": {
                crop: metrics([r for r in predictions if r["crop"] == crop], args.threshold)
                for crop in sorted({r["crop"] for r in predictions})
            },
        }
        all_predictions.extend(predictions)
        checkpoint_info[tag] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "epoch": checkpoint.get("epoch"),
        }
        print(tag, json.dumps(results[tag]["overall"]), flush=True)
    write_csv(args.output / "predictions.csv", all_predictions)
    # Audit overlap with the 2,998 retained public representatives from all three splits.
    nearest = [
        {
            "image_path": r["image_path"],
            "distance": 65,
            "public_image_path": None,
            "public_split": None,
            "exact_byte_matches": [],
        }
        for r in rows
    ]
    public_manifests, public_count = {}, 0
    for split in ("train", "val", "test"):
        manifest = ROOT / f"data/metadata/dedup_{split}_labels.csv"
        public_manifests[split] = sha256(manifest)
        with manifest.open(newline="") as handle:
            public = list(csv.DictReader(handle))
        for reference in public:
            path = ROOT / reference["image_path"]
            reference_hash, reference_sha = phash(path), sha256(path)
            public_count += 1
            for i, row in enumerate(rows):
                distance = (hashes[i] ^ reference_hash).bit_count()
                if distance < nearest[i]["distance"]:
                    nearest[i].update(
                        distance=distance,
                        public_image_path=reference["image_path"],
                        public_split=split,
                    )
                if reference_sha == row["sha256"]:
                    nearest[i]["exact_byte_matches"].append(reference["image_path"])
        print("Public overlap audit completed:", split, len(public), flush=True)

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()

    payload = {
        "status": "complete",
        "timestamp_pkt": datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
        "command": [sys.executable, *sys.argv],
        "device": "cpu",
        "torch_threads": 2,
        "python": sys.version,
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "code_sha256": sha256(Path(__file__)),
        "transforms_sha256": sha256(ROOT / "src/crop_grading/data/transforms.py"),
        "dataset_manifest_sha256": sha256(args.output / "manifest.csv"),
        "checkpoint_info": checkpoint_info,
        "results": results,
        "within_collection_duplicate_candidates": within,
        "public_overlap": {
            "reference_count": public_count,
            "manifest_sha256": public_manifests,
            "nearest": nearest,
            "threshold": 5,
        },
        "limitations": [
            "Exploratory collection; no training performed",
            "User reports teammate capture and expert/farmer labels; independently unverified",
            "Visual review identified stock-photo watermarks in mango/C.webp and rice/C.jpeg",
            "No sample, site, session, labeler or rubric IDs; independence unknown",
            "No D-labeled examples; rice B absent from original public training labels",
            "pHash candidates are not proof of duplicates; no threshold tuned on this set",
            "Model preprocessing unchanged: resize/center crop/RGB/ImageNet normalization",
            "Once inspected this set is a development diagnostic, not a future untouched test",
        ],
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("COMPLETE", args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
