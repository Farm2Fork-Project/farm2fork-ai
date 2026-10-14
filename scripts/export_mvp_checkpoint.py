"""Export the frozen seed-42 public-data grader and a checksum manifest for CPU serving."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source: Path, output_dir: Path) -> dict:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    if "crop_embed.weight" in state or "grade_head.layers.4.weight" not in state:
        raise ValueError("Expected the plain grade-only training checkpoint.")
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "best_model.pth"
    # Optimizer state is deliberately excluded; no weights are trained or altered.
    torch.save({"model_state_dict": state}, output)
    manifest = {
        "exported_at_pkt": datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
        "architecture": "grade_only",
        "backbone": "efficientnet_b0",
        "supported_crops": ["wheat", "rice", "mango", "maize"],
        "unsupported_behavior": "No inference; Other is client-only and direct requests are rejected",
        "grades": ["A", "B", "C", "D"],
        "image_size": 224,
        "seed": 42,
        "source_epoch": checkpoint.get("epoch"),
        "source_path": str(source),
        "source_sha256": digest(source),
        "artifact_sha256": digest(output),
        "artifact_bytes": output.stat().st_size,
        "quality_model_version": f"grade-only-efficientnet_b0-{digest(output)[:8]}",
        "config": "configs/grade_only_four_crops_dedup_15ep.yaml",
        "config_sha256": digest(ROOT / "configs/grade_only_four_crops_dedup_15ep.yaml"),
        "purpose": "Optional MVP suggestion, farmer-confirmed; not a certified assessment",
        "evaluation_records": {
            "curated_public_test": {"n": 466, "exact_accuracy": 439 / 466},
            "supplied_photo_diagnostic": {
                "n": 25,
                "exact_accuracy": 7 / 25,
                "provenance_and_independence_review_required": True,
            },
        },
        "selection_basis": "Previously established seed-42 public validation checkpoint; no field-set tuning",
        "new_training_performed": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "SHA256SUMS").write_text(f"{digest(output)}  best_model.pth\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "models/checkpoints/grade_only_four_crops_dedup_15ep/best_model.pth",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
