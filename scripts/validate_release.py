"""Validate local trained artifacts and emit reproducible release evidence.

This never contacts production or claims that passing artefacts establish field accuracy.
Run with the same MODEL_*/OFFICIAL_PRICES settings used for serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from crop_grading.service.grading import QualityGrader
from crop_grading.service.pricing import OfficialPrices
from crop_grading.service.settings import ServiceSettings


def validate() -> dict:
    settings = ServiceSettings.from_env()
    grader = QualityGrader(settings)
    if grader.status != "trained" or grader.photo_reference is None:
        raise ValueError("Release grading requires trained weights and a matching photo reference")
    if grader.confidence_threshold < 0.7:
        raise ValueError("Release quality confidence threshold must be at least 0.7")
    prices = OfficialPrices(settings.official_prices_path)

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "validation": "local-artifacts-only",
        "qualityModel": grader.status,
        "qualityModelVersion": grader.version,
        "photoCheck": "enabled",
        "trainedCrops": list(grader.trained_crops),
        "weightsSha256": digest(settings.checkpoint_path),
        "configSha256": digest(settings.config_path),
        "photoReferenceSha256": digest(settings.photo_reference_path),
        "pricesSha256": digest(settings.official_prices_path),
        "priceMethod": "official_wholesale",
        "pricesFetchedAt": prices.fetched_at,
        "fieldAccuracyVerified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(validate(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    main()
