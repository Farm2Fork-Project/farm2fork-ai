"""Rule-based price estimate (master context 10.3 fallback).

Transparent by design: category/crop reference ranges x seasonal pressure x
quality grade, converted to the requested unit. It is not a learned model and
says so in every response (method="rule_based", capped confidence).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from crop_grading.constants import CROP_CLASSES


class PriceNotEstimableError(ValueError):
    """No rule covers this unit/category (e.g. per-piece items)."""


@dataclass(frozen=True)
class PriceEstimate:
    min_price: float
    max_price: float
    unit: str
    confidence: float
    model_version: str
    basis: str  # "crop" or "category"
    matched_crop: str | None
    seasonal_factor: float
    grade_factor: float


class PriceRules:
    def __init__(self, path: Path) -> None:
        with path.open(encoding="utf-8") as handle:
            rules = yaml.safe_load(handle)
        self.version: str = rules["version"]
        self._max_confidence = float(rules["max_confidence"])
        self._crops: dict = rules["crops"]
        self._categories: dict = rules["categories"]
        self._seasonal: dict = rules["seasonal"]
        self._crop_category: dict = rules["crop_category"]
        self._grade: dict = rules["grade_multiplier"]
        self._units: dict = rules["unit_factor"]

    def estimate(
        self,
        *,
        product_name: str,
        category: str,
        unit: str,
        quality_grade: str | None = None,
        on: date | None = None,
    ) -> PriceEstimate:
        category = category.lower()
        unit = unit.lower()
        factor = self._units.get(unit)
        if factor is None:
            raise PriceNotEstimableError(f"No price rule for unit '{unit}'.")

        crop = _match_crop(product_name)
        if crop and crop in self._crops:
            reference, basis = self._crops[crop], "crop"
            season_key = self._crop_category.get(crop, category)
        elif category in self._categories:
            reference, basis = self._categories[category], "category"
            season_key = category
        else:
            raise PriceNotEstimableError(f"No price rule for category '{category}'.")

        # Seasons are Pakistani seasons: take "today" in Pakistan time.
        month = (on or datetime.now(ZoneInfo("Asia/Karachi")).date()).month
        seasonal = float(self._seasonal.get(season_key, [1.0] * 12)[month - 1])
        grade = float(self._grade.get((quality_grade or "B").upper(), 1.0))
        scale = seasonal * grade * float(factor)
        # A crop match narrows the reference, so it earns slightly more trust.
        confidence = self._max_confidence if basis == "crop" else self._max_confidence - 0.1
        return PriceEstimate(
            min_price=round(reference["min"] * scale, 2),
            max_price=round(reference["max"] * scale, 2),
            unit=unit,
            confidence=round(confidence, 2),
            model_version=self.version,
            basis=basis,
            matched_crop=crop,
            seasonal_factor=round(seasonal, 3),
            grade_factor=round(grade, 3),
        )


def _match_crop(product_name: str) -> str | None:
    words = set(re.findall(r"[a-z]+", product_name.lower()))
    for crop in CROP_CLASSES:
        if crop in words or f"{crop}s" in words or f"{crop}es" in words:
            return crop
    # Common local names.
    aliases = {
        "basmati": "rice",
        "chawal": "rice",
        "gandum": "wheat",
        "aam": "mango",
        "makai": "maize",
        "corn": "maize",
        "kapas": "cotton",
        "ganna": "sugarcane",
    }
    for alias, crop in aliases.items():
        if alias in words:
            return crop
    return None
