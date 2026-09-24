"""Runtime settings for the inference service, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

DEFAULT_TRAINED_CROPS = ("wheat", "rice", "mango", "maize")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


@dataclass(frozen=True)
class ServiceSettings:
    """All knobs the service reads. Defaults match the current training setup."""

    checkpoint_path: Path
    config_path: Path
    price_rules_path: Path
    crop_embed_dim: int
    allow_untrained: bool
    service_token: str | None
    max_image_bytes: int
    trained_crops: tuple[str, ...] = field(default=DEFAULT_TRAINED_CROPS)
    model_version: str | None = None

    @classmethod
    def from_env(cls) -> ServiceSettings:
        crops = os.environ.get("MODEL_TRAINED_CROPS")
        return cls(
            checkpoint_path=_resolve(
                os.environ.get(
                    "MODEL_CHECKPOINT",
                    "models/checkpoints/grade_cond_dedup/best_model.pth",
                )
            ),
            config_path=_resolve(
                os.environ.get("MODEL_CONFIG", "configs/four_crops_dedup_15ep.yaml")
            ),
            price_rules_path=_resolve(os.environ.get("PRICE_RULES", "configs/price_rules.yaml")),
            crop_embed_dim=int(os.environ.get("MODEL_CROP_EMBED_DIM", "16")),
            allow_untrained=_env_bool("ALLOW_UNTRAINED_MODEL", True),
            service_token=os.environ.get("AI_SERVICE_TOKEN") or None,
            max_image_bytes=int(os.environ.get("MAX_IMAGE_BYTES", str(8 * 1024 * 1024))),
            trained_crops=(
                tuple(c.strip().lower() for c in crops.split(",") if c.strip())
                if crops
                else DEFAULT_TRAINED_CROPS
            ),
            model_version=os.environ.get("MODEL_VERSION") or None,
        )
