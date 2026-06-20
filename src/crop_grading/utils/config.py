"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator

from crop_grading.constants import CROP_CLASSES, GRADE_CLASSES


class ProjectConfig(BaseModel):
    name: str
    seed: int = 42


class DataConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path
    metadata_dir: Path
    image_size: PositiveInt = 224
    num_workers: int = Field(default=4, ge=0)
    crops: list[str]
    grades: list[str]

    @field_validator("crops")
    @classmethod
    def validate_crops(cls, value: list[str]) -> list[str]:
        if tuple(value) != CROP_CLASSES:
            raise ValueError(f"crops must match {CROP_CLASSES}")
        return value

    @field_validator("grades")
    @classmethod
    def validate_grades(cls, value: list[str]) -> list[str]:
        if tuple(value) != GRADE_CLASSES:
            raise ValueError(f"grades must match {GRADE_CLASSES}")
        return value


class ModelConfig(BaseModel):
    backbone: str = "efficientnet_b3"
    pretrained: bool = True
    dropout_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    num_crops: PositiveInt = len(CROP_CLASSES)
    num_grades: PositiveInt = len(GRADE_CLASSES)


class TrainingConfig(BaseModel):
    batch_size: PositiveInt = 32
    epochs: PositiveInt = 50
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    crop_loss_weight: float = Field(default=0.4, ge=0.0)
    grade_loss_weight: float = Field(default=0.6, ge=0.0)
    warmup_epochs: int = Field(default=3, ge=0)

    @field_validator("grade_loss_weight")
    @classmethod
    def validate_loss_weights(cls, value: float, info: Any) -> float:
        crop_weight = info.data.get("crop_loss_weight")
        if crop_weight is not None and abs((crop_weight + value) - 1.0) > 1e-6:
            raise ValueError("crop_loss_weight and grade_loss_weight must sum to 1.0")
        return value


class InferenceConfig(BaseModel):
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    checkpoint_path: Path


class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    """Load the YAML config and validate values that affect model contracts."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)
    return AppConfig.model_validate(raw_config)
