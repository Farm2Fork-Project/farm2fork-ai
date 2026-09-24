"""Quality grading around the crop-conditioned grade model.

The farmer states the crop when listing, so the service serves
`CropConditionedGradeModel` (grade-only, conditioned on the crop) rather than
the multi-task model. The checkpoint format is exactly what
`scripts/train_grade_conditioned.py` writes: {"model_state_dict": ...}.

When no trained checkpoint exists yet the service can still run in an explicit
"untrained" mode so the rest of the platform can be tested end to end. Every
response then carries modelStatus="untrained"; clients must present such a
grade as a preview, never as an assessment. Dropping the trained checkpoint in
place (and restarting) switches to modelStatus="trained" with no API change.
"""

from __future__ import annotations

import hashlib
import io
import threading
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from crop_grading.constants import CROP_CLASSES, CROP_LABEL_TO_INDEX, GRADE_CLASSES
from crop_grading.data.transforms import build_eval_transforms
from crop_grading.models.conditioned_model import CropConditionedGradeModel
from crop_grading.service.settings import ServiceSettings
from crop_grading.utils.config import load_config

MODEL_STATUS_TRAINED = "trained"
MODEL_STATUS_UNTRAINED = "untrained"


class InvalidImageError(ValueError):
    """The upload is not a decodable image."""


class ModelUnavailableError(RuntimeError):
    """No trained checkpoint and untrained mode is disabled."""


@dataclass(frozen=True)
class GradePrediction:
    grade: str
    confidence: float
    probabilities: dict[str, float]
    low_confidence: bool


class QualityGrader:
    """Loads the model once and grades images thread-safely on CPU/GPU."""

    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings
        config = load_config(settings.config_path)
        self.image_size = config.data.image_size
        self.confidence_threshold = config.inference.confidence_threshold
        self._transform = build_eval_transforms(self.image_size)
        self._lock = threading.Lock()

        # pretrained=False: never download backbone weights at serve time; the
        # trained checkpoint carries all of them.
        model = CropConditionedGradeModel(
            backbone_name=config.model.backbone,
            num_crops=config.model.num_crops,
            num_grades=config.model.num_grades,
            crop_embed_dim=settings.crop_embed_dim,
            dropout_rate=config.model.dropout_rate,
            pretrained=False,
        )

        if settings.checkpoint_path.is_file():
            checkpoint = torch.load(settings.checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            self.status = MODEL_STATUS_TRAINED
            digest = hashlib.sha256(settings.checkpoint_path.read_bytes()).hexdigest()[:8]
            default_version = f"grade-cond-{config.model.backbone}-{digest}"
        elif settings.allow_untrained:
            torch.manual_seed(0)  # deterministic preview outputs
            self.status = MODEL_STATUS_UNTRAINED
            default_version = f"grade-cond-{config.model.backbone}-untrained"
        else:
            raise ModelUnavailableError(
                f"No trained checkpoint at {settings.checkpoint_path} and "
                "ALLOW_UNTRAINED_MODEL is false."
            )

        self.version = settings.model_version or default_version
        self._model = model.eval()

    @property
    def trained_crops(self) -> tuple[str, ...]:
        return self._settings.trained_crops

    def grade(self, image_bytes: bytes, crop: str) -> GradePrediction:
        crop = crop.lower()
        if crop not in CROP_LABEL_TO_INDEX:
            raise ValueError(f"Unknown crop '{crop}'. Expected one of {', '.join(CROP_CLASSES)}.")
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()  # cheap structural check before a full decode
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise InvalidImageError("The upload is not a supported image.") from exc

        tensor = self._transform(image).unsqueeze(0)
        crop_index = torch.tensor([CROP_LABEL_TO_INDEX[crop]])
        with self._lock, torch.no_grad():
            probs = F.softmax(self._model(tensor, crop_index), dim=1)[0]

        best = int(torch.argmax(probs).item())
        confidence = float(probs[best].item())
        return GradePrediction(
            grade=GRADE_CLASSES[best],
            confidence=round(confidence, 4),
            probabilities={g: round(float(p), 4) for g, p in zip(GRADE_CLASSES, probs)},
            low_confidence=confidence < self.confidence_threshold,
        )
