"""Checks that a photo shows the crop the farmer selected.

The grader is only meaningful for the selected crop, so a rice photo submitted
as wheat must not be graded as wheat. A crop head trained on the public
datasets cannot do this: on independent rice photos it was right 18.7% of the
time, and a CLIP linear probe trained on the same data fell from 100% on the
public test set to 12/25 on phone photos. An open-vocabulary CLIP model, used
zero-shot, transfers better (20/25 on the 14 October supplied photos), so this
module uses it with fixed text prompts and no training.

The check only refers: a mismatch asks the farmer to retake the photo or grade
manually. It never blocks a listing. Known limit: on the supplied photos it
called all four real maize photos wheat, so maize is referred often.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

MODEL_NAME = "ViT-B-32-quickgelu"  # OpenAI weights were trained with QuickGELU
MODEL_ID = "openai/clip-vit-base-patch32"

CROP_PROMPTS: dict[str, tuple[str, ...]] = {
    "wheat": ("a photo of wheat grains", "a pile of wheat grain", "wheat kernels"),
    "rice": ("a photo of rice grains", "a pile of raw rice", "uncooked rice"),
    "mango": ("a photo of mangoes", "a mango fruit", "ripe mangoes in a crate"),
    "maize": (
        "a pile of yellow corn kernels in a sack",
        "dried yellow maize grains",
        "a corn cob",
    ),
}
# Things a farmer might photograph by mistake. They only compete for probability
# mass; the check is "how likely is the selected crop among all of these".
DISTRACTOR_PROMPTS: dict[str, tuple[str, ...]] = {
    "cotton": ("a photo of cotton bolls",),
    "sugarcane": ("a photo of sugarcane stalks",),
    "banana": ("a photo of bananas",),
    "potato": ("a photo of potatoes",),
    "other_food": (
        "a photo of vegetables",
        "a photo of beans and lentils",
        "a photo of flour",
    ),
    "non_crop": (
        "a photo of a person",
        "a photo of a room",
        "a photo of a document",
        "a screenshot",
    ),
}
LOGIT_SCALE = 100.0

ImageEncoder = Callable[[Image.Image], torch.Tensor]


class CropMatchError(RuntimeError):
    """The crop-match model could not be loaded."""


@dataclass(frozen=True)
class CropMatchChecker:
    """Probability that a photo shows the selected crop, among all prompt classes."""

    encode_image: ImageEncoder
    class_names: tuple[str, ...]
    class_embeddings: torch.Tensor  # (classes, d), L2-normalised
    min_probability: float
    model_id: str = MODEL_ID

    def probability(self, image: Image.Image, crop: str) -> float:
        if crop not in self.class_names:
            raise ValueError(f"Crop '{crop}' has no crop-match prompts.")
        with torch.no_grad():
            embedding = self.encode_image(image).reshape(-1)
            embedding = embedding / embedding.norm()
            probabilities = (LOGIT_SCALE * embedding @ self.class_embeddings.T).softmax(-1)
        return float(probabilities[self.class_names.index(crop)])

    def matches(self, probability: float) -> bool:
        return probability >= self.min_probability

    @classmethod
    def load(cls, weights_path: Path, min_probability: float) -> CropMatchChecker:
        """Build from a local OpenAI CLIP ViT-B/32 weights file. Never downloads."""
        if not weights_path.is_file():
            raise CropMatchError(f"No crop-match weights at {weights_path}.")
        if not 0 < min_probability < 1:
            raise CropMatchError("CROP_MATCH_MIN_PROBABILITY must be between 0 and 1.")
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=str(weights_path)
            )
            tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        except Exception as exc:  # open_clip raises many types for bad files
            raise CropMatchError("The crop-match model could not be loaded.") from exc
        model.eval()

        names = (*CROP_PROMPTS, *DISTRACTOR_PROMPTS)
        prompts = {**CROP_PROMPTS, **DISTRACTOR_PROMPTS}
        with torch.no_grad():
            rows = []
            for name in names:
                text = model.encode_text(tokenizer(list(prompts[name])))
                text = text / text.norm(dim=-1, keepdim=True)
                mean = text.mean(0)
                rows.append(mean / mean.norm())
            embeddings = torch.stack(rows)

        def encode(image: Image.Image) -> torch.Tensor:
            return model.encode_image(preprocess(image.convert("RGB")).unsqueeze(0))

        return cls(encode, tuple(names), embeddings, min_probability)
