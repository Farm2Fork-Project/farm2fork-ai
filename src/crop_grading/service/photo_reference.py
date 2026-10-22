"""Checks whether a farmer photo resembles the photos the grader was trained on.

Softmax confidence alone does not catch unfamiliar photos: on the 14 October
supplied phone photographs the model's four most confident predictions were
all wrong. A photo is compared with the training images in the backbone's
embedding space (mean cosine similarity to its k nearest training images).
Photos less similar than the chosen validation percentile are referred for a
retake or a manual grade instead of being graded.

The reference file is tied to exact model weights through a fingerprint of the
state dict, so a different checkpoint can never silently reuse it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def weights_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """SHA-256 over parameter names and values, independent of the file format."""
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PhotoReferenceError(ValueError):
    """The reference file is missing, malformed or belongs to other weights."""


@dataclass(frozen=True)
class PhotoReference:
    embeddings: np.ndarray  # (n, d) float32, L2-normalised training embeddings
    k: int
    min_similarity: float
    fingerprint: str
    provenance: dict

    @classmethod
    def load(cls, path: Path, expected_fingerprint: str) -> PhotoReference:
        if not path.is_file():
            raise PhotoReferenceError(f"No photo reference at {path}.")
        try:
            with np.load(path, allow_pickle=False) as data:
                reference = cls(
                    embeddings=data["embeddings"].astype(np.float32),
                    k=int(data["k"]),
                    min_similarity=float(data["min_similarity"]),
                    fingerprint=str(data["fingerprint"]),
                    provenance=json.loads(str(data["provenance"])),
                )
        except (OSError, KeyError, ValueError) as exc:
            raise PhotoReferenceError("The photo reference file is malformed.") from exc
        if reference.fingerprint != expected_fingerprint:
            raise PhotoReferenceError("The photo reference was built for different weights.")
        return reference

    def similarity(self, embedding: torch.Tensor) -> float:
        """Mean cosine similarity of one normalised embedding to its k nearest neighbours."""
        sims = self.embeddings @ embedding.detach().cpu().numpy().astype(np.float32)
        k = min(self.k, sims.shape[0])
        return float(np.partition(sims, -k)[-k:].mean())


def save_reference(
    path: Path,
    *,
    embeddings: np.ndarray,
    k: int,
    min_similarity: float,
    fingerprint: str,
    provenance: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embeddings.astype(np.float16),
        k=np.int64(k),
        min_similarity=np.float64(min_similarity),
        fingerprint=np.str_(fingerprint),
        provenance=np.str_(json.dumps(provenance, sort_keys=True)),
    )
