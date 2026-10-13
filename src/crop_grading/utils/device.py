"""Device selection helper.

Prefers CUDA, then Apple Silicon MPS, then CPU. This lets training and
evaluation use the GPU on Apple Silicon Macs (M1/M2/M3) instead of falling
back to CPU, which is several times slower for CNN workloads.
"""

from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Args:
        prefer: Optional explicit device string ("cuda", "mps", "cpu"). If set,
            it is used as-is (useful for forcing CPU in tests or debugging).
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
