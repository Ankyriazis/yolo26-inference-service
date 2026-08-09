"""Runtime configuration, read from environment variables.

Every setting has a working default, so the service starts with no
configuration at all. Anything an operator might need to change without
rebuilding the image is exposed here rather than hard-coded at the call site.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw!r}") from error


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Service configuration.

    Attributes:
        model: YOLO26 weights to load (`yolo26n.pt` … `yolo26x.pt`, or a path).
        device: `cuda:0`, `cpu`, `mps`, or None to detect automatically.
        warmup_iterations: blank inferences run at startup so the first real
            request does not absorb CUDA initialisation cost.
    """

    model: str = field(default_factory=lambda: _env_str("VISION_MODEL", "yolo26n.pt"))
    device: str | None = field(
        default_factory=lambda: os.environ.get("VISION_DEVICE") or None
    )
    confidence: float = field(default_factory=lambda: _env_float("VISION_CONFIDENCE", 0.25))
    iou: float = field(default_factory=lambda: _env_float("VISION_IOU", 0.45))
    image_size: int = field(default_factory=lambda: _env_int("VISION_IMAGE_SIZE", 640))
    max_detections: int = field(
        default_factory=lambda: _env_int("VISION_MAX_DETECTIONS", 300)
    )
    warmup_iterations: int = field(
        default_factory=lambda: _env_int("VISION_WARMUP_ITERATIONS", 2)
    )
    eager_load: bool = field(default_factory=lambda: _env_bool("VISION_EAGER_LOAD", True))

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("VISION_CONFIDENCE must be in (0, 1].")
        if not 0.0 < self.iou <= 1.0:
            raise ValueError("VISION_IOU must be in (0, 1].")
        if self.image_size < 32 or self.image_size % 32 != 0:
            raise ValueError("VISION_IMAGE_SIZE must be a multiple of 32, at least 32.")
