"""YOLO26 inference service: detection core, imaging helpers and HTTP layer."""

from .detector import Detection, Detector, InferenceResult
from .imaging import ImageDecodeError, annotate, decode_image, encode_jpeg
from .metrics import Metrics

__all__ = [
    "Detection",
    "Detector",
    "InferenceResult",
    "ImageDecodeError",
    "annotate",
    "decode_image",
    "encode_jpeg",
    "Metrics",
]

__version__ = "1.0.0"
