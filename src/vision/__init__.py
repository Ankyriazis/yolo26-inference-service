"""YOLO26 inference service: detection pipelines, batching, telemetry and HTTP layer."""

import os

__version__ = "2.0.0"

# Ultralytics pip-installs missing requirements at runtime by default. In a
# service that is a supply-chain hole and a reproducibility bug: it would, for
# example, "satisfy" nvidia-modelopt[onnx] by replacing the CUDA 13 build of
# onnxruntime-gpu with a CUDA 12 one. Dependencies come from the lockfile only.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from .batching import DynamicBatcher, QueueFullError
from .detector import (
    Detection,
    Detector,
    GpuDetector,
    InferenceResult,
    UltralyticsDetector,
    build_detector,
    inspect_weights,
)
from .imaging import ImageDecodeError, annotate, decode_image, encode_jpeg
from .metrics import Metrics

__all__ = [
    "Detection",
    "Detector",
    "DynamicBatcher",
    "GpuDetector",
    "ImageDecodeError",
    "InferenceResult",
    "Metrics",
    "QueueFullError",
    "UltralyticsDetector",
    "annotate",
    "build_detector",
    "decode_image",
    "encode_jpeg",
    "inspect_weights",
]
