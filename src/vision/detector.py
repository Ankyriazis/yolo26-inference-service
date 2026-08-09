"""Object detection backed by an Ultralytics YOLO26 model.

The rest of the service never imports ultralytics. It receives plain
`Detection` records, which keeps the HTTP layer, the tests and the ONNX export
path independent of the inference framework - and makes the detector easy to
stub in tests that must run without a model or a GPU.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One detected object, in pixel coordinates of the source image."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InferenceResult:
    """Detections for one image, plus how long producing them took."""

    detections: list[Detection]
    inference_ms: float
    image_width: int
    image_height: int

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "count": len(self.detections),
            "inference_ms": round(self.inference_ms, 2),
            "image": {"width": self.image_width, "height": self.image_height},
        }


class Detector:
    """Loads a YOLO26 model once and runs inference on BGR frames.

    Loading is the expensive part - several seconds, and more when weights have
    to be downloaded - so a process should build one Detector and keep it. The
    service does this at startup rather than on the first request, so that the
    first caller is not the one who pays for it.
    """

    def __init__(
        self,
        model: str = "yolo26n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: str | None = None,
        max_detections: int = 300,
        classes: Sequence[int] | None = None,
    ) -> None:
        from ultralytics import YOLO  # imported lazily: it is slow and heavy

        self.model_name = model
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.max_detections = max_detections
        self.classes = list(classes) if classes else None

        self.device = device or self._select_device()
        logger.info("loading %s onto %s", model, self.device)
        started = time.perf_counter()
        self.model = YOLO(model)
        self.load_seconds = time.perf_counter() - started
        logger.info("loaded %s in %.2fs", model, self.load_seconds)

    @staticmethod
    def _select_device() -> str:
        """Prefer CUDA when it is genuinely available, else fall back to CPU."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:  # pragma: no cover - torch is a hard dependency
            pass
        return "cpu"

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.names

    def warmup(self, iterations: int = 2) -> None:
        """Run inference on a blank frame so the first real request is not slow.

        The first CUDA call in a process pays for kernel compilation and memory
        pool setup. Doing that here keeps it out of a user's latency.
        """
        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.predict(blank)
        logger.info("warmup complete (%d iterations)", iterations)

    def predict(self, image: np.ndarray) -> InferenceResult:
        """Detect objects in a single BGR image."""
        if image is None or image.size == 0:
            raise ValueError("Received an empty image.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel BGR image, got shape {image.shape}."
            )

        height, width = image.shape[:2]
        started = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            max_det=self.max_detections,
            classes=self.classes,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        return InferenceResult(
            detections=self._to_detections(results),
            inference_ms=elapsed_ms,
            image_width=width,
            image_height=height,
        )

    def _to_detections(self, results: Iterable) -> list[Detection]:
        names = self.model.names
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            for box, confidence, class_id in zip(xyxy, confidences, class_ids):
                detections.append(
                    Detection(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        confidence=float(confidence),
                        class_id=int(class_id),
                        class_name=names.get(int(class_id), str(class_id)),
                    )
                )
        return detections
