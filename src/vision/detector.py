"""Object detection backed by YOLO26, through one of two pipelines.

`UltralyticsDetector` is the stock path: `YOLO(...).predict()`. It runs any
weights format on any device, and is the reference the other is measured
against.

`GpuDetector` keeps the whole request on the device. Decode (optionally),
letterbox, normalise, forward pass, box decoding and rescaling all run as CUDA
work, and the only device-to-host copy per batch is the final
`(batch, 300, 6)` detection tensor - a few kilobytes. It exists because once
the model runs on TensorRT the forward pass stops being the expensive part:
CPU-side resizing and per-box Python post-processing start to dominate, and
the GPU sits idle waiting for them. YOLO26's NMS-free head is what makes this
cheap to write: post-processing is a threshold and an affine rescale, with no
NMS to port.

The rest of the service never imports ultralytics or torch. It receives plain
`Detection` records, which keeps the HTTP layer and its tests independent of
the inference framework - and makes the detector easy to stub in tests that
must run without a model or a GPU.
"""

from __future__ import annotations

import ast
import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- records


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
    """Detections for one image, plus how producing them was spent.

    `inference_ms` is the wall time of the whole batch this image rode in;
    with batching, several results share one figure. `stages_ms` splits that
    batch time by pipeline stage.
    """

    detections: list[Detection]
    inference_ms: float
    image_width: int
    image_height: int
    batch_size: int = 1
    stages_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        body = {
            "detections": [d.to_dict() for d in self.detections],
            "count": len(self.detections),
            "inference_ms": round(self.inference_ms, 2),
            "batch_size": self.batch_size,
            "image": {"width": self.image_width, "height": self.image_height},
        }
        if self.stages_ms:
            body["stages_ms"] = {k: round(v, 3) for k, v in self.stages_ms.items()}
        return body


class UnsupportedModelError(RuntimeError):
    """The weights cannot run on the requested pipeline."""


# ---------------------------------------------------------------- weights


@dataclass(frozen=True)
class WeightsInfo:
    """What a weights file is, read without loading it."""

    path: str
    backend: str  # pytorch | onnxruntime | tensorrt | openvino | torchscript
    precision: str  # fp32 | fp16 | int8
    max_batch: int | None  # None: no hard limit
    dynamic: bool
    end2end: bool | None  # None: decided at load time (.pt)
    metadata: dict


_SUFFIX_BACKEND = {
    ".pt": "pytorch",
    ".onnx": "onnxruntime",
    ".engine": "tensorrt",
    ".torchscript": "torchscript",
}


def _engine_metadata(path: Path) -> dict:
    """Ultralytics engines carry a JSON header: 4-byte length, then JSON."""
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(4), "little")
        if 0 < size < 1 << 20:
            try:
                return json.loads(handle.read(size))
            except ValueError:
                pass
    return {}


def _onnx_metadata(path: Path) -> dict:
    try:  # reads the protobuf map directly; needs neither onnx nor onnxruntime
        from ultralytics.nn.backends.base import BaseBackend

        return BaseBackend.read_metadata(path)
    except Exception:
        return {}


def _literal(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def inspect_weights(model: str, half: bool = False) -> WeightsInfo:
    path = Path(model)
    if path.is_dir() and path.name.endswith("_openvino_model"):
        return WeightsInfo(model, "openvino", "fp32", None, True, None, {})
    backend = _SUFFIX_BACKEND.get(path.suffix.lower(), "pytorch")

    metadata: dict = {}
    if path.exists() and backend == "tensorrt":
        metadata = _engine_metadata(path)
    elif path.exists() and backend == "onnxruntime":
        metadata = _onnx_metadata(path)
    args = _literal(metadata.get("args", {})) or {}

    quantize = args.get("quantize")
    if quantize in (8, "8", "int8") or args.get("int8"):
        precision = "int8"
    elif quantize in (16, "16", "fp16") or args.get("half") or (backend == "pytorch" and half):
        precision = "fp16"
    else:
        precision = "fp32"

    dynamic = bool(args.get("dynamic", backend in {"pytorch", "torchscript"}))
    batch = int(metadata.get("batch", args.get("batch", 1)) or 1)
    max_batch = None if backend == "pytorch" else batch if not dynamic else max(batch, 1)
    end2end = None
    if metadata:
        end2end = bool(_literal(metadata.get("end2end", False)) or args.get("nms") is False)
    return WeightsInfo(model, backend, precision, max_batch, dynamic, end2end, metadata)


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


# ---------------------------------------------------------------- stock pipeline


class UltralyticsDetector:
    """Loads a YOLO26 model once and runs the stock Ultralytics predictor.

    Loading is the expensive part - several seconds, and more when weights have
    to be downloaded - so a process should build one detector and keep it. The
    service does this at startup rather than on the first request, so that the
    first caller is not the one who pays for it.
    """

    pipeline = "ultralytics"
    accepts_encoded = False

    def __init__(
        self,
        model: str = "yolo26n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: str | None = None,
        max_detections: int = 300,
        classes: Sequence[int] | None = None,
        half: bool = False,
        nms_free: bool = True,
    ) -> None:
        from ultralytics import YOLO  # imported lazily: it is slow and heavy

        self.info = inspect_weights(model, half)
        self.model_name = Path(model).name
        self.backend = self.info.backend
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.max_detections = max_detections
        self.classes = list(classes) if classes else None
        self.device = device or _select_device()
        self.half = half and self.device.startswith("cuda") and self.backend == "pytorch"
        self.precision = "fp16" if self.half else self.info.precision
        self.nms_free = nms_free
        self.max_batch_size = self.info.max_batch or 64

        logger.info("loading %s onto %s (%s)", model, self.device, self.backend)
        started = time.perf_counter()
        self.model = YOLO(model, task="detect")
        self.load_seconds = time.perf_counter() - started
        logger.info("loaded %s in %.2fs", model, self.load_seconds)

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.names

    def warmup(self, iterations: int = 2, batch_sizes: Iterable[int] = (1,)) -> None:
        """Run inference on blank frames so the first real request is not slow.

        The first CUDA call in a process pays for kernel selection and memory
        pool setup, and a dynamic TensorRT engine pays again for each new
        input shape. Doing that here keeps it out of a user's latency.
        """
        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        for batch in batch_sizes:
            for _ in range(iterations):
                self.predict_batch([blank] * batch)
        logger.info("warmup complete (%d iterations, batches %s)", iterations, list(batch_sizes))

    def predict(self, image: np.ndarray) -> InferenceResult:
        """Detect objects in a single BGR image."""
        return _raise_if_error(self.predict_batch([image])[0])

    def predict_batch(self, images: Sequence[Any]) -> list[InferenceResult | Exception]:
        for image in images:
            _check_image(image)

        started = time.perf_counter()
        results = self.model.predict(
            source=list(images),
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            max_det=self.max_detections,
            classes=self.classes,
            quantize=16 if self.half else None,
            nms=False if self.nms_free else None,
            # Minimal-padding letterboxes (640x480 for a 3:4 photo) only suit
            # weights that accept any shape. An exported engine is built for a
            # fixed input size, so pad to the full square.
            rect=self.backend == "pytorch",
            batch=len(images),
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        speed = getattr(results[0], "speed", None) or {}
        # Ultralytics reports per-image averages; scale back to the batch.
        stages = {k: float(v) * len(images) for k, v in speed.items() if v is not None}

        names = self.model.names
        out: list[InferenceResult | Exception] = []
        for image, result in zip(images, results, strict=True):
            out.append(
                InferenceResult(
                    detections=list(_boxes_to_detections(result, names)),
                    inference_ms=elapsed_ms,
                    image_width=image.shape[1],
                    image_height=image.shape[0],
                    batch_size=len(images),
                    stages_ms=stages,
                )
            )
        return out


# Backwards-compatible name from v1.
Detector = UltralyticsDetector


def _boxes_to_detections(result: Any, names: dict[int, str]) -> Iterable[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    for box, confidence, class_id in zip(xyxy, confidences, class_ids, strict=True):
        yield Detection(
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
            confidence=float(confidence),
            class_id=int(class_id),
            class_name=names.get(int(class_id), str(class_id)),
        )


def _check_image(image: Any) -> None:
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("Received an empty image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR image, got shape {image.shape}.")


def _raise_if_error(result: InferenceResult | Exception) -> InferenceResult:
    if isinstance(result, Exception):
        raise result
    return result


# ---------------------------------------------------------------- GPU pipeline


class GpuDetector:
    """Decode-to-detections on the GPU, around any CUDA-capable backend.

    The forward pass goes through Ultralytics' `AutoBackend`, so the same
    class serves `.pt` (PyTorch, optionally FP16), `.onnx` (ONNX Runtime CUDA)
    and `.engine` (TensorRT) weights. Everything either side of the forward
    pass is plain PyTorch on the device:

    - JPEG bytes are decoded by nvJPEG in one batched call when `gpu_decode`
      is on; already-decoded BGR arrays are uploaded as uint8 (a third of the
      bytes of float32) and converted on the device.
    - Letterboxing is a bilinear resize into a preallocated, grey-filled batch
      tensor - the same geometry as Ultralytics' CPU `LetterBox`.
    - YOLO26's one-to-one head emits `(batch, 300, 6)` final boxes, so
      post-processing is a threshold plus an affine rescale. The one-to-many
      head is handled too, with `torchvision.ops.batched_nms` on the device.

    Stage timings come from CUDA events, so they measure device time without
    adding a host synchronisation between stages.
    """

    pipeline = "gpu"
    accepts_encoded = True

    def __init__(
        self,
        model: str = "yolo26n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: str | None = None,
        max_detections: int = 300,
        classes: Sequence[int] | None = None,
        half: bool = False,
        nms_free: bool = True,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise UnsupportedModelError("the GPU pipeline needs CUDA")
        device = device or "cuda:0"
        if not device.startswith("cuda"):
            raise UnsupportedModelError(f"the GPU pipeline needs a CUDA device, not {device!r}")

        self.info = inspect_weights(model, half)
        if self.info.backend not in {"pytorch", "onnxruntime", "tensorrt", "torchscript"}:
            raise UnsupportedModelError(f"{self.info.backend} weights do not run on CUDA here")

        from ultralytics.nn.autobackend import AutoBackend

        self._torch = torch
        self.device = device
        self._device = torch.device(device)
        self.model_name = Path(model).name
        self.backend = self.info.backend
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.max_detections = max_detections
        self.classes = torch.tensor(list(classes), device=self._device) if classes else None

        logger.info("loading %s onto %s (%s, GPU pipeline)", model, device, self.backend)
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "device": self._device,
            "fp16": half,
            "fuse": True,
            "verbose": False,
        }
        if self.backend == "pytorch":
            kwargs["end2end"] = nms_free
        self.model = AutoBackend(model, **kwargs)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started

        self.end2end = bool(getattr(self.model, "end2end", False))
        self.fp16 = bool(getattr(self.model, "fp16", False))
        self.dtype = torch.float16 if self.fp16 else torch.float32
        self.precision = (
            "fp16" if (self.backend == "pytorch" and self.fp16) else self.info.precision
        )
        self.max_batch_size = self.info.max_batch or 64
        self._names = dict(self.model.names)
        self._pad = 114 / 255.0
        logger.info(
            "loaded %s in %.2fs (end2end=%s, input %s)",
            model,
            self.load_seconds,
            self.end2end,
            self.dtype,
        )

    @property
    def class_names(self) -> dict[int, str]:
        return self._names

    # ------------------------------------------------------------ public

    def warmup(self, iterations: int = 2, batch_sizes: Iterable[int] = (1,)) -> None:
        # A non-square frame, so the resize kernels are warmed too.
        blank = np.zeros((self.image_size * 3 // 4, self.image_size, 3), dtype=np.uint8)
        for batch in batch_sizes:
            for _ in range(iterations):
                self.predict_batch([blank] * batch)
        logger.info("warmup complete (%d iterations, batches %s)", iterations, list(batch_sizes))

    def predict(self, image: Any) -> InferenceResult:
        return _raise_if_error(self.predict_batch([image])[0])

    def predict_batch(self, items: Sequence[Any]) -> list[InferenceResult | Exception]:
        """Detect objects in a batch of BGR arrays and/or JPEG byte strings."""
        torch = self._torch
        with torch.inference_mode():
            return self._predict_batch(items)

    # ------------------------------------------------------------ internals

    def _predict_batch(self, items: Sequence[Any]) -> list[InferenceResult | Exception]:
        torch = self._torch
        started = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        events[0].record()

        frames, errors = self._upload(items)
        events[1].record()
        live = [i for i, f in enumerate(frames) if f is not None]
        out: list[InferenceResult | Exception] = [errors.get(i) for i in range(len(items))]
        if not live:
            return out

        batch, gains, pads = self._letterbox([frames[i] for i in live])
        events[2].record()

        raw = self.model(batch)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        events[3].record()

        host = self._postprocess(raw, gains, pads, [frames[i].shape[1:] for i in live])
        events[4].record()
        events[4].synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000

        stages = {
            "upload_decode": events[0].elapsed_time(events[1]),
            "preprocess": events[1].elapsed_time(events[2]),
            "inference": events[2].elapsed_time(events[3]),
            "postprocess": events[3].elapsed_time(events[4]),
        }
        for slot, index in enumerate(live):
            height, width = frames[index].shape[1:]
            out[index] = InferenceResult(
                detections=self._to_detections(host[slot]),
                inference_ms=elapsed_ms,
                image_width=int(width),
                image_height=int(height),
                batch_size=len(live),
                stages_ms=stages,
            )
        return out

    def _upload(self, items: Sequence[Any]) -> tuple[list[Any], dict[int, Exception]]:
        """Every item to a (3, H, W) uint8 RGB CUDA tensor, or an error."""
        torch = self._torch
        frames: list[Any] = [None] * len(items)
        errors: dict[int, Exception] = {}

        encoded = [
            i for i, item in enumerate(items) if isinstance(item, (bytes, bytearray, memoryview))
        ]
        skip = set(encoded)
        for i, item in enumerate(items):
            if i in skip:
                continue
            try:
                _check_image(item)
                tensor = torch.from_numpy(np.ascontiguousarray(item)).to(
                    self._device, non_blocking=True
                )
                frames[i] = tensor.permute(2, 0, 1).flip(0)  # HWC BGR -> CHW RGB
            except Exception as error:
                errors[i] = error if isinstance(error, ValueError) else ValueError(str(error))

        if encoded:
            from torchvision.io import ImageReadMode, decode_jpeg

            data = [torch.frombuffer(bytearray(items[i]), dtype=torch.uint8) for i in encoded]
            try:
                decoded = decode_jpeg(data, device=self._device, mode=ImageReadMode.RGB)
                for i, frame in zip(encoded, decoded, strict=True):
                    frames[i] = frame
            except RuntimeError:
                # One bad file fails nvJPEG's whole batch. Retry one at a
                # time so it fails alone, falling back to OpenCV for files
                # nvJPEG cannot handle (CMYK, some progressive variants).
                for i, tensor in zip(encoded, data, strict=True):
                    try:
                        frames[i] = decode_jpeg(tensor, device=self._device, mode=ImageReadMode.RGB)
                    except RuntimeError:
                        try:
                            from .imaging import decode_image

                            bgr = decode_image(bytes(items[i]))
                            frames[i] = (
                                torch.from_numpy(bgr).to(self._device).permute(2, 0, 1).flip(0)
                            )
                        except ValueError as error:
                            errors[i] = error
        return frames, errors

    def _letterbox(self, frames: list[Any]):
        """Resize-and-pad into one (B, 3, S, S) batch, like Ultralytics' LetterBox."""
        torch = self._torch
        functional = torch.nn.functional
        size = self.image_size
        batch = torch.full(
            (len(frames), 3, size, size), self._pad, dtype=self.dtype, device=self._device
        )
        gains: list[float] = []
        pads: list[tuple[int, int]] = []
        for slot, frame in enumerate(frames):
            height, width = frame.shape[1:]
            gain = min(size / height, size / width)
            new_h, new_w = round(height * gain), round(width * gain)
            top = round((size - new_h) / 2 - 0.1)
            left = round((size - new_w) / 2 - 0.1)
            image = frame.unsqueeze(0).to(torch.float32)
            if (new_h, new_w) != (height, width):
                image = functional.interpolate(
                    image, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
            batch[slot, :, top : top + new_h, left : left + new_w] = (
                image[0].div_(255.0).to(self.dtype)
            )
            gains.append(gain)
            pads.append((left, top))
        return batch, np.asarray(gains, np.float32), np.asarray(pads, np.float32)

    def _postprocess(self, raw, gains, pads, shapes) -> list[np.ndarray]:
        """Per image: (N, 6) [x1, y1, x2, y2, conf, cls] in source pixels.

        The device work ends with one copy of a few kilobytes; thresholding
        and rescaling 300 rows per image is cheaper in NumPy than the kernel
        launches and synchronisations it would take on the GPU.
        """
        torch = self._torch
        if raw.ndim == 3 and raw.shape[-1] == 6:  # one-to-one head: final boxes
            host = raw.float().cpu().numpy()
            per_image = list(host)
        else:  # one-to-many head: (B, 4 + nc, anchors), NMS on the device
            kept = self._nms(raw.float())
            sizes = [len(k) for k in kept]
            flat = torch.cat(kept).cpu().numpy() if kept else np.empty((0, 6), np.float32)
            per_image = np.split(flat, np.cumsum(sizes)[:-1])

        classes = None if self.classes is None else self.classes.cpu().numpy()
        out = []
        for slot, det in enumerate(per_image):
            det = det[det[:, 4] >= self.confidence]
            if classes is not None:
                det = det[np.isin(det[:, 5].astype(int), classes)]
            det = det[: self.max_detections].copy()
            left, top = pads[slot]
            det[:, [0, 2]] = (det[:, [0, 2]] - left) / gains[slot]
            det[:, [1, 3]] = (det[:, [1, 3]] - top) / gains[slot]
            height, width = shapes[slot]
            det[:, [0, 2]] = det[:, [0, 2]].clip(0, int(width))
            det[:, [1, 3]] = det[:, [1, 3]].clip(0, int(height))
            out.append(det)
        return out

    def _nms(self, raw) -> list[Any]:
        import torchvision

        torch = self._torch
        preds = raw.transpose(1, 2)  # (B, anchors, 4 + nc)
        result = []
        for pred in preds:
            scores, classes = pred[:, 4:].max(dim=1)
            keep = scores >= self.confidence
            pred, scores, classes = pred[keep], scores[keep], classes[keep]
            xy, wh = pred[:, :2], pred[:, 2:4] / 2
            boxes = torch.cat((xy - wh, xy + wh), dim=1)
            kept = torchvision.ops.batched_nms(boxes, scores, classes, self.iou)
            kept = kept[: self.max_detections]
            result.append(
                torch.cat((boxes[kept], scores[kept, None], classes[kept, None].float()), dim=1)
            )
        return result

    def _to_detections(self, rows: np.ndarray) -> list[Detection]:
        names = self._names
        return [
            Detection(
                x1=float(r[0]),
                y1=float(r[1]),
                x2=float(r[2]),
                y2=float(r[3]),
                confidence=float(r[4]),
                class_id=int(r[5]),
                class_name=names.get(int(r[5]), str(int(r[5]))),
            )
            for r in rows
        ]


# ---------------------------------------------------------------- factory


def build_detector(settings: Any) -> UltralyticsDetector | GpuDetector:
    """Build the detector `settings` describes, falling back when `auto` must."""
    common = {
        "model": settings.model,
        "confidence": settings.confidence,
        "iou": settings.iou,
        "image_size": settings.image_size,
        "device": settings.device,
        "max_detections": settings.max_detections,
        "half": settings.half,
        "nms_free": settings.nms_free,
    }
    choice = settings.pipeline
    if choice in {"auto", "gpu"}:
        try:
            return GpuDetector(**common)
        except UnsupportedModelError as error:
            if choice == "gpu":
                raise
            logger.info("GPU pipeline unavailable (%s); using the Ultralytics predictor", error)
    return UltralyticsDetector(**common)
