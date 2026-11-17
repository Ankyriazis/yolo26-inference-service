"""Runtime configuration, read from `VISION_*` environment variables.

Every setting has a working default, so the service starts with no
configuration at all. Anything an operator might need to change without
rebuilding the image is exposed here rather than hard-coded at the call site.
"""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Pipeline = Literal["auto", "gpu", "ultralytics"]


class Settings(BaseSettings):
    """Service configuration.

    Attributes:
        model: weights to load. The suffix picks the runtime: `.pt` runs under
            PyTorch, `.onnx` under ONNX Runtime, `.engine` under TensorRT.
        device: `cuda:0`, `cpu`, `mps`, or None to detect automatically.
        pipeline: `gpu` keeps decode-to-detections on the device (CUDA only),
            `ultralytics` uses the stock predictor (any device, any format),
            `auto` picks `gpu` whenever it can.
        half: FP16 for `.pt` weights on CUDA. A TensorRT engine's precision is
            fixed when it is built, so this is ignored for `.engine` files.
        max_batch_size: upper bound for dynamic batching. Concurrent requests
            are coalesced into one forward pass of up to this many images.
        max_batch_wait_ms: how long the first request in a batch may wait for
            company. Trades a bounded amount of latency for throughput.
        max_queue_depth: requests allowed to wait for the GPU. Beyond this the
            service sheds load with a 503 instead of letting latency grow
            without bound.
        nms_free: use YOLO26's one-to-one head, which emits final boxes with
            no NMS step. False selects the one-to-many head plus NMS. Applies
            to `.pt` weights; an exported graph carries whichever head it was
            exported with.
        gpu_decode: decode JPEG uploads on the GPU with nvJPEG. Off by default:
            on GPUs without a hardware JPEG engine (GeForce, most workstation
            cards) nvJPEG competes with inference for the same SMs, and a
            many-core host decodes faster on its CPU threads. See README.
    """

    model_config = SettingsConfigDict(env_prefix="VISION_", frozen=True, extra="ignore")

    model: str = "yolo26n.pt"
    device: str | None = None
    pipeline: Pipeline = "auto"
    half: bool = False
    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 640
    max_detections: int = 300
    warmup_iterations: int = 3
    eager_load: bool = True
    max_batch_size: int = 8
    max_batch_wait_ms: float = 2.0
    max_queue_depth: int = 256
    max_upload_mb: int = 20
    nms_free: bool = True
    gpu_decode: bool = False

    @field_validator("device", mode="before")
    @classmethod
    def _blank_device_is_auto(cls, value: object) -> object:
        return value or None

    @model_validator(mode="after")
    def _check_ranges(self) -> Settings:
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("VISION_CONFIDENCE must be in (0, 1].")
        if not 0.0 < self.iou <= 1.0:
            raise ValueError("VISION_IOU must be in (0, 1].")
        if self.image_size < 32 or self.image_size % 32 != 0:
            raise ValueError("VISION_IMAGE_SIZE must be a multiple of 32, at least 32.")
        if not 1 <= self.max_batch_size <= 256:
            raise ValueError("VISION_MAX_BATCH_SIZE must be between 1 and 256.")
        if self.max_batch_wait_ms < 0:
            raise ValueError("VISION_MAX_BATCH_WAIT_MS cannot be negative.")
        if self.max_queue_depth < 1:
            raise ValueError("VISION_MAX_QUEUE_DEPTH must be at least 1.")
        if self.max_upload_mb < 1:
            raise ValueError("VISION_MAX_UPLOAD_MB must be at least 1.")
        return self
