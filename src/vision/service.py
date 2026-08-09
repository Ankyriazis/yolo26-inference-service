"""HTTP service exposing YOLO26 object detection.

The model is loaded and warmed at startup, not on the first request, so the
first caller does not pay for CUDA initialisation. Liveness and readiness are
separate endpoints: the process answers /healthz well before the model has
finished loading, and an orchestrator needs to be able to tell those apart.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Query, Response, UploadFile

from .config import Settings
from .detector import Detector
from .imaging import ImageDecodeError, annotate, decode_image, encode_jpeg
from .metrics import Metrics

logger = logging.getLogger(__name__)

# Process-wide state. A single model is shared across requests; Ultralytics
# inference is GPU-bound and does not benefit from several copies competing
# for the same device.
_state: dict = {"detector": None, "settings": None, "metrics": None, "started": 0.0}


def get_detector() -> Detector:
    detector = _state.get("detector")
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="The model is still loading. Retry shortly, or poll /readyz.",
        )
    return detector


def get_metrics() -> Metrics:
    return _state["metrics"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = _state.get("settings") or Settings()
    _state["settings"] = settings
    _state["metrics"] = Metrics()
    _state["started"] = time.time()

    if settings.eager_load:
        detector = Detector(
            model=settings.model,
            confidence=settings.confidence,
            iou=settings.iou,
            image_size=settings.image_size,
            device=settings.device,
            max_detections=settings.max_detections,
        )
        detector.warmup(settings.warmup_iterations)
        _state["detector"] = detector
    else:
        logger.warning("VISION_EAGER_LOAD is off; the model will not be loaded")

    yield

    _state["detector"] = None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts settings so tests can override them."""
    if settings is not None:
        _state["settings"] = settings

    app = FastAPI(
        title="YOLO26 Inference Service",
        description="Object detection over HTTP, using YOLO26 and OpenCV.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/healthz", tags=["operations"])
    def healthz() -> dict:
        """Liveness. Answers as soon as the process is up."""
        return {"status": "alive", "uptime_seconds": round(time.time() - _state["started"], 1)}

    @app.get("/readyz", tags=["operations"])
    def readyz(response: Response) -> dict:
        """Readiness. Only true once the model is loaded and warmed."""
        detector = _state.get("detector")
        if detector is None:
            response.status_code = 503
            return {"status": "loading", "ready": False}
        return {
            "status": "ready",
            "ready": True,
            "model": detector.model_name,
            "device": detector.device,
            "load_seconds": round(detector.load_seconds, 2),
        }

    @app.get("/metrics", tags=["operations"])
    def metrics(collector: Metrics = Depends(get_metrics)) -> dict:
        """Request counts and latency percentiles since start."""
        return collector.snapshot()

    @app.get("/classes", tags=["model"])
    def classes(detector: Detector = Depends(get_detector)) -> dict:
        """The classes this model can detect."""
        return {"count": len(detector.class_names), "classes": detector.class_names}

    @app.post("/detect", tags=["inference"])
    async def detect(
        file: UploadFile = File(..., description="JPEG, PNG, BMP or WebP image"),
        detector: Detector = Depends(get_detector),
        collector: Metrics = Depends(get_metrics),
    ) -> dict:
        """Detect objects in an uploaded image and return them as JSON."""
        image = _decode(await file.read())
        started = time.perf_counter()
        try:
            result = detector.predict(image)
        except ValueError as error:
            collector.record_error()
            raise HTTPException(status_code=400, detail=str(error)) from error
        collector.record(( time.perf_counter() - started) * 1000)

        payload = result.to_dict()
        payload["model"] = detector.model_name
        payload["device"] = detector.device
        return payload

    @app.post("/detect/annotated", tags=["inference"])
    async def detect_annotated(
        file: UploadFile = File(..., description="JPEG, PNG, BMP or WebP image"),
        confidence_labels: bool = Query(True, description="draw confidence beside labels"),
        detector: Detector = Depends(get_detector),
        collector: Metrics = Depends(get_metrics),
    ) -> Response:
        """Detect objects and return the image with boxes drawn on it."""
        image = _decode(await file.read())
        started = time.perf_counter()
        try:
            result = detector.predict(image)
        except ValueError as error:
            collector.record_error()
            raise HTTPException(status_code=400, detail=str(error)) from error
        collector.record((time.perf_counter() - started) * 1000)

        annotated = annotate(image, result.detections, show_confidence=confidence_labels)
        return Response(
            content=encode_jpeg(annotated),
            media_type="image/jpeg",
            headers={
                "X-Detection-Count": str(len(result.detections)),
                "X-Inference-Ms": f"{result.inference_ms:.2f}",
            },
        )

    return app


def _decode(payload: bytes):
    try:
        return decode_image(payload)
    except ImageDecodeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


app = create_app()
