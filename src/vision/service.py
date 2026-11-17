"""HTTP service exposing YOLO26 object detection.

The request path, end to end:

    upload ─▶ size checks ─▶ decode ─▶ batching queue ─▶ inference thread ─▶ JSON
                              │                            (one per GPU)
                              ├─ JPEG + GPU pipeline: decoded on the GPU (nvJPEG)
                              └─ anything else: OpenCV, on a worker thread

The event loop never runs model code. Decoding happens on a thread pool and
inference on one dedicated thread fed by the dynamic batcher, so `/healthz`
answers in microseconds even while the GPU is saturated.

The model is loaded and warmed at startup, not on the first request, so the
first caller does not pay for CUDA initialisation. Liveness and readiness are
separate endpoints: the process answers /healthz well before the model has
finished loading, and an orchestrator needs to be able to tell those apart.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.concurrency import run_in_threadpool

from . import __version__
from .batching import BatcherClosedError, BatchReport, DynamicBatcher, QueueFullError
from .config import Settings
from .detector import InferenceResult, build_detector
from .gpu import GpuTelemetry
from .imaging import (
    MAX_PIXELS,
    ImageDecodeError,
    annotate,
    check_upload,
    decode_image,
    encode_jpeg,
    is_jpeg,
    jpeg_dimensions,
)
from .metrics import Metrics

logger = logging.getLogger(__name__)

DetectorFactory = Callable[[Settings], Any]


@dataclass
class Engine:
    """A loaded detector and the batcher that feeds it."""

    detector: Any
    batcher: DynamicBatcher
    settings: Settings
    started: float = field(default_factory=time.time)

    @property
    def gpu_decode(self) -> bool:
        return self.settings.gpu_decode and getattr(self.detector, "accepts_encoded", False)

    def describe(self) -> dict:
        d = self.detector
        return {
            "model": d.model_name,
            "backend": d.backend,
            "precision": d.precision,
            "pipeline": d.pipeline,
            "device": d.device,
            "gpu_decode": self.gpu_decode,
            "max_batch_size": self.batcher.max_batch_size,
            "max_batch_wait_ms": self.batcher.max_wait * 1000,
            "load_seconds": round(d.load_seconds, 2),
        }


def get_engine(request: Request) -> Engine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="The model is still loading. Retry shortly, or poll /readyz.",
        )
    return engine


def get_metrics(request: Request) -> Metrics:
    return request.app.state.metrics


EngineDep = Annotated[Engine, Depends(get_engine)]
MetricsDep = Annotated[Metrics, Depends(get_metrics)]
ImageUpload = Annotated[UploadFile, File(description="JPEG, PNG, BMP or WebP image")]


def create_app(
    settings: Settings | None = None,
    detector_factory: DetectorFactory | None = None,
) -> FastAPI:
    """Build the application.

    Args:
        settings: configuration; read from the environment when omitted.
        detector_factory: builds the detector at startup. Tests pass a stub
            here so the whole request path, batcher included, runs without
            weights or a GPU.
    """
    settings = settings or Settings()
    gpu = GpuTelemetry()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started = time.time()
        app.state.engine = None
        metrics: Metrics = app.state.metrics

        factory = detector_factory or (build_detector if settings.eager_load else None)
        if factory is None:
            logger.warning("VISION_EAGER_LOAD is off; the model will not be loaded")
            yield
            return

        # Loading and warming take seconds; keep them off the event loop so
        # /healthz answers throughout.
        detector = await run_in_threadpool(factory, settings)
        batch_limit = min(settings.max_batch_size, getattr(detector, "max_batch_size", 1))
        if batch_limit < settings.max_batch_size:
            logger.warning(
                "VISION_MAX_BATCH_SIZE=%d exceeds what %s supports; using %d",
                settings.max_batch_size,
                detector.model_name,
                batch_limit,
            )
        await run_in_threadpool(
            detector.warmup, settings.warmup_iterations, sorted({1, batch_limit})
        )
        # The batcher emits every size from 1 to the limit, and the first
        # batch at a new shape pays for allocation and kernel selection. Take
        # each one once now, so no caller ever pays it.
        if batch_limit > 2:
            await run_in_threadpool(detector.warmup, 1, range(2, batch_limit))

        def on_batch(report: BatchReport) -> None:
            metrics.record_batch(report.size, report.inference_ms, report.queue_wait_ms)

        batcher = DynamicBatcher(
            detector.predict_batch,
            max_batch_size=batch_limit,
            max_wait_ms=settings.max_batch_wait_ms,
            max_queue=settings.max_queue_depth,
            on_batch=on_batch,
        )
        await batcher.start()
        engine = Engine(detector, batcher, settings)
        metrics.model_info.info({k: str(v) for k, v in engine.describe().items()})
        app.state.engine = engine
        logger.info("ready: %s", engine.describe())
        try:
            yield
        finally:
            app.state.engine = None
            await batcher.stop()

    app = FastAPI(
        title="YOLO26 Inference Service",
        description=(
            "Object detection over HTTP: YOLO26 on TensorRT, ONNX Runtime or PyTorch, "
            "with dynamic batching and GPU telemetry."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.metrics = Metrics(gpu=gpu)
    app.state.gpu = gpu
    app.state.started = time.time()
    app.state.engine = None
    max_bytes = settings.max_upload_mb * 1024 * 1024

    @app.middleware("http")
    async def refuse_oversized_bodies(request: Request, call_next):
        # Refuse on the header, before the body is read into memory at all.
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes + 64 * 1024:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Upload exceeds the {settings.max_upload_mb} MB limit."},
            )
        return await call_next(request)

    # ------------------------------------------------------------ operations

    @app.get("/healthz", tags=["operations"])
    def healthz() -> dict:
        """Liveness. Answers as soon as the process is up."""
        return {"status": "alive", "uptime_seconds": round(time.time() - app.state.started, 1)}

    @app.get("/readyz", tags=["operations"])
    def readyz(request: Request, response: Response) -> dict:
        """Readiness. Only true once the model is loaded and warmed."""
        engine: Engine | None = request.app.state.engine
        if engine is None:
            response.status_code = 503
            return {"status": "loading", "ready": False}
        return {"status": "ready", "ready": True, **engine.describe()}

    @app.get("/metrics", tags=["operations"], response_class=Response)
    def metrics(collector: MetricsDep) -> Response:
        """Prometheus exposition: request, batching and GPU metrics."""
        return Response(collector.prometheus(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/stats", tags=["operations"])
    def stats(request: Request, collector: MetricsDep) -> dict:
        """Recent-window latency percentiles and batching behaviour, as JSON."""
        snapshot = collector.snapshot()
        engine: Engine | None = request.app.state.engine
        snapshot["queue_depth"] = engine.batcher.queue_depth if engine else 0
        return snapshot

    @app.get("/gpu", tags=["operations"])
    def gpu_state() -> dict:
        """Live GPU state from NVML: utilisation, memory, clocks, PCIe link."""
        sample = gpu.sample()
        if sample is None:
            return {"available": False}
        return {"available": True, "driver": gpu.driver_version(), **sample.to_dict()}

    # ------------------------------------------------------------ model

    @app.get("/classes", tags=["model"])
    def classes(engine: EngineDep) -> dict:
        """The classes this model can detect."""
        names = engine.detector.class_names
        return {"count": len(names), "classes": names}

    # ------------------------------------------------------------ inference

    async def run(engine: Engine, collector: Metrics, item: Any) -> InferenceResult:
        collector.queue_depth.set(engine.batcher.queue_depth)
        try:
            return await engine.batcher.submit(item)
        except QueueFullError as error:
            collector.record_rejected()
            raise HTTPException(
                status_code=503, detail=f"Overloaded: {error}.", headers={"Retry-After": "1"}
            ) from error
        except BatcherClosedError as error:
            raise HTTPException(status_code=503, detail="Shutting down.") from error
        except ValueError as error:  # ImageDecodeError included
            collector.record_error()
            raise HTTPException(status_code=400, detail=str(error)) from error

    async def prepare(engine: Engine, payload: bytes) -> Any:
        """Decide where this upload is decoded, and validate what we can cheaply."""
        try:
            check_upload(payload, max_bytes)
            if engine.gpu_decode and is_jpeg(payload):
                dims = jpeg_dimensions(payload)
                if dims is not None and dims[0] * dims[1] <= MAX_PIXELS:
                    return payload  # decoded on the GPU, inside the batch
            return await run_in_threadpool(decode_image, payload, max_bytes)
        except ImageDecodeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    def respond(result: InferenceResult, engine: Engine, started: float) -> dict:
        body = result.to_dict()
        body["server_ms"] = round((time.perf_counter() - started) * 1000, 2)
        body["model"] = engine.detector.model_name
        body["backend"] = engine.detector.backend
        body["precision"] = engine.detector.precision
        body["device"] = engine.detector.device
        return body

    @app.post("/detect", tags=["inference"])
    async def detect(file: ImageUpload, engine: EngineDep, collector: MetricsDep) -> dict:
        """Detect objects in an uploaded image and return them as JSON."""
        started = time.perf_counter()
        item = await prepare(engine, await file.read())
        result = await run(engine, collector, item)
        collector.record((time.perf_counter() - started) * 1000)
        return respond(result, engine, started)

    @app.post(
        "/detect/raw",
        tags=["inference"],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}},
            }
        },
    )
    async def detect_raw(request: Request, engine: EngineDep, collector: MetricsDep) -> dict:
        """The image as the raw request body, e.g. `Content-Type: image/jpeg`.

        Same answer as /detect without multipart framing to parse. Once the
        model runs on TensorRT, the Python HTTP layer is the bottleneck, and
        this is the cheaper of the two ways in. See README, finding 2.
        """
        started = time.perf_counter()
        item = await prepare(engine, await request.body())
        result = await run(engine, collector, item)
        collector.record((time.perf_counter() - started) * 1000)
        return respond(result, engine, started)

    @app.post("/detect/batch", tags=["inference"])
    async def detect_batch(
        files: Annotated[list[UploadFile], File(description="Up to 64 images")],
        engine: EngineDep,
        collector: MetricsDep,
    ) -> dict:
        """Several images in one request. Each joins the batching queue on its own."""
        if len(files) > 64:
            raise HTTPException(status_code=400, detail="At most 64 images per request.")
        started = time.perf_counter()
        payloads = [await f.read() for f in files]
        items: Sequence[Any] = await asyncio.gather(*(prepare(engine, p) for p in payloads))
        results = await asyncio.gather(*(run(engine, collector, i) for i in items))
        elapsed = (time.perf_counter() - started) * 1000
        for _ in results:
            collector.record(elapsed)
        return {
            "results": [respond(r, engine, started) for r in results],
            "count": len(results),
            "server_ms": round(elapsed, 2),
        }

    @app.post("/detect/annotated", tags=["inference"])
    async def detect_annotated(
        file: ImageUpload,
        engine: EngineDep,
        collector: MetricsDep,
        confidence_labels: Annotated[
            bool, Query(description="draw confidence beside labels")
        ] = True,
    ) -> Response:
        """Detect objects and return the image with boxes drawn on it."""
        started = time.perf_counter()
        payload = await file.read()
        try:
            image = await run_in_threadpool(decode_image, payload, max_bytes)
        except ImageDecodeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        result = await run(engine, collector, image)
        collector.record((time.perf_counter() - started) * 1000)

        jpeg = await run_in_threadpool(
            lambda: encode_jpeg(annotate(image, result.detections, confidence_labels))
        )
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={
                "X-Detection-Count": str(len(result.detections)),
                "X-Inference-Ms": f"{result.inference_ms:.2f}",
                "X-Batch-Size": str(result.batch_size),
            },
        )

    return app


app = create_app()
