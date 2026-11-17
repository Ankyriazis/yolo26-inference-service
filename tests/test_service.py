"""Tests for the inference service.

None of these download weights, load a model, or need a GPU. The detector is
injected through `create_app(detector_factory=...)`, which is the reason the
HTTP layer was kept ignorant of Ultralytics: request handling, validation,
batching, load shedding and error paths are all verifiable on their own -
including the real `DynamicBatcher`, which runs here exactly as in production.
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from typing import ClassVar

import cv2
import httpx2 as httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vision.batching import BatcherClosedError, DynamicBatcher, QueueFullError
from vision.config import Settings
from vision.detector import Detection, InferenceResult, inspect_weights
from vision.imaging import (
    ImageDecodeError,
    annotate,
    decode_image,
    encode_jpeg,
    is_jpeg,
    jpeg_dimensions,
)
from vision.metrics import Metrics
from vision.service import create_app


class StubDetector:
    """Stands in for a real model: same surface, no weights."""

    model_name = "stub"
    backend = "stub"
    precision = "fp32"
    pipeline = "stub"
    device = "cpu"
    load_seconds = 0.0
    accepts_encoded = False
    class_names: ClassVar[dict[int, str]] = {0: "person", 2: "car"}

    def __init__(self, detections=None, delay_s: float = 0.0, max_batch_size: int = 8):
        self.detections = (
            detections
            if detections is not None
            else [Detection(10.0, 20.0, 110.0, 220.0, 0.91, 0, "person")]
        )
        self.delay_s = delay_s
        self.max_batch_size = max_batch_size
        self.batches: list[int] = []
        self.threads: set[str] = set()
        self.warmed = False

    def warmup(self, iterations=1, batch_sizes=(1,)):
        self.warmed = True

    def predict_batch(self, images):
        self.batches.append(len(images))
        self.threads.add(threading.current_thread().name)
        if self.delay_s:
            time.sleep(self.delay_s)
        out = []
        for image in images:
            if isinstance(image, np.ndarray) and image.shape[0] == 13:  # "poisoned" input
                out.append(ValueError("this one image is bad"))
                continue
            height, width = image.shape[:2]
            out.append(InferenceResult(self.detections, 4.2, width, height, len(images)))
        return out

    @property
    def calls(self):
        return sum(self.batches)


def png_bytes(width=160, height=120, colour=(30, 90, 200)) -> bytes:
    image = np.full((height, width, 3), colour, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def make_app(stub=None, **settings):
    stub = stub or StubDetector()
    settings.setdefault("warmup_iterations", 1)
    app = create_app(Settings(**settings), detector_factory=lambda _: stub)
    return app, stub


@pytest.fixture
def client():
    """A client whose model is stubbed, so no weights are ever loaded."""
    app, stub = make_app()
    with TestClient(app) as test_client:
        test_client.stub = stub
        yield test_client


def upload(data=None, name="x.png", kind="image/png"):
    return {"file": (name, data if data is not None else png_bytes(), kind)}


# ---------------------------------------------------------------- imaging


def test_decodes_a_png():
    image = decode_image(png_bytes())
    assert image.shape == (120, 160, 3)


def test_rejects_empty_upload():
    with pytest.raises(ImageDecodeError):
        decode_image(b"")


def test_rejects_bytes_that_are_not_an_image():
    with pytest.raises(ImageDecodeError):
        decode_image(b"this is definitely not a picture")


def test_rejects_an_oversized_upload():
    with pytest.raises(ImageDecodeError, match="larger than"):
        decode_image(b"\x00" * (21 * 1024 * 1024))


def test_jpeg_round_trip():
    image = decode_image(encode_jpeg(np.full((40, 60, 3), 128, np.uint8)))
    assert image.shape == (40, 60, 3)


def test_jpeg_dimensions_read_from_the_header():
    jpeg = encode_jpeg(np.zeros((37, 91, 3), np.uint8))
    assert is_jpeg(jpeg)
    assert jpeg_dimensions(jpeg) == (91, 37)
    assert jpeg_dimensions(png_bytes()) is None


def test_decompression_bomb_is_refused_from_its_header():
    """A tiny file that claims 40,000 x 40,000 pixels never reaches the decoder."""
    sof = b"\xff\xc0" + struct.pack(">HBHH", 17, 8, 40_000, 40_000) + b"\x03" + b"\x00" * 9
    bomb = b"\xff\xd8" + sof + b"\xff\xd9"
    assert jpeg_dimensions(bomb) == (40_000, 40_000)
    with pytest.raises(ImageDecodeError, match="megapixels"):
        decode_image(bomb)


def test_annotate_does_not_modify_the_original():
    image = np.zeros((100, 100, 3), np.uint8)
    before = image.copy()
    annotate(image, [Detection(5, 5, 50, 50, 0.9, 0, "person")])
    assert np.array_equal(image, before)


def test_annotate_handles_a_box_against_the_top_edge():
    """The label must stay inside the frame rather than being drawn off it."""
    image = np.zeros((100, 100, 3), np.uint8)
    out = annotate(image, [Detection(0, 0, 60, 40, 0.8, 0, "person")])
    assert out.shape == image.shape
    assert out.any(), "nothing was drawn"


# ---------------------------------------------------------------- weights


def test_engine_metadata_is_read_without_loading_the_engine(tmp_path):
    meta = {"batch": 32, "args": {"quantize": 8, "dynamic": True, "nms": False}, "end2end": True}
    header = json.dumps(meta).encode()
    path = tmp_path / "m.engine"
    path.write_bytes(len(header).to_bytes(4, "little") + header + b"\x00engine-bytes")
    info = inspect_weights(str(path))
    assert info.backend == "tensorrt"
    assert info.precision == "int8"
    assert info.max_batch == 32
    assert info.dynamic is True
    assert info.end2end is True


def test_weights_suffix_selects_the_runtime():
    assert inspect_weights("missing.onnx").backend == "onnxruntime"
    assert inspect_weights("missing.pt").backend == "pytorch"
    assert inspect_weights("missing.pt", half=True).precision == "fp16"
    assert inspect_weights("missing.pt").max_batch is None


# ---------------------------------------------------------------- metrics


def test_metrics_start_empty():
    snapshot = Metrics().snapshot()
    assert snapshot["requests_total"] == 0
    assert snapshot["latency_ms"]["p50"] is None


def test_metrics_percentiles():
    collector = Metrics()
    for value in range(1, 101):
        collector.record(float(value))
    snapshot = collector.snapshot()
    assert snapshot["requests_total"] == 100
    assert snapshot["latency_ms"]["p50"] == 50.0
    assert snapshot["latency_ms"]["p99"] == 99.0
    assert snapshot["latency_ms"]["max"] == 100.0


def test_metrics_window_is_bounded():
    collector = Metrics(window=10)
    for value in range(100):
        collector.record(float(value))
    assert collector.snapshot()["latency_ms"]["count"] == 10
    assert collector.snapshot()["requests_total"] == 100


def test_metrics_count_errors_and_rejections_as_requests():
    collector = Metrics()
    collector.record_error()
    collector.record_rejected()
    snapshot = collector.snapshot()
    assert snapshot["errors_total"] == 1
    assert snapshot["rejected_total"] == 1
    assert snapshot["requests_total"] == 2


def test_metrics_export_prometheus_text():
    collector = Metrics()
    collector.record(12.0)
    collector.record_batch(4, 8.0, [0.5, 1.0, 1.5, 2.0])
    text = collector.prometheus().decode()
    assert 'vision_requests_total{outcome="ok"} 1.0' in text
    assert "vision_batch_size_bucket" in text
    assert "vision_queue_wait_seconds_count 4.0" in text
    assert collector.snapshot()["batching"]["mean_size"] == 4


def test_two_metrics_instances_do_not_collide():
    Metrics().record(1.0)
    Metrics().record(1.0)  # would raise "Duplicated timeseries" on a shared registry


# ---------------------------------------------------------------- config


def test_settings_reject_an_out_of_range_confidence():
    with pytest.raises(ValueError, match="CONFIDENCE"):
        Settings(confidence=1.5)


def test_settings_reject_a_bad_image_size():
    with pytest.raises(ValueError, match="IMAGE_SIZE"):
        Settings(image_size=100)  # not a multiple of 32


def test_settings_reject_a_zero_batch():
    with pytest.raises(ValueError, match="MAX_BATCH_SIZE"):
        Settings(max_batch_size=0)


def test_settings_read_the_environment(monkeypatch):
    monkeypatch.setenv("VISION_MAX_BATCH_SIZE", "16")
    monkeypatch.setenv("VISION_PIPELINE", "ultralytics")
    monkeypatch.setenv("VISION_DEVICE", "")
    settings = Settings()
    assert settings.max_batch_size == 16
    assert settings.pipeline == "ultralytics"
    assert settings.device is None


# ---------------------------------------------------------------- batcher


def run(coro):
    return asyncio.run(coro)


def test_batcher_coalesces_concurrent_requests():
    seen: list[int] = []

    def infer(items):
        seen.append(len(items))
        time.sleep(0.02)
        return [i * 10 for i in items]

    async def scenario():
        batcher = DynamicBatcher(infer, max_batch_size=4, max_wait_ms=20)
        await batcher.start()
        results = await asyncio.gather(*(batcher.submit(i) for i in range(10)))
        await batcher.stop()
        return results

    assert run(scenario()) == [i * 10 for i in range(10)]
    assert max(seen) == 4, "never filled a batch"
    assert sum(seen) == 10
    assert len(seen) < 10, "every request ran alone"


def test_batcher_with_zero_wait_still_batches_under_load():
    """Requests that queue while the GPU is busy form the next batch for free."""
    seen: list[int] = []

    def infer(items):
        seen.append(len(items))
        time.sleep(0.02)
        return list(items)

    async def scenario():
        batcher = DynamicBatcher(infer, max_batch_size=8, max_wait_ms=0)
        await batcher.start()
        await asyncio.gather(*(batcher.submit(i) for i in range(12)))
        await batcher.stop()

    run(scenario())
    assert max(seen) > 1


def test_batcher_isolates_a_failing_item():
    def infer(items):
        return [ValueError("bad") if i == 2 else i for i in items]

    async def scenario():
        batcher = DynamicBatcher(infer, max_batch_size=8, max_wait_ms=5)
        await batcher.start()
        out = await asyncio.gather(*(batcher.submit(i) for i in range(4)), return_exceptions=True)
        await batcher.stop()
        return out

    out = run(scenario())
    assert out[0] == 0 and out[1] == 1 and out[3] == 3
    assert isinstance(out[2], ValueError)


def test_batcher_fails_the_whole_batch_when_inference_raises():
    def infer(items):
        raise RuntimeError("CUDA error: device-side assert")

    async def scenario():
        batcher = DynamicBatcher(infer, max_batch_size=8, max_wait_ms=5)
        await batcher.start()
        out = await asyncio.gather(*(batcher.submit(i) for i in range(3)), return_exceptions=True)
        await batcher.stop()
        return out

    assert all(isinstance(e, RuntimeError) for e in run(scenario()))


def test_batcher_sheds_load_when_the_queue_is_full():
    gate = threading.Event()

    def infer(items):
        gate.wait(2)
        return list(items)

    async def scenario():
        batcher = DynamicBatcher(infer, max_batch_size=1, max_wait_ms=0, max_queue=2)
        await batcher.start()
        first = asyncio.create_task(batcher.submit(0))  # occupies the GPU
        await asyncio.sleep(0.05)
        queued = [asyncio.create_task(batcher.submit(i)) for i in (1, 2)]
        await asyncio.sleep(0.01)
        with pytest.raises(QueueFullError):
            await batcher.submit(3)
        gate.set()
        await asyncio.gather(first, *queued)
        await batcher.stop()

    run(scenario())


def test_batcher_refuses_work_after_stop():
    async def scenario():
        batcher = DynamicBatcher(lambda items: list(items))
        await batcher.start()
        await batcher.stop()
        with pytest.raises(BatcherClosedError):
            await batcher.submit(1)

    run(scenario())


# ---------------------------------------------------------------- http


def test_healthz_is_alive(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readyz_describes_the_engine(client):
    body = client.get("/readyz").json()
    assert body["ready"] is True
    assert body["backend"] == "stub"
    assert body["max_batch_size"] == 8
    assert client.stub.warmed


def test_detect_returns_json_detections(client):
    response = client.post("/detect", files=upload())
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["detections"][0]["class_name"] == "person"
    assert body["image"] == {"width": 160, "height": 120}
    assert body["batch_size"] == 1
    assert body["server_ms"] >= 0
    assert client.stub.calls == 1


def test_inference_runs_on_the_dedicated_thread(client):
    client.post("/detect", files=upload())
    assert client.stub.threads == {"inference_0"}


def test_raw_body_endpoint_matches_multipart(client):
    raw = client.post("/detect/raw", content=png_bytes(), headers={"content-type": "image/png"})
    multipart = client.post("/detect", files=upload())
    assert raw.status_code == 200
    assert raw.json()["detections"] == multipart.json()["detections"]
    assert client.post("/detect/raw", content=b"").status_code == 400


def test_detect_rejects_a_non_image(client):
    response = client.post("/detect", files=upload(b"nope", "x.txt", "text/plain"))
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_detect_requires_a_file(client):
    assert client.post("/detect").status_code == 422


def test_a_bad_image_in_a_batch_fails_alone(client):
    good, bad = png_bytes(), png_bytes(height=13)
    response = client.post(
        "/detect/batch",
        files=[("files", ("a.png", good, "image/png")), ("files", ("b.png", bad, "image/png"))],
    )
    assert response.status_code == 400  # the batch endpoint is all-or-nothing for the caller
    assert client.post("/detect", files=upload(good)).status_code == 200


def test_batch_endpoint(client):
    files = [("files", (f"{i}.png", png_bytes(), "image/png")) for i in range(5)]
    body = client.post("/detect/batch", files=files).json()
    assert body["count"] == 5
    assert all(r["count"] == 1 for r in body["results"])


def test_annotated_endpoint_returns_a_jpeg(client):
    response = client.post("/detect/annotated", files=upload())
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["X-Detection-Count"] == "1"
    assert decode_image(response.content).shape == (120, 160, 3)


def test_classes_endpoint(client):
    body = client.get("/classes").json()
    assert body["count"] == 2
    assert body["classes"]["0"] == "person"


def test_stats_track_requests(client):
    client.post("/detect", files=upload())
    client.post("/detect", files=upload())
    stats = client.get("/stats").json()
    assert stats["latency_ms"]["count"] == 2
    assert stats["latency_ms"]["p50"] is not None
    assert stats["batching"]["batches"] >= 1


def test_metrics_endpoint_is_prometheus(client):
    client.post("/detect", files=upload())
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "vision_request_latency_seconds_bucket" in response.text
    assert 'vision_model_info{backend="stub"' in response.text


def test_gpu_endpoint_answers_with_or_without_a_gpu(client):
    assert "available" in client.get("/gpu").json()


def test_oversized_body_is_refused_before_it_is_read(client):
    response = client.post(
        "/detect",
        content=b"x",
        headers={"content-length": str(50 * 1024 * 1024), "content-type": "multipart/form-data"},
    )
    assert response.status_code == 413


def test_readyz_is_503_while_the_model_is_absent():
    """Liveness and readiness must not answer the same way during startup."""
    app = create_app(Settings(eager_load=False))
    with TestClient(app) as bare:
        assert bare.get("/healthz").status_code == 200
        assert bare.get("/readyz").status_code == 503
        assert bare.get("/readyz").json()["ready"] is False


def test_detect_is_503_before_the_model_is_loaded():
    app = create_app(Settings(eager_load=False))
    with TestClient(app) as bare:
        response = bare.post("/detect", files=upload())
        assert response.status_code == 503


def test_batch_limit_is_clamped_to_what_the_model_supports():
    app, _ = make_app(StubDetector(max_batch_size=4), max_batch_size=32)
    with TestClient(app) as client:
        assert client.get("/readyz").json()["max_batch_size"] == 4


# ---------------------------------------------------------------- concurrency


async def _serve(app, scenario):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            return await scenario(http)


def test_concurrent_requests_share_forward_passes():
    """Eight simultaneous uploads should not cost eight forward passes."""
    app, stub = make_app(StubDetector(delay_s=0.05), max_batch_size=8, max_batch_wait_ms=20)

    async def scenario(http):
        started = time.perf_counter()
        responses = await asyncio.gather(*(http.post("/detect", files=upload()) for _ in range(8)))
        return responses, time.perf_counter() - started

    responses, elapsed = asyncio.run(_serve(app, scenario))
    assert all(r.status_code == 200 for r in responses)
    assert max(stub.batches) > 1
    assert elapsed < 8 * 0.05, "requests were served one forward pass at a time"


def test_health_stays_responsive_while_the_gpu_is_busy():
    """The event loop must never block on inference."""
    app, _ = make_app(StubDetector(delay_s=0.3), max_batch_size=1)

    async def scenario(http):
        busy = asyncio.create_task(http.post("/detect", files=upload()))
        await asyncio.sleep(0.05)
        started = time.perf_counter()
        health = await http.get("/healthz")
        health_ms = (time.perf_counter() - started) * 1000
        await busy
        return health, health_ms

    health, health_ms = asyncio.run(_serve(app, scenario))
    assert health.status_code == 200
    assert health_ms < 100, f"/healthz took {health_ms:.0f} ms behind a 300 ms inference"


def test_overload_returns_503_with_retry_after():
    app, _ = make_app(StubDetector(delay_s=0.2), max_batch_size=1, max_queue_depth=1)

    async def scenario(http):
        return await asyncio.gather(*(http.post("/detect", files=upload()) for _ in range(6)))

    responses = asyncio.run(_serve(app, scenario))
    codes = sorted(r.status_code for r in responses)
    assert 200 in codes and 503 in codes
    shed = next(r for r in responses if r.status_code == 503)
    assert shed.headers["retry-after"] == "1"
