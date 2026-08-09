"""Tests for the inference service.

None of these download weights, load a model, or need a GPU. The detector is
injected through FastAPI's dependency system, which is the reason the HTTP
layer was kept ignorant of Ultralytics: the request handling, validation and
error paths are all verifiable on their own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision.config import Settings  # noqa: E402
from vision.detector import Detection, InferenceResult  # noqa: E402
from vision.imaging import (  # noqa: E402
    ImageDecodeError,
    annotate,
    decode_image,
    encode_jpeg,
)
from vision.metrics import Metrics  # noqa: E402
from vision.service import create_app, get_detector  # noqa: E402


class StubDetector:
    """Stands in for a real model: same surface, no weights."""

    model_name = "stub"
    device = "cpu"
    load_seconds = 0.0
    class_names = {0: "person", 2: "car"}

    def __init__(self, detections=None):
        self.detections = detections if detections is not None else [
            Detection(10.0, 20.0, 110.0, 220.0, 0.91, 0, "person")
        ]
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        height, width = image.shape[:2]
        return InferenceResult(
            detections=self.detections,
            inference_ms=4.2,
            image_width=width,
            image_height=height,
        )


def png_bytes(width=160, height=120, colour=(30, 90, 200)) -> bytes:
    image = np.full((height, width, 3), colour, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


@pytest.fixture
def client():
    """A client whose model is stubbed, so no weights are ever loaded."""
    app = create_app(Settings(eager_load=False))
    stub = StubDetector()
    app.dependency_overrides[get_detector] = lambda: stub
    with TestClient(app) as test_client:
        test_client.stub = stub
        yield test_client


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


def test_metrics_count_errors_as_requests():
    collector = Metrics()
    collector.record_error()
    snapshot = collector.snapshot()
    assert snapshot["errors_total"] == 1
    assert snapshot["requests_total"] == 1


# ---------------------------------------------------------------- config

def test_settings_reject_an_out_of_range_confidence():
    with pytest.raises(ValueError, match="CONFIDENCE"):
        Settings(confidence=1.5)


def test_settings_reject_a_bad_image_size():
    with pytest.raises(ValueError, match="IMAGE_SIZE"):
        Settings(image_size=100)   # not a multiple of 32


# ---------------------------------------------------------------- http

def test_healthz_is_alive(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_detect_returns_json_detections(client):
    response = client.post("/detect", files={"file": ("x.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["detections"][0]["class_name"] == "person"
    assert body["image"] == {"width": 160, "height": 120}
    assert client.stub.calls == 1


def test_detect_rejects_a_non_image(client):
    response = client.post("/detect", files={"file": ("x.txt", b"nope", "text/plain")})
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_detect_requires_a_file(client):
    assert client.post("/detect").status_code == 422


def test_annotated_endpoint_returns_a_jpeg(client):
    response = client.post(
        "/detect/annotated", files={"file": ("x.png", png_bytes(), "image/png")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["X-Detection-Count"] == "1"
    assert decode_image(response.content).shape == (120, 160, 3)


def test_classes_endpoint(client):
    body = client.get("/classes").json()
    assert body["count"] == 2
    assert body["classes"]["0"] == "person"


def test_metrics_track_requests(client):
    client.post("/detect", files={"file": ("x.png", png_bytes(), "image/png")})
    client.post("/detect", files={"file": ("x.png", png_bytes(), "image/png")})
    latency = client.get("/metrics").json()["latency_ms"]
    assert latency["count"] == 2
    assert latency["p50"] is not None


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
        response = bare.post("/detect", files={"file": ("x.png", png_bytes(), "image/png")})
        assert response.status_code == 503
