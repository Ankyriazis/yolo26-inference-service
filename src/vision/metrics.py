"""Request metrics: Prometheus exposition plus a small JSON summary.

Two views of the same events:

- `/metrics` speaks the Prometheus text format: counters and histograms that a
  scraper aggregates across replicas and over time. This is what dashboards,
  alerts and autoscalers should read.
- `/stats` is a human-readable JSON summary over a bounded window of recent
  requests, with exact percentiles. Handy at a terminal; per-process by nature.

Every app gets its own `CollectorRegistry`, so several apps in one process
(the test suite builds many) never collide on metric names.
"""

from __future__ import annotations

import threading
from collections import deque

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily

from .gpu import GpuTelemetry

# Latency buckets in seconds, dense where a GPU detector actually lives.
_LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.25,
    0.5,
    1.0,
    2.5,
)
_BATCH_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)


class _GpuCollector:
    """Reads NVML at scrape time, so gauges are never stale."""

    def __init__(self, telemetry: GpuTelemetry) -> None:
        self.telemetry = telemetry

    def collect(self):
        sample = self.telemetry.sample()
        if sample is None:
            return
        gauges = {
            "vision_gpu_utilization_percent": ("GPU busy fraction", sample.utilization_pct),
            "vision_gpu_memory_used_bytes": (
                "GPU memory in use, all processes",
                sample.memory_used_mb * 2**20,
            ),
            "vision_gpu_temperature_celsius": ("GPU temperature", sample.temperature_c),
            "vision_gpu_sm_clock_hertz": (
                "SM clock",
                sample.sm_clock_mhz * 1e6 if sample.sm_clock_mhz else None,
            ),
            "vision_gpu_power_watts": ("Board power draw", sample.power_w),
        }
        for name, (doc, value) in gauges.items():
            if value is None:
                continue
            family = GaugeMetricFamily(name, doc, labels=["gpu"])
            family.add_metric([str(self.telemetry.index)], float(value))
            yield family


class Metrics:
    """Thread-safe request accounting, exported two ways."""

    def __init__(
        self,
        window: int = 2048,
        registry: CollectorRegistry | None = None,
        gpu: GpuTelemetry | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=window)
        self._batch_sizes: deque[int] = deque(maxlen=window)
        self._requests = 0
        self._errors = 0
        self._rejected = 0

        self.registry = registry or CollectorRegistry()
        self._requests_total = Counter(
            "vision_requests",
            "Inference requests by outcome",
            ["outcome"],
            registry=self.registry,
        )
        self._latency = Histogram(
            "vision_request_latency_seconds",
            "Server-side latency: decode, queue, inference, response",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self._queue_wait = Histogram(
            "vision_queue_wait_seconds",
            "Time a request waited for its batch to start",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self._batch_latency = Histogram(
            "vision_batch_inference_seconds",
            "Wall time of one batched forward pass",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self._batch_size = Histogram(
            "vision_batch_size",
            "Images per forward pass",
            buckets=_BATCH_BUCKETS,
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "vision_queue_depth", "Requests waiting for the GPU", registry=self.registry
        )
        self.model_info = Info("vision_model", "Loaded model", registry=self.registry)
        if gpu is not None and gpu.available:
            self.registry.register(_GpuCollector(gpu))

    # --------------------------------------------------------------- record

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._requests += 1
            self._latencies.append(latency_ms)
        self._requests_total.labels("ok").inc()
        self._latency.observe(latency_ms / 1000)

    def record_error(self) -> None:
        with self._lock:
            self._requests += 1
            self._errors += 1
        self._requests_total.labels("error").inc()

    def record_rejected(self) -> None:
        """A request shed because the queue was full."""
        with self._lock:
            self._requests += 1
            self._rejected += 1
        self._requests_total.labels("rejected").inc()

    def record_batch(self, size: int, inference_ms: float, queue_wait_ms: list[float]) -> None:
        with self._lock:
            self._batch_sizes.append(size)
        self._batch_size.observe(size)
        self._batch_latency.observe(inference_ms / 1000)
        for wait in queue_wait_ms:
            self._queue_wait.observe(wait / 1000)

    # --------------------------------------------------------------- export

    def prometheus(self) -> bytes:
        return generate_latest(self.registry)

    def snapshot(self) -> dict:
        with self._lock:
            latencies = sorted(self._latencies)
            batches = list(self._batch_sizes)
            requests, errors, rejected = self._requests, self._errors, self._rejected

        return {
            "requests_total": requests,
            "errors_total": errors,
            "rejected_total": rejected,
            "latency_ms": {
                "count": len(latencies),
                "p50": _percentile(latencies, 50),
                "p90": _percentile(latencies, 90),
                "p99": _percentile(latencies, 99),
                "max": round(latencies[-1], 2) if latencies else None,
            },
            "batching": {
                "batches": len(batches),
                "mean_size": round(sum(batches) / len(batches), 2) if batches else None,
                "max_size": max(batches) if batches else None,
            },
        }


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    """Nearest-rank percentile. Returns None for an empty window."""
    if not sorted_values:
        return None
    rank = max(1, round(percentile / 100 * len(sorted_values)))
    return round(sorted_values[rank - 1], 2)
