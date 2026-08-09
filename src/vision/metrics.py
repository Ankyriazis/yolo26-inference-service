"""In-process request metrics.

Deliberately small: counts and latency percentiles held in a bounded window,
with no external dependency. Percentiles are what matter for an inference
service - a mean hides the tail, and the tail is what a caller notices.

This is per-process state. Behind several replicas each reports its own view,
which is the point at which a real metrics backend earns its place.
"""

from __future__ import annotations

import threading
from collections import deque


class Metrics:
    """Thread-safe counters and a rolling latency window."""

    def __init__(self, window: int = 512) -> None:
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=window)
        self._requests = 0
        self._errors = 0

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._requests += 1
            self._latencies.append(latency_ms)

    def record_error(self) -> None:
        with self._lock:
            self._requests += 1
            self._errors += 1

    def snapshot(self) -> dict:
        with self._lock:
            latencies = sorted(self._latencies)
            requests, errors = self._requests, self._errors

        return {
            "requests_total": requests,
            "errors_total": errors,
            "latency_ms": {
                "count": len(latencies),
                "p50": _percentile(latencies, 50),
                "p90": _percentile(latencies, 90),
                "p99": _percentile(latencies, 99),
                "max": round(latencies[-1], 2) if latencies else None,
            },
        }


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    """Nearest-rank percentile. Returns None for an empty window."""
    if not sorted_values:
        return None
    rank = max(1, round(percentile / 100 * len(sorted_values)))
    return round(sorted_values[rank - 1], 2)
