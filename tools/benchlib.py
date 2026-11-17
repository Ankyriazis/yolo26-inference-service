"""Shared helpers for the benchmark tools: paths, statistics, provenance."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # see src/vision/__init__.py

CACHE = Path(os.environ.get("YOLO26_CACHE", Path.home() / ".cache" / "yolo26-inference-service"))
MODELS = CACHE / "models"
DATASETS = CACHE / "datasets"
RESULTS = ROOT / "results"
SAMPLE_IMAGE = ROOT / "data" / "bus.jpg"


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile, the definition the service's /stats uses."""
    ordered = sorted(values)
    rank = max(1, round(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(latencies_ms: list[float]) -> dict:
    return {
        "p50": round(percentile(latencies_ms, 50), 3),
        "p90": round(percentile(latencies_ms, 90), 3),
        "p99": round(percentile(latencies_ms, 99), 3),
        "mean": round(sum(latencies_ms) / len(latencies_ms), 3),
        "min": round(min(latencies_ms), 3),
        "max": round(max(latencies_ms), 3),
        "n": len(latencies_ms),
    }


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def machine() -> dict:
    """Everything a reader needs to judge whether a number transfers to them."""
    info = {
        "cpu": cpu_name(),
        "cpu_threads": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 2**30, 1)
            info["gpu_sm"] = f"{props.major}.{props.minor}"
            info["gpu_sms"] = props.multi_processor_count
    except ImportError:
        pass
    for module in ("tensorrt", "onnxruntime", "ultralytics", "cv2", "fastapi"):
        with contextlib.suppress(Exception):
            info[module] = __import__(module).__version__
    try:
        from vision.gpu import GpuTelemetry

        telemetry = GpuTelemetry()
        info["driver"] = telemetry.driver_version()
        sample = telemetry.sample()
        if sample is not None:
            info["gpu_power_limit_w"] = sample.power_limit_w
            info["gpu_power_reported"] = sample.power_w is not None
    except Exception:
        pass
    return info


def write_results(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
