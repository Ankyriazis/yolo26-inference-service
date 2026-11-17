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


class TrtEngine:
    """A TensorRT engine run directly - no Ultralytics, so any I/O layout works.

    Reads the JSON header tools/export.py writes in front of the serialized
    plan. Runs on its own CUDA stream: on the legacy default stream TensorRT
    adds synchronisations of its own, which would be timed as model latency.
    """

    def __init__(self, path: Path) -> None:
        import numpy as np
        import tensorrt as trt
        import torch

        data = Path(path).read_bytes()
        size = int.from_bytes(data[:4], "little", signed=True)
        self.metadata = json.loads(data[4 : 4 + size])
        self._logger = trt.Logger(trt.Logger.ERROR)
        self.engine = trt.Runtime(self._logger).deserialize_cuda_engine(data[4 + size :])
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        is_input = trt.TensorIOMode.INPUT
        self.input = next(n for n in names if self.engine.get_tensor_mode(n) == is_input)
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) != is_input]
        self.dtypes = {
            n: torch.from_numpy(np.empty(0, trt.nptype(self.engine.get_tensor_dtype(n)))).dtype
            for n in names
        }
        self.static = -1 not in tuple(self.engine.get_tensor_shape(self.input))
        self.activation_mb = self.engine.device_memory_size_v2 / 2**20
        self.stream = torch.cuda.Stream()
        self._outputs: dict[tuple, dict] = {}

    def bind(self, x) -> dict:
        """Point the context at x and at output buffers for its shape (reused)."""
        import torch

        shape = tuple(x.shape)
        self.context.set_input_shape(self.input, shape)
        self.context.set_tensor_address(self.input, x.data_ptr())
        if shape not in self._outputs:
            self._outputs[shape] = {
                n: torch.empty(
                    tuple(self.context.get_tensor_shape(n)), dtype=self.dtypes[n], device="cuda"
                )
                for n in self.outputs
            }
        for name, tensor in self._outputs[shape].items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        return self._outputs[shape]

    def enqueue(self) -> None:
        self.context.execute_async_v3(self.stream.cuda_stream)

    def __call__(self, x) -> dict:
        self.stream.wait_stream(__import__("torch").cuda.current_stream())
        out = self.bind(x)
        self.enqueue()
        self.stream.synchronize()
        return out

    def capture(self, x):
        """Record one inference as a CUDA graph; replaying it skips the per-launch
        CPU cost. Static engines only. Returns (replay, outputs): replay() runs
        the graph on this engine's stream (PyTorch would use the current one)."""
        import torch

        out = self(x)  # the first run allocates; keep that out of the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            self.enqueue()

        def replay() -> None:
            with torch.cuda.stream(self.stream):
                graph.replay()

        return replay, out


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
