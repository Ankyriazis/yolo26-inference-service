"""GPU telemetry through NVML.

Used in two places: the service reports device state on `/gpu` and as
Prometheus gauges, and the benchmark tools sample it while they run so a
throughput figure can be read alongside how busy the GPU actually was.

NVML is optional. On a machine without an NVIDIA driver every call returns
None rather than raising, so nothing that imports this module needs a GPU.
Some boards do not report power draw at all (the RTX 4060 this was developed
on is one); those fields come back as None and callers must cope.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

try:
    import pynvml
except ImportError:  # pragma: no cover - nvidia-ml-py is a declared dependency
    pynvml = None


@dataclass(frozen=True)
class GpuSample:
    name: str
    utilization_pct: int
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: int | None
    sm_clock_mhz: int | None
    power_w: float | None
    power_limit_w: float | None
    pcie_gen: int | None
    pcie_width: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def _try(call, *args):
    try:
        return call(*args)
    except Exception:  # NVML raises a zoo of NotSupported/Unknown errors
        return None


class GpuTelemetry:
    """Reads one GPU's state. Every method is safe to call without a GPU."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self._handle = None
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as error:
            logger.info("NVML unavailable (%s); GPU telemetry disabled", error)
            self._handle = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def driver_version(self) -> str | None:
        if not self.available:
            return None
        return _try(pynvml.nvmlSystemGetDriverVersion)

    def sample(self) -> GpuSample | None:
        if not self.available:
            return None
        handle = self._handle
        util = _try(pynvml.nvmlDeviceGetUtilizationRates, handle)
        memory = _try(pynvml.nvmlDeviceGetMemoryInfo, handle)
        power_mw = _try(pynvml.nvmlDeviceGetPowerUsage, handle)
        limit_mw = _try(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)
        return GpuSample(
            name=_try(pynvml.nvmlDeviceGetName, handle) or "unknown",
            utilization_pct=util.gpu if util else 0,
            memory_used_mb=round(memory.used / 2**20, 1) if memory else 0.0,
            memory_total_mb=round(memory.total / 2**20, 1) if memory else 0.0,
            temperature_c=_try(
                pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
            ),
            sm_clock_mhz=_try(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM),
            power_w=power_mw / 1000 if power_mw else None,
            power_limit_w=limit_mw / 1000 if limit_mw else None,
            pcie_gen=_try(pynvml.nvmlDeviceGetCurrPcieLinkGeneration, handle),
            pcie_width=_try(pynvml.nvmlDeviceGetCurrPcieLinkWidth, handle),
        )


class GpuSampler:
    """Samples a GPU on a background thread for the duration of a `with` block.

        with GpuSampler() as sampler:
            run_benchmark()
        print(sampler.summary())

    NVML's utilisation counter is "fraction of the last sample period in which
    a kernel was running", so it reads 100% for a GPU that is busy but badly
    fed with tiny kernels. Treat it as an upper bound on how hard the GPU is
    working, and read it next to throughput, never instead of it.
    """

    def __init__(self, interval_s: float = 0.05, index: int = 0) -> None:
        self.interval_s = interval_s
        self.telemetry = GpuTelemetry(index)
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self.duration_s = 0.0

    def __enter__(self) -> GpuSampler:
        self._started = time.perf_counter()
        if self.telemetry.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.duration_s = time.perf_counter() - self._started

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self.telemetry.sample()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def summary(self) -> dict:
        if not self.samples:
            return {"available": False}
        util = [s.utilization_pct for s in self.samples]
        memory = [s.memory_used_mb for s in self.samples]
        power = [s.power_w for s in self.samples if s.power_w is not None]
        clocks = [s.sm_clock_mhz for s in self.samples if s.sm_clock_mhz]
        first = self.samples[0]
        return {
            "available": True,
            "name": first.name,
            "samples": len(self.samples),
            "utilization_mean_pct": round(sum(util) / len(util), 1),
            "utilization_max_pct": max(util),
            "memory_peak_mb": max(memory),
            "sm_clock_mean_mhz": round(sum(clocks) / len(clocks)) if clocks else None,
            "power_mean_w": round(sum(power) / len(power), 1) if power else None,
            "power_limit_w": first.power_limit_w,
            "temperature_max_c": max((s.temperature_c or 0) for s in self.samples),
        }
