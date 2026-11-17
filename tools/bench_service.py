"""End-to-end service benchmark: real HTTP, real uploads, concurrency sweeps.

    python tools/bench_service.py
    python tools/bench_service.py --only trt16 --levels 1 8 64

Starts the service once per configuration, waits until every worker reports
ready, then holds 1, 2, 4 ... 128 requests in flight for a fixed time at each
level. Per level it records client-side throughput and latency percentiles,
the mean batch size and queue wait the server reported (from its Prometheus
histograms, differenced across the level), GPU utilisation, and how much CPU
the server's process tree burned.

The first configuration is v1 itself - the code as of the previous release,
extracted from git - so "before" is a measurement, not a recollection.

Results are written after every configuration, so a failure late in the run
keeps everything measured before it; `--only` re-runs a subset and merges.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from benchlib import MODELS, RESULTS, ROOT, SAMPLE_IMAGE, machine, write_results
from loadtest import run_level

V1_COMMIT = "eb850b4"
LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
TRT16 = "yolo26n-fp16-dyn32.engine"
BATCHING = {"VISION_MAX_BATCH_SIZE": "32", "VISION_MAX_BATCH_WAIT_MS": "2"}

CONFIGS = [
    {
        "key": "v1",
        "label": "v1: PyTorch FP32, blocking handler, no batching",
        "v1": True,
        "env": {"VISION_MODEL": "yolo26n.pt"},
    },
    {
        "key": "pt-nobatch",
        "label": "v2: PyTorch FP32, no batching",
        "env": {"VISION_MODEL": "yolo26n.pt", "VISION_MAX_BATCH_SIZE": "1"},
    },
    {
        "key": "trt16-nobatch",
        "label": "v2: TensorRT FP16, no batching",
        "env": {"VISION_MODEL": TRT16, "VISION_MAX_BATCH_SIZE": "1"},
    },
    {
        "key": "trt16-batch",
        "label": "v2: TensorRT FP16, dynamic batching (up to 32, 2 ms)",
        "env": {"VISION_MODEL": TRT16, **BATCHING},
    },
    {
        "key": "trt16-batch-raw",
        "label": "v2: TensorRT FP16, batching, raw-body upload",
        "endpoint": "/detect/raw",
        "env": {"VISION_MODEL": TRT16, **BATCHING},
    },
    {
        "key": "trt16-batch-raw-4w",
        "label": "v2: TensorRT FP16, batching, raw body, 4 workers on one GPU",
        "endpoint": "/detect/raw",
        "workers": 4,
        "env": {"VISION_MODEL": TRT16, **BATCHING},
    },
    {
        "key": "trt16-batch-nvjpeg",
        "label": "v2: TensorRT FP16, batching, nvJPEG decode",
        "env": {"VISION_MODEL": TRT16, **BATCHING, "VISION_GPU_DECODE": "true"},
    },
]


def scrape(url: str) -> dict[str, float]:
    import httpx2 as httpx

    try:
        text = httpx.get(f"{url}/metrics", timeout=5).text
    except Exception:
        return {}
    values = {}
    for name in (
        "vision_batch_size_sum",
        "vision_batch_size_count",
        "vision_queue_wait_seconds_sum",
        "vision_queue_wait_seconds_count",
    ):
        match = re.search(rf"^{name} ([0-9.e+-]+)$", text, re.MULTILINE)
        if match:
            values[name] = float(match.group(1))
    return values


def wait_ready(url: str, proc: subprocess.Popen, workers: int, timeout_s: float = 300) -> float:
    """Ready means every worker is ready: demand a long run of consecutive 200s,
    since each probe lands on whichever worker accepts the connection."""
    import httpx2 as httpx

    started = time.perf_counter()
    streak, needed = 0, 1 if workers == 1 else 40
    while time.perf_counter() - started < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        ok = False
        with contextlib.suppress(Exception), httpx.Client(timeout=2) as http:
            ok = http.get(f"{url}/readyz").status_code == 200
        streak = streak + 1 if ok else 0
        if streak >= needed:
            return time.perf_counter() - started
        time.sleep(0.05 if ok else 0.25)
    raise TimeoutError("server never became ready")


@contextlib.contextmanager
def server(config: dict, port: int, log_path: Path):
    env = {**os.environ, **config["env"], "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
    env["VISION_MODEL"] = str(MODELS / config["env"]["VISION_MODEL"])
    tmp = None
    if config.get("v1"):
        tmp = tempfile.TemporaryDirectory()
        archive = subprocess.run(
            ["git", "archive", V1_COMMIT, "src"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(tmp.name, filter="data")
        env["PYTHONPATH"] = str(Path(tmp.name) / "src")
    else:
        env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [sys.executable, "-m", "uvicorn", "vision.service:app", "--host", "127.0.0.1",
           "--port", str(port), "--log-level", "warning", "--no-access-log"]  # fmt: skip
    if config.get("workers", 1) > 1:
        cmd += ["--workers", str(config["workers"])]
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield proc
        finally:
            import psutil

            with contextlib.suppress(psutil.NoSuchProcess):
                for child in psutil.Process(proc.pid).children(recursive=True):
                    child.kill()
            proc.kill()
            proc.wait(15)
            if tmp is not None:
                tmp.cleanup()


def run_config(config: dict, args, payload: bytes) -> dict:
    import psutil

    from vision.gpu import GpuSampler

    url = f"http://127.0.0.1:{args.port}"
    workers = config.get("workers", 1)
    endpoint = config.get("endpoint", "/detect")
    logs = Path(tempfile.gettempdir()) / "yolo26-bench-service"
    logs.mkdir(exist_ok=True)
    with server(config, args.port, logs / f"{config['key']}.log") as proc:
        ready_s = wait_ready(url, proc, workers)
        print(f"    ready in {ready_s:.1f}s", flush=True)
        # On Windows a venv's python.exe is a launcher that runs the real
        # interpreter as a child; with --workers there are more. Measure the tree.
        root = psutil.Process(proc.pid)
        tree = [root, *root.children(recursive=True)]
        levels = []
        for level in args.levels:
            before = scrape(url) if workers == 1 else {}
            for member in tree:
                member.cpu_percent(None)
            with GpuSampler(interval_s=0.05) as sampler:
                outcome = run_level(
                    url, payload, level, args.duration, 2.0, args.processes, endpoint
                )
            outcome["server_cpu_pct"] = round(sum(m.cpu_percent(None) for m in tree), 1)
            # /metrics is per process: with several workers a scrape reaches
            # one of them, so batch statistics are only taken with one worker.
            after = scrape(url) if workers == 1 else {}
            done = after.get("vision_batch_size_count", 0) - before.get(
                "vision_batch_size_count", 0
            )
            if done > 0:
                total = after["vision_batch_size_sum"] - before["vision_batch_size_sum"]
                outcome["mean_batch_size"] = round(total / done, 2)
                waits = (
                    after["vision_queue_wait_seconds_count"]
                    - before["vision_queue_wait_seconds_count"]
                )
                wait_sum = (
                    after["vision_queue_wait_seconds_sum"] - before["vision_queue_wait_seconds_sum"]
                )
                outcome["mean_queue_wait_ms"] = round(1000 * wait_sum / waits, 2)
            outcome["gpu"] = sampler.summary()
            levels.append(outcome)
            lat = outcome.get("latency_ms", {})
            print(
                f"    c={level:<4d} {outcome['throughput_rps']:8.1f} req/s  "
                f"p50 {lat.get('p50', float('nan')):7.2f}  "
                f"p99 {lat.get('p99', float('nan')):7.2f} ms  "
                f"batch {outcome.get('mean_batch_size', '-'):>5}  "
                f"gpu {outcome['gpu'].get('utilization_mean_pct', '-'):>5}%  "
                f"cpu {outcome['server_cpu_pct']:6.1f}%  503s {outcome['shed_503']}",
                flush=True,
            )
    return {**config, "ready_seconds": round(ready_s, 1), "levels": levels}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default=None, help="substring filter on config key; merges")
    parser.add_argument("--levels", type=int, nargs="+", default=LEVELS)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--processes", type=int, default=6, help="load-generator processes")
    parser.add_argument("--port", type=int, default=8077)
    args = parser.parse_args()

    payload = SAMPLE_IMAGE.read_bytes()
    existing = RESULTS / "service.json"
    results: dict[str, dict] = {}
    if args.only and existing.exists():
        results = {c["key"]: c for c in json.loads(existing.read_text())["configs"]}

    info = machine()
    for config in CONFIGS:
        if args.only and args.only not in config["key"]:
            continue
        print(f"\n=== {config['label']}", flush=True)
        try:
            results[config["key"]] = run_config(config, args, payload)
        except Exception as error:
            print(f"    FAILED: {error!r}", flush=True)
            continue
        order = [c["key"] for c in CONFIGS]
        write_results(
            "service",
            {
                "machine": info,
                "image_bytes": len(payload),
                "duration_s": args.duration,
                "load_processes": args.processes,
                "configs": sorted(results.values(), key=lambda c: order.index(c["key"])),
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
