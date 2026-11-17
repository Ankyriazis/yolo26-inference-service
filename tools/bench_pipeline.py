"""Pipeline benchmark: runtime x precision x batch x model size, in-process.

    python tools/bench_pipeline.py                 # every suite
    python tools/bench_pipeline.py --suite runtime # one suite

Measures detector.predict_batch() directly - no HTTP - on a real 810x1080
photograph that has already been decoded, which is the state an image is in
when it reaches the batcher. Each configuration runs in its own process, so
GPU memory and CUDA state from one cannot leak into the next.

Per configuration and batch size it records latency percentiles, images per
second, the per-stage split of each batch (from CUDA events on the GPU
pipeline), and GPU utilisation sampled through NVML while it ran. Per
configuration it records cold-start costs: import, load, the first inference
before any warmup, and the VRAM the process added.

Suites:
    runtime   yolo26n through PyTorch, ONNX Runtime and TensorRT, FP32/FP16/INT8
    pipeline  stock Ultralytics predictor vs the GPU pipeline; nvJPEG; NMS head
    scaling   yolo26 s/m/l/x on TensorRT FP16 and INT8
    cpu       the same model with no GPU at all
    decode    JPEG decode: OpenCV on 1..N threads vs nvJPEG
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

_PROCESS_STARTED = time.perf_counter()

from benchlib import MODELS, SAMPLE_IMAGE, append_jsonl, machine, summarize, write_results  # noqa: E402

ALL_BATCHES = [1, 2, 4, 8, 16, 32]
FEW_BATCHES = [1, 8, 32]
# yolo26x cannot build for batch 32 on an 8 GB GPU (TensorRT runs out of memory
# mid-build), so its engines top out at 16.
MAX_BATCH = {"n": 32, "s": 32, "m": 32, "l": 32, "x": 16}


def cfg(suite, label, model, pipeline="gpu", batches=None, **extra):
    return {
        "suite": suite,
        "label": label,
        "model": model,
        "pipeline": pipeline,
        "batches": batches or ALL_BATCHES,
        **extra,
    }


MATRIX = [
    # ---- runtime x precision on yolo26n
    cfg("runtime", "PyTorch FP32", "yolo26n.pt"),
    cfg("runtime", "PyTorch FP16", "yolo26n.pt", half=True),
    cfg("runtime", "ONNX Runtime CUDA FP32", "yolo26n-fp32-dyn32.onnx"),
    cfg("runtime", "TensorRT FP32", "yolo26n-fp32-dyn32.engine"),
    cfg("runtime", "TensorRT FP16", "yolo26n-fp16-dyn32.engine"),
    cfg("runtime", "TensorRT INT8", "yolo26n-int8-dyn32.engine"),
    cfg("runtime", "TensorRT FP16 static b1 (CUDA graph)", "yolo26n-fp16-b1.engine", batches=[1]),
    # ---- where the time goes outside the model
    cfg(
        "pipeline", "Ultralytics predictor, PyTorch FP32", "yolo26n.pt", "ultralytics", FEW_BATCHES
    ),
    cfg(
        "pipeline",
        "Ultralytics predictor, TensorRT FP16",
        "yolo26n-fp16-dyn32.engine",
        "ultralytics",
        FEW_BATCHES,
    ),
    cfg("pipeline", "GPU pipeline, TensorRT FP16", "yolo26n-fp16-dyn32.engine", "gpu", FEW_BATCHES),
    cfg(
        "pipeline",
        "GPU pipeline + nvJPEG decode, TensorRT FP16",
        "yolo26n-fp16-dyn32.engine",
        "gpu",
        FEW_BATCHES,
        input="jpeg",
    ),
    cfg(
        "pipeline",
        "GPU pipeline, TensorRT FP16, one-to-many head + NMS",
        "yolo26n-o2m-fp16-dyn32.engine",
        "gpu",
        FEW_BATCHES,
        nms_free=False,
    ),
    # ---- model scaling
    *[
        cfg(
            "scaling",
            f"yolo26{s} TensorRT {p.upper()}",
            f"yolo26{s}-{p}-dyn{MAX_BATCH[s]}.engine",
            batches=[1, 8, MAX_BATCH[s]],
            size=s,
            precision=p,
        )
        for s in "nsmlx"
        for p in ("fp16", "int8")
    ],
    # ---- no GPU
    cfg("cpu", "CPU PyTorch FP32", "yolo26n.pt", "ultralytics", [1, 8], device="cpu", seconds=6),
    cfg(
        "cpu",
        "CPU ONNX Runtime FP32",
        "yolo26n-fp32-dyn32.onnx",
        "ultralytics",
        [1, 8],
        device="cpu",
        seconds=6,
    ),
]


# ------------------------------------------------------------------ worker


def run_config(config: dict) -> dict:
    import_started = time.perf_counter()
    import cv2

    from vision.detector import GpuDetector, UltralyticsDetector
    from vision.gpu import GpuSampler, GpuTelemetry

    import_seconds = time.perf_counter() - import_started
    telemetry = GpuTelemetry()
    baseline = telemetry.sample()

    frame = cv2.imread(str(SAMPLE_IMAGE))
    jpeg = SAMPLE_IMAGE.read_bytes()
    cls = GpuDetector if config["pipeline"] == "gpu" else UltralyticsDetector
    detector = cls(
        model=str(MODELS / config["model"]),
        device=config.get("device"),
        half=config.get("half", False),
        nms_free=config.get("nms_free", True),
    )
    item = jpeg if config.get("input") == "jpeg" else frame

    started = time.perf_counter()
    first = detector.predict_batch([item])[0]
    first_call_ms = (time.perf_counter() - started) * 1000
    if isinstance(first, Exception):
        raise first

    record = {
        **config,
        "backend": detector.backend,
        "precision_actual": detector.precision,
        "end2end": getattr(detector, "end2end", None),
        "import_seconds": round(import_seconds, 2),
        "load_seconds": round(detector.load_seconds, 2),
        "first_call_ms": round(first_call_ms, 1),
        "detections": len(first.detections),
        "batches": [],
    }

    budget = config.get("seconds", 3.0)
    peak_memory = 0.0
    for batch in config["batches"]:
        items = [item] * batch
        for _ in range(5):  # warm this shape: dynamic engines select kernels per shape
            detector.predict_batch(items)

        latencies, stages = [], []
        with GpuSampler(interval_s=0.02) as sampler:
            loop_started = time.perf_counter()
            while True:
                t0 = time.perf_counter()
                results = detector.predict_batch(items)
                latencies.append((time.perf_counter() - t0) * 1000)
                stages.append(results[0].stages_ms)
                elapsed = time.perf_counter() - loop_started
                if (elapsed > budget and len(latencies) >= 20) or len(latencies) >= 2000:
                    break
        wall = time.perf_counter() - loop_started
        gpu = sampler.summary()
        peak_memory = max(peak_memory, gpu.get("memory_peak_mb", 0.0))
        stage_means = {
            key: round(sum(s.get(key, 0.0) for s in stages) / len(stages), 3)
            for key in (stages[0] or {})
        }
        entry = {
            "batch": batch,
            "latency_ms": summarize(latencies),
            "images_per_second": round(batch * len(latencies) / wall, 1),
            "stages_ms": stage_means,
            "gpu": gpu,
        }
        record["batches"].append(entry)
        print(
            f"  b={batch:<3d} p50 {entry['latency_ms']['p50']:8.2f} ms  "
            f"p99 {entry['latency_ms']['p99']:8.2f} ms  "
            f"{entry['images_per_second']:8.1f} img/s  "
            f"util {gpu.get('utilization_mean_pct', '-')}%",
            flush=True,
        )

    if baseline is not None and peak_memory:
        record["vram_added_mb"] = round(peak_memory - baseline.memory_used_mb, 1)
    record["process_seconds"] = round(time.perf_counter() - _PROCESS_STARTED, 1)
    return record


def run_decode() -> dict:
    """JPEG decode throughput: where should decode happen?"""
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    import numpy as np
    import torch
    from torchvision.io import ImageReadMode, decode_jpeg

    payload = SAMPLE_IMAGE.read_bytes()
    buffer = np.frombuffer(payload, np.uint8)
    out = {"suite": "decode", "label": "JPEG decode", "image_bytes": len(payload), "results": []}

    def measure(label, fn, images_per_call, seconds=2.0):
        for _ in range(3):
            fn()
        count, started = 0, time.perf_counter()
        while time.perf_counter() - started < seconds:
            fn()
            count += images_per_call
        rate = count / (time.perf_counter() - started)
        out["results"].append({"label": label, "images_per_second": round(rate, 1)})
        print(f"  {label:40s} {rate:8.1f} img/s", flush=True)

    measure("OpenCV, 1 thread", lambda: cv2.imdecode(buffer, cv2.IMREAD_COLOR), 1)
    for threads in (4, 8, 16):
        pool = ThreadPoolExecutor(threads)
        work = 64

        def parallel(pool=pool, work=work):
            list(pool.map(lambda _: cv2.imdecode(buffer, cv2.IMREAD_COLOR), range(work)))

        measure(f"OpenCV, {threads} threads", parallel, work)
        pool.shutdown()

    tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    for batch in (1, 8, 32):

        def gpu(batch=batch):
            decode_jpeg([tensor] * batch, device="cuda", mode=ImageReadMode.RGB)
            torch.cuda.synchronize()

        measure(f"nvJPEG (CUDA), batch {batch}", gpu, batch)
    return out


# ------------------------------------------------------------------ orchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", action="append", help="runtime|pipeline|scaling|cpu|decode")
    parser.add_argument("--only", default=None, help="substring filter on label; merges results")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        config = json.loads(args.worker)
        result = run_decode() if config["suite"] == "decode" else run_config(config)
        print("RESULT " + json.dumps(result), flush=True)
        return 0

    suites = set(args.suite or ["runtime", "pipeline", "scaling", "cpu", "decode"])
    plan = [c for c in MATRIX if c["suite"] in suites]
    if args.only:
        plan = [c for c in plan if args.only in c["label"]]
    if "decode" in suites:
        plan.append({"suite": "decode", "label": "JPEG decode"})

    from benchlib import RESULTS

    log = RESULTS / "pipeline.jsonl"
    log.unlink(missing_ok=True)
    records = []
    for config in plan:
        if config["suite"] != "decode" and not (MODELS / config["model"]).exists():
            print(f"\n--- {config['label']}: SKIPPED, {config['model']} not built")
            continue
        print(f"\n--- {config['label']}  ({config.get('model', '')})", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", json.dumps(config)],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [line for line in proc.stdout.splitlines() if not line.startswith("RESULT ")]
        print("\n".join(lines))
        result = next(
            (line[7:] for line in proc.stdout.splitlines() if line.startswith("RESULT ")), None
        )
        if proc.returncode != 0 or result is None:
            print(f"FAILED:\n{proc.stderr[-2000:]}")
            records.append({**config, "error": proc.stderr[-2000:]})
            continue
        record = json.loads(result)
        records.append(record)
        append_jsonl(log, record)

    if args.only or args.suite:  # partial run: replace matching records, keep the rest
        previous = RESULTS / "pipeline.json"
        if previous.exists():
            fresh = {r["label"] for r in records}
            kept = [
                r for r in json.loads(previous.read_text())["records"] if r["label"] not in fresh
            ]
            order = [c["label"] for c in MATRIX] + ["JPEG decode"]
            records = sorted(
                kept + records,
                key=lambda r: order.index(r["label"]) if r["label"] in order else len(order),
            )
    write_results("pipeline", {"machine": machine(), "records": records})
    return 0


if __name__ == "__main__":
    sys.exit(main())
