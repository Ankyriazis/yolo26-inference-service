"""Oriented boxes: YOLO26-OBB vs RiO-DETRv2 on one TensorRT 11 harness.

    python tools/bench_obb.py                 # speed and parity -> results/obb.json
    python tools/bench_obb.py --suite speed

Build the engines first (1024x1024, FP16; m/l/x top out at batch 8 on 8 GB):
    python tools/export.py --model yolo26n-obb.pt --imgsz 1024 --batch 16 --opt-batch 8
    python tools/export_rio.py --rio-src ~/src/RiO-DETRv2 --size n --batch 16
and, for the batch-1 latency check, the same with --static --batch 1.

Both families go through the identical path: FP32 input already on the GPU (the
state a batch is in when it reaches the engine), FP16 baked in by ModelOpt
AutoCast, the same TensorRT build and profile, top-k post-processing inside
the graph, no NMS. Timing is per batch with a synchronise after each, as a
serving loop runs. Each engine runs in its own process, and the whole matrix
runs three times, interleaved; the report takes the median of each measurement.

Accuracy is not measured here. DOTA-v1.0 test labels are withheld (scores
come from the official server) and RiO-DETRv2 trains on train+val, so no
local split is fair to both. The published test-server scores are recorded
with their protocol instead. What runs here is a parity check - each FP16
engine against its own FP32 ONNX graph on 8 DOTA tiles (Ultralytics' dota8) -
and a ground-truth sanity check that boxes, angles and classes decode.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from benchlib import DATASETS, MODELS, TrtEngine, append_jsonl, machine, summarize, write_results

SIZES = "nsmlx"
DYN_BATCH = {"n": 16, "s": 16, "m": 8, "l": 8, "x": 8}
FAMILIES = {
    "yolo26-obb": lambda s: f"yolo26{s}-obb",
    "riov2": lambda s: f"riov2-{s}-obb",
}
LABEL = {"yolo26-obb": "YOLO26-OBB", "riov2": "RiO-DETRv2"}
DOTA8_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/dota8.zip"
# RiO-DETRv2 class order -> Ultralytics DOTAv1 class order (dota8 labels use the latter)
RIO_TO_DOTAV1 = [0, 3, 8, 6, 10, 9, 1, 4, 5, 2, 13, 12, 7, 14, 11]

# DOTA-v1.0 test-server scores, as each author publishes them. Different
# protocols, so they are reported side by side, never on one axis.
PUBLISHED = {
    "yolo26-obb": {
        "source": "https://docs.ultralytics.com/tasks/obb/",
        "protocol": "DOTA-v1.0 test server, multi-scale",
        "map50_95": {"n": 52.4, "s": 54.8, "m": 55.3, "l": 56.2, "x": 56.7},
        "map50": {"n": 78.9, "s": 80.9, "m": 81.0, "l": 81.6, "x": 81.7},
        "params_m": {"n": 2.4, "s": 9.8, "m": 21.2, "l": 25.6, "x": 57.6},
        "t4_tensorrt_ms": {"n": 2.8, "s": 4.9, "m": 10.2, "l": 13.0, "x": 30.5},
    },
    "riov2": {
        "source": "https://github.com/RicePasteM/RiO-DETRv2",
        "protocol": "DOTA-v1.0 test server, single-scale",
        "map50_95": {"n": 49.20, "s": 51.82, "m": 53.20, "l": 53.63, "x": 54.13},
        "map50": {"n": 79.10, "s": 79.56, "m": 81.20, "l": 80.77, "x": 80.99},
        "params_m": {"n": 4.3, "s": 8.9, "m": 19.5, "l": 28.6, "x": 64.6},
    },
}


def plan() -> list[dict]:
    configs = []
    for size in SIZES:
        for family, stem in FAMILIES.items():
            batch = DYN_BATCH[size]
            configs.append(
                {
                    "family": family,
                    "size": size,
                    "label": f"{LABEL[family]}-{size.upper()}",
                    "engine": f"{stem(size)}-fp16-dyn{batch}.engine",
                    "onnx": f"{stem(size)}-dyn{batch}.onnx",
                    "static": False,
                    "batches": sorted({1, 8, batch}),
                }
            )
            configs.append(
                {
                    "family": family,
                    "size": size,
                    "label": f"{LABEL[family]}-{size.upper()} static b1",
                    "engine": f"{stem(size)}-fp16-b1.engine",
                    "static": True,
                    "batches": [1],
                }
            )
    return configs


# ------------------------------------------------------------------ speed


def timed(fn, stream, seconds: float):
    import torch

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall, gpu = [], []
    began = time.perf_counter()
    while time.perf_counter() - began < seconds or len(wall) < 50:
        t0 = time.perf_counter()
        start.record(stream)
        fn()
        end.record(stream)
        stream.synchronize()
        wall.append((time.perf_counter() - t0) * 1000)
        gpu.append(start.elapsed_time(end))
    return wall, gpu, time.perf_counter() - began


def speed_worker(config: dict, seconds: float) -> dict:
    import torch

    from vision.gpu import GpuSampler

    engine = TrtEngine(MODELS / config["engine"])
    record = {
        **config,
        "activation_memory_mb": round(engine.activation_mb, 1),
        "engine_mb": round((MODELS / config["engine"]).stat().st_size / 1e6, 1),
        "results": [],
    }
    modes = [("plain", b) for b in config["batches"]]
    # The same engine enqueued on CUDA's legacy default stream, where TensorRT
    # inserts synchronisations of its own - how many quick scripts call it.
    modes.append(("default_stream", 1))
    if config["static"]:
        modes.append(("cuda_graph", 1))
    for mode, batch in modes:
        x = torch.rand(batch, 3, 1024, 1024, device="cuda")
        stream = engine.stream
        if mode == "cuda_graph":
            run, _ = engine.capture(x)
        elif mode == "default_stream":
            stream = torch.cuda.default_stream()
            engine.bind(x)

            def run(stream=stream):
                engine.context.execute_async_v3(stream.cuda_stream)
        else:
            engine.bind(x)
            run = engine.enqueue
        for _ in range(20):
            run()
        stream.synchronize()
        with GpuSampler(interval_s=0.02) as sampler:
            wall, gpu, elapsed = timed(run, stream, seconds)
        entry = {
            "mode": mode,
            "batch": batch,
            "latency_ms": summarize(wall),
            "gpu_ms": summarize(gpu),
            "images_per_second": round(batch * len(wall) / elapsed, 1),
            "gpu": sampler.summary(),
        }
        record["results"].append(entry)
        print(
            f"  {config['label']:28s} {mode:10s} b={batch:<2d} "
            f"p50 {entry['latency_ms']['p50']:7.2f} ms  p99 {entry['latency_ms']['p99']:7.2f} ms  "
            f"{entry['images_per_second']:7.1f} img/s  "
            f"util {entry['gpu'].get('utilization_mean_pct', '-')}%",
            flush=True,
        )
    return record


# ------------------------------------------------------------------ parity


def dota8() -> Path:
    root = DATASETS / "dota8"
    if not (root / "images").exists():
        DATASETS.mkdir(parents=True, exist_ok=True)
        archive = DATASETS / "dota8.zip"
        import urllib.request

        urllib.request.urlretrieve(DOTA8_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DATASETS)
    return root


def tiles():
    import cv2
    import numpy as np
    import torch
    from ultralytics.utils.ops import xyxyxyxy2xywhr

    root = dota8()
    files = sorted((root / "images").rglob("*.jpg"))
    frames = [cv2.imread(str(f)) for f in files]
    assert all(f.shape[:2] == (1024, 1024) for f in frames), "dota8 tiles are 1024x1024"
    x = np.stack([f[..., ::-1].transpose(2, 0, 1) for f in frames]).astype(np.float32) / 255
    truth = []
    for f in files:
        rows = np.loadtxt(root / "labels" / f.parent.name / f"{f.stem}.txt", ndmin=2)
        boxes = torch.as_tensor(xyxyxyxy2xywhr(rows[:, 1:] * 1024)).float()
        truth.append((torch.tensor(rows[:, 0]).long(), boxes))
    return np.ascontiguousarray(x), truth


def decode(family: str, outputs: list) -> list:
    """Engine or ONNX outputs -> per image (labels in DOTAv1 order, xywhr, scores)."""
    import torch

    outputs = [torch.as_tensor(o).cpu() for o in outputs]
    if family == "riov2":
        labels, boxes, scores = outputs
        lut = torch.tensor(RIO_TO_DOTAV1)
        return [
            (lut[lab.long()], box.float(), sco.float())
            for lab, box, sco in zip(labels, boxes, scores, strict=True)
        ]
    (raw,) = outputs  # end2end OBB head: x y w h score class angle
    return [(r[:, 5].long(), r[:, [0, 1, 2, 3, 6]].float(), r[:, 4].float()) for r in raw.float()]


def recall(dets, truth, conf=0.25, iou=0.5) -> dict:
    from ultralytics.utils.metrics import batch_probiou

    hit = total = kept = 0
    for (labels, boxes, scores), (gl, gb) in zip(dets, truth, strict=True):
        keep = scores >= conf
        labels, boxes = labels[keep], boxes[keep]
        kept += int(keep.sum())
        total += len(gl)
        if len(boxes):
            same = gl[:, None] == labels[None, :]
            hit += int(((batch_probiou(gb, boxes) >= iou) & same).any(1).sum())
    return {"recall": round(hit / total, 3), "objects": total, "detections": kept}


def agreement(ref, test, conf=0.3) -> dict:
    """Share of reference detections (score >= conf) the engine reproduces:
    same class, ProbIoU >= 0.9. Plus the mean score difference on those."""
    from ultralytics.utils.metrics import batch_probiou

    n = matched = 0
    diffs = []
    for (la, ba, sa), (lb, bb, sb) in zip(ref, test, strict=True):
        keep = sa >= conf
        la, ba, sa = la[keep], ba[keep], sa[keep]
        n += len(la)
        if not len(la):
            continue
        ok = (batch_probiou(ba, bb) >= 0.9) & (la[:, None] == lb[None, :])
        hits = ok.any(1)
        matched += int(hits.sum())
        best = (batch_probiou(ba, bb) * ok).argmax(1)
        diffs += (sa - sb[best]).abs()[hits].tolist()
    return {
        "detections": n,
        "matched": matched,
        "match_rate": round(matched / max(n, 1), 4),
        "mean_abs_score_diff": round(sum(diffs) / max(len(diffs), 1), 4),
    }


def parity_worker(config: dict) -> dict:
    import onnxruntime as ort
    import torch

    x, truth = tiles()
    session = ort.InferenceSession(
        str(MODELS / config["onnx"]), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name
    chunks = [session.run(None, {name: x[i : i + 2]}) for i in range(0, len(x), 2)]
    reference = decode(
        config["family"],
        [torch.cat([torch.as_tensor(c[k]) for c in chunks]) for k in range(len(chunks[0]))],
    )
    del session

    engine = TrtEngine(MODELS / config["engine"])
    out = engine(torch.from_numpy(x).cuda())
    fp16 = decode(config["family"], [out[n] for n in engine.outputs])
    return {
        "family": config["family"],
        "size": config["size"],
        "label": config["label"],
        "fp16_vs_fp32": agreement(reference, fp16),
        "ground_truth_fp32": recall(reference, truth),
        "ground_truth_fp16": recall(fp16, truth),
    }


# ------------------------------------------------------------------ orchestrator


def run(kind: str, config: dict, seconds: float) -> dict | None:
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", kind, json.dumps(config), str(seconds)],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = proc.stdout.splitlines()
    print("\n".join(line for line in lines if not line.startswith("RESULT ")))
    result = next((line[7:] for line in lines if line.startswith("RESULT ")), None)
    if proc.returncode or result is None:
        print(f"FAILED {config['label']}:\n{proc.stderr[-1500:]}")
        return None
    return json.loads(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", action="append", choices=["speed", "parity"])
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=3.0, help="per batch size")
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        kind, config, seconds = args.worker
        config = json.loads(config)
        fn = speed_worker if kind == "speed" else lambda c, _s: parity_worker(c)
        print("RESULT " + json.dumps(fn(config, float(seconds))), flush=True)
        return 0

    from benchlib import RESULTS

    suites = set(args.suite or ["speed", "parity"])
    configs = [c for c in plan() if (MODELS / c["engine"]).exists()]
    missing = [c["engine"] for c in plan() if c not in configs]
    if missing:
        print("not built, skipped: " + ", ".join(missing))

    previous = RESULTS / "obb.json"
    payload = json.loads(previous.read_text()) if previous.exists() else {}
    log = RESULTS / "obb.jsonl"
    if "speed" in suites:
        speed = []
        for repeat in range(args.passes):
            print(f"\n=== speed, pass {repeat + 1} of {args.passes}", flush=True)
            for config in configs:
                record = run("speed", config, args.seconds)
                if record:
                    record["pass"] = repeat
                    speed.append(record)
                    append_jsonl(log, record)
        payload["speed"] = speed
    if "parity" in suites:
        print("\n=== parity on dota8", flush=True)
        parity = []
        for config in configs:
            if config["static"] or not (MODELS / config["onnx"]).exists():
                continue
            record = run("parity", config, 0)
            if record:
                print(f"  {record['label']:16s} {json.dumps(record['fp16_vs_fp32'])}", flush=True)
                parity.append(record)
        payload["parity"] = parity
    payload["published"] = PUBLISHED
    payload["machine"] = machine()
    write_results("obb", payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
