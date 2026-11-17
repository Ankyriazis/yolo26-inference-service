"""Accuracy on COCO val2017, measured through the same detectors the service runs.

    python tools/bench_accuracy.py                       # every configuration
    python tools/bench_accuracy.py --only yolo26n        # a subset
    python tools/bench_accuracy.py --limit 500           # quick check

Why measure rather than quote the model card: precision is a deployment
decision, and its cost in accuracy depends on the quantisation recipe, the
runtime and the pre-processing actually used. This scores each artifact on
all 5,000 val2017 images with the official COCO evaluator (pycocotools), at
the conventional settings - confidence 0.001, up to 300 detections.

It also validates the GPU pipeline: the same weights through the stock
Ultralytics predictor and through `GpuDetector` should score within noise of
each other, or the custom pre/post-processing is wrong.

Each run doubles as an offline-throughput measurement: images per second for
the whole job, JPEG decode (on a CPU thread pool, overlapped with the GPU)
included - the figure that sizes a batch-processing job over an archive.

INT8 engines were calibrated on coco128, which is drawn from train2017, so no
evaluation image was seen during calibration.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from benchlib import DATASETS, MODELS, append_jsonl, machine, write_results

COCO = DATASETS / "coco"

MATRIX = [
    # pipeline validation: same weights, two pre/post-processing paths
    {
        "label": "yolo26n PyTorch FP32 (Ultralytics predictor)",
        "model": "yolo26n.pt",
        "pipeline": "ultralytics",
        "size": "n",
        "precision": "fp32",
        "runtime": "PyTorch",
    },
    {
        "label": "yolo26n PyTorch FP32 (GPU pipeline)",
        "model": "yolo26n.pt",
        "pipeline": "gpu",
        "size": "n",
        "precision": "fp32",
        "runtime": "PyTorch",
    },
    {
        "label": "yolo26n PyTorch FP32, one-to-many head + NMS (GPU pipeline)",
        "model": "yolo26n.pt",
        "pipeline": "gpu",
        "nms_free": False,
        "size": "n",
        "precision": "fp32",
        "runtime": "PyTorch",
    },
    {
        "label": "yolo26n TensorRT FP32",
        "model": "yolo26n-fp32-dyn32.engine",
        "pipeline": "gpu",
        "size": "n",
        "precision": "fp32",
        "runtime": "TensorRT",
    },
    # precision x size on TensorRT
    *[
        {
            "label": f"yolo26{s} TensorRT {p.upper()}",
            "model": f"yolo26{s}-{p}-dyn{16 if s == 'x' else 32}.engine",
            "pipeline": "gpu",
            "size": s,
            "precision": p,
            "runtime": "TensorRT",
        }
        for s in "nsmlx"
        for p in ("fp16", "int8")
    ],
]


def coco80_to_91() -> list[int]:
    from ultralytics.data.converter import coco80_to_coco91_class

    return coco80_to_coco91_class()


def evaluate(config: dict, limit: int | None, batch: int) -> dict:
    import cv2
    from pycocotools.coco import COCO as CocoApi
    from pycocotools.cocoeval import COCOeval

    from vision.detector import GpuDetector, UltralyticsDetector
    from vision.gpu import GpuSampler

    with contextlib.redirect_stdout(io.StringIO()):
        truth = CocoApi(str(COCO / "annotations" / "instances_val2017.json"))
    images = sorted(truth.dataset["images"], key=lambda i: i["id"])
    if limit:
        images = images[:limit]
    image_dir = COCO / "images" / "val2017"
    to_coco = coco80_to_91()

    cls = GpuDetector if config["pipeline"] == "gpu" else UltralyticsDetector
    detector = cls(
        model=str(MODELS / config["model"]),
        confidence=0.001,
        max_detections=300,
        nms_free=config.get("nms_free", True),
    )
    batch = min(batch, getattr(detector, "max_batch_size", batch) or batch)
    detector.warmup(2, [1, batch])

    def load(info):
        return info["id"], cv2.imread(str(image_dir / info["file_name"]))

    predictions = []
    pool = ThreadPoolExecutor(8)
    started = time.perf_counter()
    gpu_seconds = 0.0
    with GpuSampler(interval_s=0.1) as sampler:
        # Decode the next batch on CPU threads while the GPU runs this one.
        chunks = [images[i : i + batch] for i in range(0, len(images), batch)]
        pending = pool.map(load, chunks[0])
        for index in range(len(chunks)):
            loaded = list(pending)
            if index + 1 < len(chunks):
                pending = pool.map(load, chunks[index + 1])
            t0 = time.perf_counter()
            results = detector.predict_batch([frame for _, frame in loaded])
            gpu_seconds += time.perf_counter() - t0
            for (image_id, _), result in zip(loaded, results, strict=True):
                if isinstance(result, Exception):
                    raise result
                for d in result.detections:
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": to_coco[d.class_id],
                            "bbox": [
                                round(d.x1, 2),
                                round(d.y1, 2),
                                round(d.x2 - d.x1, 2),
                                round(d.y2 - d.y1, 2),
                            ],
                            "score": round(d.confidence, 5),
                        }
                    )
    wall = time.perf_counter() - started
    pool.shutdown()

    with contextlib.redirect_stdout(io.StringIO()):
        detections = truth.loadRes(predictions) if predictions else None
        evaluator = COCOeval(truth, detections, "bbox")
        evaluator.params.imgIds = [i["id"] for i in images]
        evaluator.evaluate()
        evaluator.accumulate()
    summary = io.StringIO()
    with contextlib.redirect_stdout(summary):
        evaluator.summarize()
    stats = evaluator.stats
    return {
        **config,
        "images": len(images),
        "batch": batch,
        "map50_95": round(float(stats[0]) * 100, 2),
        "map50": round(float(stats[1]) * 100, 2),
        "map75": round(float(stats[2]) * 100, 2),
        "map_small": round(float(stats[3]) * 100, 2),
        "map_medium": round(float(stats[4]) * 100, 2),
        "map_large": round(float(stats[5]) * 100, 2),
        "wall_seconds": round(wall, 1),
        "images_per_second_end_to_end": round(len(images) / wall, 1),
        "images_per_second_inference": round(len(images) / gpu_seconds, 1),
        "gpu": sampler.summary(),
        "coco_summary": summary.getvalue(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default=None, help="substring filter on label")
    parser.add_argument("--limit", type=int, default=None, help="evaluate the first N images")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        result = evaluate(json.loads(args.worker), args.limit, args.batch)
        print("RESULT " + json.dumps(result), flush=True)
        return 0

    if not (COCO / "annotations" / "instances_val2017.json").exists():
        raise SystemExit(f"COCO val2017 not found under {COCO}; see README 'Reproducing'")

    from benchlib import RESULTS

    log = RESULTS / "accuracy.jsonl"
    done = {}
    if log.exists():  # resumable: a long run can be interrupted and continued
        for line in log.read_text().splitlines():
            record = json.loads(line)
            if record.get("images") == (args.limit or 5000):
                done[record["label"]] = record

    records = []
    for config in MATRIX:
        if args.only and args.only not in config["label"]:
            continue
        if config["label"] in done:
            print(f"--- {config['label']}: cached")
            records.append(done[config["label"]])
            continue
        if not (MODELS / config["model"]).exists():
            print(f"--- {config['label']}: SKIPPED, {config['model']} not built")
            continue
        print(f"--- {config['label']}", flush=True)
        cmd = [sys.executable, __file__, "--worker", json.dumps(config), "--batch", str(args.batch)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        line = next((x[7:] for x in proc.stdout.splitlines() if x.startswith("RESULT ")), None)
        if proc.returncode != 0 or line is None:
            print(f"FAILED:\n{proc.stderr[-2000:]}")
            continue
        record = json.loads(line)
        print(
            f"    mAP50-95 {record['map50_95']:5.2f}  mAP50 {record['map50']:5.2f}  "
            f"{record['images_per_second_end_to_end']:7.1f} img/s end-to-end",
            flush=True,
        )
        records.append(record)
        append_jsonl(log, record)

    if not args.limit:
        write_results("accuracy", {"machine": machine(), "records": records})
    return 0


if __name__ == "__main__":
    sys.exit(main())
