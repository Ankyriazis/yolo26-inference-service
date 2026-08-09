"""Measure inference latency, so the README can quote numbers instead of adjectives.

    python tools/benchmark.py --device cuda:0 --runs 50
    python tools/benchmark.py --device cpu --runs 20

Reports percentiles rather than a mean: for an inference service the tail is
what a caller actually experiences. Warmup iterations are discarded, because
the first CUDA call in a process pays for kernel compilation and would
otherwise dominate the result.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision.detector import Detector  # noqa: E402


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = max(1, round(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--device", default=None, help="cuda:0, cpu, mps (default: auto)")
    parser.add_argument("--image", default=None, help="image to use (default: synthetic)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    detector = Detector(model=args.model, device=args.device, image_size=args.imgsz)

    if args.image:
        import cv2

        frame = cv2.imread(args.image)
        if frame is None:
            print(f"error: could not read {args.image}", file=sys.stderr)
            return 1
    else:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    print(f"model   {args.model}")
    print(f"device  {detector.device}")
    print(f"image   {frame.shape[1]}x{frame.shape[0]}"
          f"{' (synthetic)' if not args.image else ''}")
    print(f"load    {detector.load_seconds:.2f}s")

    for _ in range(args.warmup):
        detector.predict(frame)

    latencies = []
    started = time.perf_counter()
    for _ in range(args.runs):
        latencies.append(detector.predict(frame).inference_ms)
    wall = time.perf_counter() - started

    print(f"\nlatency over {args.runs} runs (warmup {args.warmup} discarded)")
    print(f"  p50   {percentile(latencies, 50):7.2f} ms")
    print(f"  p90   {percentile(latencies, 90):7.2f} ms")
    print(f"  p99   {percentile(latencies, 99):7.2f} ms")
    print(f"  mean  {statistics.mean(latencies):7.2f} ms")
    print(f"  min   {min(latencies):7.2f} ms")
    print(f"  max   {max(latencies):7.2f} ms")
    print(f"\nthroughput  {args.runs / wall:.1f} inferences/second")
    return 0


if __name__ == "__main__":
    sys.exit(main())
