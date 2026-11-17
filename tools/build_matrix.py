"""Build every artifact the benchmark suite measures.

    python tools/build_matrix.py --models-dir ~/.cache/yolo26-inference-service/models \
        --calib-dir ~/.cache/yolo26-inference-service/datasets/coco128/images

Engines are specific to the GPU model and TensorRT version that built them, so
this runs on the target hardware - in CI on the same SKU, or as an init step at
deploy time. ONNX is the portable artifact you ship; the engine is a cache.

Sequential on purpose: two builds at once fight over the GPU and both of their
kernel-timing measurements come out wrong.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / "tools" / "export.py"

# (weights, format, precision, extra args)
MATRIX: list[tuple[str, str, str, list[str]]] = [
    # Runtime x precision, on the smallest model
    ("yolo26n.pt", "onnx", "fp32", []),
    ("yolo26n.pt", "engine", "fp32", []),
    ("yolo26n.pt", "engine", "fp16", []),
    ("yolo26n.pt", "engine", "int8", []),
    ("yolo26n.pt", "engine", "fp16", ["--static", "--batch", "1"]),
    ("yolo26n.pt", "engine", "fp16", ["--one-to-many"]),
    # Model scaling: every size, the two serving precisions
    *[(f"yolo26{s}.pt", "engine", p, []) for s in "smlx" for p in ("fp16", "int8")],
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--only", default=None, help="substring filter on weights name")
    args = parser.parse_args()

    failures = []
    for weights, fmt, precision, extra in MATRIX:
        if args.only and args.only not in weights:
            continue
        cmd = [
            sys.executable,
            str(EXPORT),
            "--model",
            str(args.models_dir / weights),
            "--format",
            fmt,
            "--precision",
            precision,
            *extra,
        ]
        if precision == "int8":
            cmd += ["--calib-dir", str(args.calib_dir)]
        label = f"{weights} {fmt} {precision} {' '.join(extra)}".strip()
        print(f"\n=== {label}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(cmd, check=False)
        status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
        print(f"=== {label}: {status} in {time.perf_counter() - started:.0f}s", flush=True)
        if result.returncode != 0:
            failures.append(label)

    if failures:
        print("\nfailed:\n  " + "\n  ".join(failures))
        return 1
    print("\nall artifacts built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
