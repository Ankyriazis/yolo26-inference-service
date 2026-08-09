"""Export YOLO26 weights to ONNX, and verify the export actually loads.

    python tools/export_onnx.py --model yolo26n.pt --imgsz 640

ONNX decouples deployment from PyTorch: the resulting graph runs under ONNX
Runtime on machines with no torch install and no CUDA, which is usually what
makes a model deployable somewhere other than a workstation.

The export is verified rather than assumed. `ultralytics` reports success as
soon as it has written a file, so this loads the result back through ONNX
Runtime and runs one inference before declaring it good.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument(
        "--dynamic", action="store_true", help="allow variable batch and image size"
    )
    parser.add_argument("--half", action="store_true", help="FP16 weights (GPU targets)")
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"exporting {args.model} -> ONNX (opset {args.opset}, imgsz {args.imgsz})")
    model = YOLO(args.model)
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        half=args.half,
    )

    path = Path(exported)
    if not path.exists():
        print("error: export reported success but produced no file", file=sys.stderr)
        return 1
    print(f"wrote {path} ({path.stat().st_size / 1_000_000:.1f} MB)")

    # Verify: a file on disk is not proof that the graph is loadable.
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nonnxruntime is not installed; skipping verification.")
        print("  pip install onnxruntime   (or onnxruntime-gpu)")
        return 0

    print("\nverifying with ONNX Runtime...")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()[0]
    print(f"  input   {inputs.name} {inputs.shape} {inputs.type}")

    shape = [d if isinstance(d, int) else 1 for d in inputs.shape]
    if len(shape) == 4:
        shape[2] = shape[2] if isinstance(shape[2], int) else args.imgsz
        shape[3] = shape[3] if isinstance(shape[3], int) else args.imgsz
    dummy = np.zeros(shape, dtype=np.float32)

    outputs = session.run(None, {inputs.name: dummy})
    print(f"  output  {[tuple(o.shape) for o in outputs]}")
    print("\nverified: the exported graph loads and runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
