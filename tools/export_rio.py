"""Export RiO-DETRv2 (oriented boxes, DOTA-v1.0) to TensorRT 11 - the YOLO26 recipe.

    # Once: RiO-DETRv2's source, and the extra modules its package imports,
    # installed to a side directory so they cannot touch the CUDA stack.
    git clone https://github.com/RicePasteM/RiO-DETRv2 ~/src/RiO-DETRv2
    uv pip install --target ~/.cache/yolo26-inference-service/rio-deps --no-deps \\
        timm huggingface_hub safetensors faster-coco-eval numba llvmlite defusedxml \\
        tensorboard tensorboard-data-server absl-py markdown werkzeug

    # Dynamic batch, tuned for 8 - the serving shape. m/l/x: --batch 8 (8 GB GPU)
    python tools/export_rio.py --rio-src ~/src/RiO-DETRv2 --size n --batch 16
    # Static batch 1, for CUDA-graph latency
    python tools/export_rio.py --rio-src ~/src/RiO-DETRv2 --size n --static --batch 1

RiO-DETRv2 (https://github.com/RicePasteM/RiO-DETRv2, AGPL-3.0 like this
repository) is a real-time detection transformer for oriented boxes. Nothing
of it is vendored here: the tool imports a checkout you point it at, and
downloads the published DOTA-v1.0 single-scale checkpoint, checked against
the authors' SHA256SUMS before it is loaded.

The engine is built exactly as tools/export.py builds YOLO26-OBB, so the two
compare like for like: one FP32 image input (B,3,S,S) in [0,1] RGB; FP16 baked
into the graph by ModelOpt AutoCast; the same strongly typed TensorRT build,
profile and JSON header. The model's own top-k post-processing is inside the
graph - like YOLO26's NMS-free head, it needs no NMS - and it outputs
labels (B,300), boxes (B,300,5) as cx cy w h in input pixels plus the angle in
radians, and scores (B,300).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
import urllib.request
from pathlib import Path

from benchlib import CACHE, MODELS, TrtEngine
from export import apply_precision, build_engine, stamp_metadata

HF = "https://huggingface.co/RicePasteM/RiO-DETRv2/resolve/main"
# RiO-DETRv2's DOTA class order (engine/data/dataset/dota_dataset.py)
DOTA_CLASSES = [
    "plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle",
    "large-vehicle", "ship", "tennis-court", "basketball-court", "storage-tank",
    "soccer-ball-field", "roundabout", "harbor", "swimming-pool", "helicopter",
]  # fmt: skip


def checkpoint(size: str, directory: Path) -> Path:
    """The published checkpoint, downloaded once and verified every time."""
    name = f"riov2_hgnetv2_{size}_dota_1_ss.pth"
    path = directory / name
    sums = directory / "riov2-SHA256SUMS"
    directory.mkdir(parents=True, exist_ok=True)
    if not sums.exists():
        urllib.request.urlretrieve(f"{HF}/SHA256SUMS", sums)
    expected = next(
        (line.split()[0] for line in sums.read_text().splitlines() if line.endswith(f"/{name}")),
        None,
    )
    if expected is None:
        raise SystemExit(f"{name} is not in the published SHA256SUMS")
    if not path.exists():
        print(f"[weights] downloading {name}")
        partial = path.with_suffix(".partial")
        urllib.request.urlretrieve(f"{HF}/dota_1_ss/{name}?download=true", partial)
        partial.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"{name}: SHA256 {digest} does not match the published {expected}")
    print(f"[weights] {name}: SHA256 matches the published checksum")
    return path


def load_model(rio_src: Path, rio_deps: Path, size: str, weights: Path):
    """Build RiO-DETRv2 from its own config and load the EMA weights."""
    import torch

    sys.path[:0] = [str(rio_src), str(rio_deps)]
    # Imported by its profiler for FLOP counts only; it pulls in all of `transformers`.
    sys.modules.setdefault("calflops", types.ModuleType("calflops"))
    sys.modules["calflops"].calculate_flops = None
    cwd = os.getcwd()
    os.chdir(rio_src)  # its YAML includes are relative to the checkout
    try:
        from engine.core import YAMLConfig

        cfg = YAMLConfig(f"configs/rio-dota_1_ss/riov2_hgnetv2_{size}_dota_1.yml")
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False  # the checkpoint carries the backbone
        state = torch.load(weights, map_location="cpu", weights_only=True)
        cfg.model.load_state_dict(state["ema"]["module"] if "ema" in state else state["model"])
        return cfg.model.deploy(), cfg.postprocessor.deploy()
    finally:
        os.chdir(cwd)


def wrap(model, post, imgsz: int):
    import torch

    class OneInput(torch.nn.Module):
        """images -> (labels, boxes, scores); boxes in input pixels."""

        def __init__(self) -> None:
            super().__init__()
            self.model, self.post = model, post

        def forward(self, images):
            sizes = torch.ones_like(images[:, 0, 0, :2]) * imgsz  # keeps the batch dim dynamic
            return self.post(self.model(images), sizes)

    return OneInput().eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rio-src", type=Path, required=True, help="RiO-DETRv2 checkout")
    parser.add_argument("--rio-deps", type=Path, default=CACHE / "rio-deps")
    parser.add_argument("--size", default="n", choices=list("nsmlx"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16, help="max batch, or the batch if static")
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--static", action="store_true", help="fixed batch; CUDA-graph capable")
    parser.add_argument("--workspace", type=float, default=4.0)
    parser.add_argument("--out-dir", type=Path, default=MODELS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import onnx
    import onnxslim
    import torch

    shape_tag = f"b{args.batch}" if args.static else f"dyn{args.batch}"
    stem = f"riov2-{args.size}-obb"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{stem}-fp16-{shape_tag}.engine"
    if target.exists() and not args.force:
        print(f"{target} exists; pass --force to rebuild")
        return 0

    started = time.perf_counter()
    weights = checkpoint(args.size, out_dir)
    net = wrap(
        *load_model(args.rio_src.resolve(), args.rio_deps.resolve(), args.size, weights), args.imgsz
    )
    example = torch.rand(args.batch if args.static else 2, 3, args.imgsz, args.imgsz)

    base = out_dir / f"{stem}-{shape_tag}.onnx"
    names = ["labels", "boxes", "scores"]
    print(f"[onnx] exporting {base.name}")
    # The TorchScript exporter, as RiO-DETRv2's own export script uses.
    torch.onnx.export(
        net,
        (example,),
        str(base),
        input_names=["images"],
        output_names=names,
        dynamic_axes=None if args.static else {n: {0: "batch"} for n in ["images", *names]},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.save(onnxslim.slim(str(base)), str(base))

    # The same FP16 step tools/export.py applies to YOLO26: ModelOpt AutoCast.
    shape = (args.batch if args.static else args.opt_batch, 3, args.imgsz, args.imgsz)
    graph = apply_precision(base, "fp16", shape, not args.static, None)
    metadata = {
        "description": f"RiO-DETRv2-{args.size.upper()} DOTA-v1.0 single-scale",
        "task": "obb",
        "imgsz": [args.imgsz, args.imgsz],
        "batch": args.batch,
        "names": dict(enumerate(DOTA_CLASSES)),
        "outputs": "labels (B,300); boxes (B,300,5) cx cy w h px + angle rad; scores (B,300)",
        "args": {
            "batch": args.batch,
            "dynamic": not args.static,
            "quantize": 16,
            "imgsz": args.imgsz,
        },
    }
    stamp_metadata(graph, metadata)  # the precision-baked ONNX travels on its own, as for YOLO26

    info = build_engine(
        graph,
        target,
        imgsz=args.imgsz,
        batch=args.batch,
        opt_batch=min(args.opt_batch, args.batch),
        static=args.static,
        workspace_gb=args.workspace,
        opt_level=3,
        metadata=metadata,
        timing_cache=out_dir / "timing.cache",
    )

    # A file on disk is not proof the graph runs: load it back and run it.
    engine = TrtEngine(target)
    shapes = {}
    for batch in [args.batch] if args.static else sorted({1, args.batch}):
        out = engine(torch.rand(batch, 3, args.imgsz, args.imgsz, device="cuda"))
        shapes[batch] = {name: list(out[name].shape) for name in names}
    info.update(
        {
            "verified_output_shapes": shapes,
            "artifact": target.name,
            "source": weights.name,
            "precision": "fp16",
            "static": args.static,
            "batch": args.batch,
            "opt_batch": None if args.static else min(args.opt_batch, args.batch),
            "imgsz": args.imgsz,
            "gpu": torch.cuda.get_device_name(0),
            "total_seconds": round(time.perf_counter() - started, 1),
        }
    )
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"\nverified: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
