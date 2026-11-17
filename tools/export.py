"""Export YOLO26 to ONNX or TensorRT, with explicit precision and batch profiles.

    # TensorRT, FP16, dynamic batch 1..32 tuned for 8 - the serving default
    python tools/export.py --model yolo26n.pt --format engine --precision fp16

    # Latency-optimised: static batch 1, replayed as a CUDA graph
    python tools/export.py --model yolo26n.pt --format engine --precision fp16 --static --batch 1

    # INT8, calibrated on images that are not in the evaluation set
    python tools/export.py --model yolo26n.pt --format engine --precision int8 --calib-dir coco128

    # Portable ONNX for ONNX Runtime
    python tools/export.py --model yolo26n.pt --format onnx

    # Split build: bake precision into ONNX once (CI, no GPU needed) ...
    python tools/export.py --model yolo26n.pt --format onnx --precision fp16
    # ... then compile it on each GPU node with TensorRT alone (no ModelOpt)
    python tools/export.py --onnx yolo26n-fp16-dyn32.onnx --engine-cache /var/cache/engines

Why not `model.export(format="engine")`? Three reasons, each a deployment
concern rather than a preference:

1. Optimisation profile. Ultralytics' dynamic engines accept up to 2x the
   image size and tune kernels for the largest batch. Serving at a fixed 640
   wastes that range: the context reserves activation memory for the biggest
   shape it might see (1.4 GB for yolo26n at batch 32), which caps how many
   replicas fit on one GPU. Here the spatial size is fixed and kernels are
   tuned for the batch the dynamic batcher actually produces.
2. Head. Ultralytics 8.4 exports the one-to-many head plus external NMS by
   default. This exports YOLO26's NMS-free one-to-one head unless asked not to.
3. Provenance. Each engine gets a JSON sidecar recording build time, device
   memory, TensorRT version and GPU - the figures capacity planning needs.

TensorRT 11 is strongly typed: the FP16/INT8 builder flags and the INT8
calibrator are gone, and precision must be baked into the ONNX graph before
the build. That step uses NVIDIA ModelOpt through Ultralytics' wrapper (FP16
AutoCast, INT8 explicit Q/DQ with max calibration), so the quantisation
recipe - which layers stay in float - is the maintained upstream one.

Every artifact is verified by loading it back and running inference before
the tool reports success. A file on disk is not proof the graph runs.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Never let Ultralytics pip-install at export time; see src/vision/__init__.py.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PRECISION_CODE = {"fp32": None, "fp16": 16, "int8": 8}
PRECISION_NAME = {None: "fp32", 16: "fp16", 8: "int8"}


def artifact_name(stem: str, fmt: str, precision: str, static: bool, batch: int) -> str:
    shape = f"b{batch}" if static else f"dyn{batch}"
    suffix = ".engine" if fmt == "engine" else ".onnx"
    return f"{stem}-{precision}-{shape}{suffix}"


def export_onnx(
    weights: Path, out: Path, imgsz: int, batch: int, static: bool, nms_free: bool
) -> Path:
    from ultralytics import YOLO

    print(
        f"[onnx] exporting {weights.name} (batch {batch}, {'static' if static else 'dynamic'}, "
        f"{'NMS-free' if nms_free else 'one-to-many + NMS'} head)"
    )
    exported = Path(
        YOLO(str(weights)).export(
            format="onnx",
            imgsz=imgsz,
            batch=batch,
            dynamic=not static,
            nms=False if nms_free else None,
            simplify=True,
            device=0 if _cuda() else "cpu",
            verbose=False,
        )
    )
    shutil.move(exported, out)
    return out


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def read_metadata(onnx_file: Path) -> dict:
    from ultralytics.nn.backends.base import BaseBackend

    meta = BaseBackend.read_metadata(onnx_file)
    for key in ("stride", "batch", "channels", "imgsz", "names", "args", "end2end"):
        if isinstance(meta.get(key), str):
            with contextlib.suppress(ValueError, SyntaxError):
                meta[key] = ast.literal_eval(meta[key])
    return meta


def stamp_metadata(onnx_file: Path, metadata: dict) -> None:
    """ModelOpt drops ONNX metadata; put it back so the graph is self-describing.

    That is what lets a precision-baked ONNX travel on its own - from a CI job
    with no GPU to a node that compiles it with nothing but TensorRT.
    """
    import onnx

    model = onnx.load(str(onnx_file))
    del model.metadata_props[:]
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, str(value)
    onnx.save(model, str(onnx_file))


def engine_cache_dir(root: Path) -> Path:
    """An engine is valid for one GPU model and one TensorRT version: key on both."""
    import tensorrt as trt
    import torch

    gpu = re.sub(r"[^A-Za-z0-9.]+", "-", torch.cuda.get_device_name(0)).strip("-")
    return root / gpu / f"tensorrt-{trt.__version__}"


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    gain = min(size / h, size / w)
    nh, nw = round(h * gain), round(w * gain)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = round((size - nh) / 2 - 0.1), round((size - nw) / 2 - 0.1)
    canvas[top : top + nh, left : left + nw] = cv2.resize(
        image, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    return canvas[..., ::-1].transpose(2, 0, 1)  # BGR HWC -> RGB CHW


def calibration_files(calib_dir: Path, limit: int) -> list[Path]:
    files = sorted(p for p in calib_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise SystemExit(f"no calibration images under {calib_dir}")
    return files[:limit]


def calibration_batches(files: list[Path], size: int, batch: int = 16):
    """Yields what Ultralytics' ModelOpt wrapper expects: {"img": uint8 BCHW}."""
    import cv2
    import torch

    print(f"[int8] calibrating on {len(files)} images from {files[0].parent}")
    for start in range(0, len(files), batch):
        chunk = [letterbox(cv2.imread(str(f)), size) for f in files[start : start + batch]]
        yield {"img": torch.from_numpy(np.ascontiguousarray(np.stack(chunk)))}


def apply_precision(onnx_file: Path, precision: str, shape, dynamic: bool, calib) -> Path:
    if precision == "fp32":
        return onnx_file
    from ultralytics.utils.export.engine import modelopt_quantize_onnx

    print(f"[{precision}] baking precision into the graph with NVIDIA ModelOpt")
    if precision == "fp16":
        # AutoCast calibrates on one image repeated across the batch, so a bigger
        # batch adds no information, only memory: ModelOpt holds every
        # intermediate tensor of its reference run on the CPU - 26 GB for
        # yolo26l-obb at batch 8 and 1024 px. Batch 1 picks the same nodes.
        shape = (1, *shape[1:])
    return Path(
        modelopt_quantize_onnx(
            str(onnx_file), PRECISION_CODE[precision], calib, shape, dynamic, prefix="[modelopt]"
        )
    )


def build_engine(
    onnx_file: Path,
    engine_file: Path,
    *,
    imgsz: int,
    batch: int,
    opt_batch: int,
    static: bool,
    workspace_gb: float,
    opt_level: int,
    metadata: dict,
    timing_cache: Path,
) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TensorRT 11: explicit batch, strongly typed
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_file}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.builder_optimization_level = opt_level

    # Kernel timings are reused across builds on the same GPU and TensorRT
    # version, which turns a rebuild from minutes into seconds.
    cache_bytes = timing_cache.read_bytes() if timing_cache.exists() else b""
    cache = config.create_timing_cache(cache_bytes)
    config.set_timing_cache(cache, ignore_mismatch=False)

    name = network.get_input(0).name
    if not static:
        profile = builder.create_optimization_profile()
        profile.set_shape(
            name,
            min=(1, 3, imgsz, imgsz),
            opt=(opt_batch, 3, imgsz, imgsz),
            max=(batch, 3, imgsz, imgsz),
        )
        config.add_optimization_profile(profile)

    shape = f"={batch}" if static else f"1..{batch} (opt {opt_batch})"
    print(
        f"[trt] building {engine_file.name}: batch {shape}, {imgsz}x{imgsz}, opt level {opt_level}"
    )
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed; see the log above")
    timing_cache.write_bytes(memoryview(config.get_timing_cache().serialize()))

    header = json.dumps(metadata).encode()
    with engine_file.open("wb") as handle:
        handle.write(len(header).to_bytes(4, "little", signed=True))
        handle.write(header)
        handle.write(serialized)

    engine = trt.Runtime(logger).deserialize_cuda_engine(bytes(serialized))
    memory = engine.device_memory_size_v2
    return {
        "build_seconds": round(build_seconds, 1),
        "activation_memory_mb": round(memory / 2**20, 1),
        "engine_mb": round(engine_file.stat().st_size / 1e6, 2),
        "tensorrt": trt.__version__,
        "opt_level": opt_level,
    }


def verify(path: Path, imgsz: int, batch: int, static: bool) -> dict:
    """Load the artifact back and run it. Returns output shapes."""
    import torch
    from ultralytics.nn.autobackend import AutoBackend

    device = torch.device("cuda:0" if _cuda() else "cpu")
    model = AutoBackend(str(path), device=device, verbose=False)
    sizes = [batch] if static else sorted({1, batch})
    shapes = {}
    for size in sizes:
        x = torch.rand(size, 3, imgsz, imgsz, device=device)
        y = model(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        shapes[size] = list(y.shape)
    return {"verified_output_shapes": shapes, "end2end": bool(model.end2end)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--format", choices=["engine", "onnx"], default="engine")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch", type=int, default=32, help="max batch (dynamic) or the batch (static)"
    )
    parser.add_argument("--opt-batch", type=int, default=8, help="batch TensorRT tunes kernels for")
    parser.add_argument(
        "--static", action="store_true", help="fixed batch; enables CUDA-graph replay"
    )
    parser.add_argument("--one-to-many", action="store_true", help="export the NMS head instead")
    parser.add_argument("--calib-dir", type=Path, default=None, help="INT8 calibration images")
    parser.add_argument("--calib-images", type=int, default=512)
    parser.add_argument("--workspace", type=float, default=4.0, help="builder workspace, GiB")
    parser.add_argument("--opt-level", type=int, default=3, choices=range(6))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="rebuild even if the artifact exists")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="compile this precision-baked ONNX (from --format onnx) with TensorRT alone",
    )
    parser.add_argument(
        "--engine-cache", type=Path, default=None, help="with --onnx: DIR/<gpu>/<tensorrt>/"
    )
    parser.add_argument("--link-to", type=Path, default=None, help="also copy the engine here")
    args = parser.parse_args()

    if args.onnx:
        return compile_onnx(args)

    weights = Path(args.model).resolve()
    if not weights.exists():  # let Ultralytics download a named release
        from ultralytics import YOLO

        YOLO(args.model)
        weights = Path(args.model).resolve()
    out_dir = (args.out_dir or weights.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    nms_free = not args.one_to_many
    stem = weights.stem + ("" if nms_free else "-o2m")
    target = out_dir / artifact_name(stem, args.format, args.precision, args.static, args.batch)
    if target.exists() and not args.force:
        print(f"{target} exists; pass --force to rebuild")
        return 0

    started = time.perf_counter()
    base = export_onnx(
        weights,
        out_dir / f"{stem}-{'b' if args.static else 'dyn'}{args.batch}.onnx",
        args.imgsz,
        args.batch,
        args.static,
        nms_free,
    )
    metadata = read_metadata(base)
    shape = (args.opt_batch if not args.static else args.batch, 3, args.imgsz, args.imgsz)

    calib, calib_count = None, None
    if args.precision == "int8":
        if args.calib_dir is None:
            raise SystemExit("--precision int8 needs --calib-dir (images NOT in your eval set)")
        files = calibration_files(args.calib_dir, args.calib_images)
        calib, calib_count = calibration_batches(files, args.imgsz), len(files)
    graph = apply_precision(base, args.precision, shape, not args.static, calib)

    metadata["batch"] = args.batch
    metadata["imgsz"] = [args.imgsz, args.imgsz]
    metadata["end2end"] = nms_free
    metadata["args"] = {
        **metadata.get("args", {}),
        "batch": args.batch,
        "dynamic": not args.static,
        "quantize": PRECISION_CODE[args.precision],
        "nms": False if nms_free else None,
        "imgsz": args.imgsz,
    }

    stamp_metadata(graph, metadata)
    if args.format == "onnx":
        if graph != target:
            shutil.copyfile(graph, target)
        info = {"engine_mb": round(target.stat().st_size / 1e6, 2)}
    else:
        info = build_engine(
            graph,
            target,
            imgsz=args.imgsz,
            batch=args.batch,
            opt_batch=args.opt_batch,
            static=args.static,
            workspace_gb=args.workspace,
            opt_level=args.opt_level,
            metadata=metadata,
            timing_cache=out_dir / "timing.cache",
        )

    info.update(verify(target, args.imgsz, args.batch, args.static))

    import torch

    info.update(
        {
            "artifact": target.name,
            "source": weights.name,
            "format": args.format,
            "precision": args.precision,
            "static": args.static,
            "batch": args.batch,
            "opt_batch": None if args.static else args.opt_batch,
            "imgsz": args.imgsz,
            "nms_free": nms_free,
            "calibration_images": calib_count,
            "gpu": torch.cuda.get_device_name(0),
            "total_seconds": round(time.perf_counter() - started, 1),
        }
    )
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"\nverified: {target}")
    return 0


def compile_onnx(args: argparse.Namespace) -> int:
    """Build an engine from an ONNX whose precision is already in the graph.

    TensorRT 11 is strongly typed, so the graph alone decides precision: this
    step needs TensorRT and nothing else - no ModelOpt, no calibration data.
    It is the half of the build that has to run on the target GPU.
    """
    source = args.onnx.resolve()
    metadata = read_metadata(source)
    if not metadata.get("args"):
        raise SystemExit(f"{source} carries no export metadata; produce it with --format onnx")
    meta_args = metadata["args"]
    static = not meta_args.get("dynamic", True)
    batch = int(metadata.get("batch", args.batch))
    imgsz = int((metadata.get("imgsz") or [args.imgsz])[0])

    out_dir = engine_cache_dir(args.engine_cache) if args.engine_cache else args.out_dir
    out_dir = (out_dir or source.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{source.stem}.engine"
    with build_lock(target) as must_build:
        if must_build or args.force:
            compile_into(args, source, target, metadata, imgsz, batch, static, out_dir)
        else:
            print(f"[cache] {target} already built for this GPU and TensorRT; reusing it")
    if args.link_to:
        args.link_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, args.link_to)
        print(f"[link] {args.link_to}")
    print(f"verified: {target}")
    return 0


@contextlib.contextmanager
def build_lock(target: Path, timeout_s: float = 900):
    """Several pods on one node may start at once and all find no engine.
    One takes the lock and builds; the rest wait, then reuse its result.
    Yields True when the caller should build."""
    lock = target.with_name(target.name + ".lock")
    deadline = time.monotonic() + timeout_s
    while True:
        if target.exists():
            yield False
            return
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            # A crashed builder leaves its lock behind; treat old locks as stale.
            with contextlib.suppress(FileNotFoundError):
                if time.time() - lock.stat().st_mtime > timeout_s:
                    lock.unlink()
            if time.monotonic() > deadline:
                raise TimeoutError(f"gave up waiting for another build of {target.name}") from None
            print(f"[lock] another process is building {target.name}; waiting", flush=True)
            time.sleep(5)
    try:
        os.close(fd)
        yield not target.exists()
    finally:
        lock.unlink(missing_ok=True)


def compile_into(args, source, target, metadata, imgsz, batch, static, out_dir) -> None:
    """Build to a temporary name and rename into place: a reader never sees a
    half-written engine, and the rename is atomic on one filesystem."""
    partial = target.with_name(f"{target.stem}.partial-{os.getpid()}{target.suffix}")
    started = time.perf_counter()
    try:
        info = build_engine(
            source,
            partial,
            imgsz=imgsz,
            batch=batch,
            opt_batch=min(args.opt_batch, batch),
            static=static,
            workspace_gb=args.workspace,
            opt_level=args.opt_level,
            metadata=metadata,
            timing_cache=out_dir / "timing.cache",
        )
        info.update(verify(partial, imgsz, batch, static))
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    info.update(
        {
            "artifact": target.name,
            "source": source.name,
            "format": "engine",
            "precision": PRECISION_NAME.get(metadata["args"].get("quantize"), "fp32"),
            "static": static,
            "batch": batch,
            "imgsz": imgsz,
            "total_seconds": round(time.perf_counter() - started, 1),
        }
    )
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    sys.exit(main())
