"""Render the README's media from real detections: GIFs and stills.

    python tools/make_media.py              # everything, into docs/media/
    python tools/make_media.py --only obb

Every frame is drawn from a real engine's output - the TensorRT engines the
benchmarks measure - never from labels.

    coco  yolo26n, TensorRT FP16, through the service's GPU pipeline, on eight
          COCO val2017 photos licensed CC BY 2.0, confidence >= 0.5. The latency on each frame is
          measured while rendering: median of 30 runs at batch 1, pre-process,
          inference and post-process, the photo already decoded.
    obb   YOLO26-OBB-S against RiO-DETRv2-S on six DOTA tiles from dota8,
          side by side. Their latencies are the batch-1 medians in
          results/obb.json, so the GIF and the tables cannot disagree.
    api   the service's own POST /detect/annotated answer for data/bus.jpg,
          through the real app (lifespan, settings, batcher) in-process.

Credits and licences for the images: docs/media/CREDITS.md.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from benchlib import DATASETS, MODELS, RESULTS, ROOT, SAMPLE_IMAGE

MEDIA = ROOT / "docs" / "media"
COCO_IMAGES = DATASETS / "coco" / "images" / "val2017"
# COCO val2017, "Attribution License" (CC BY 2.0) only: no NC, ND or SA terms.
COCO_IDS = [184324, 190140, 192670, 274272, 495146, 538364, 139872, 119233]
SURFACE = (26, 26, 25)
INK, INK_MUTED = (240, 239, 236), (170, 168, 160)
# DOTAv1 classes (Ultralytics order), one colour each.
DOTA_NAMES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
    "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter",
    "roundabout", "soccer ball field", "swimming pool",
]  # fmt: skip
# Bright and well separated, so every class stays visible on green and grey ground.
DOTA_COLOURS = [
    (0, 150, 255), (255, 140, 0), (0, 230, 200), (255, 220, 0), (255, 100, 190),
    (180, 120, 255), (255, 60, 60), (230, 0, 230), (120, 255, 120), (255, 255, 255),
    (255, 200, 120), (0, 255, 255), (200, 255, 0), (160, 200, 255), (100, 100, 255),
]  # fmt: skip
RIO_TO_DOTAV1 = [0, 3, 8, 6, 10, 9, 1, 4, 5, 2, 13, 12, 7, 14, 11]
# Six of the eight dota8 tiles, in this order: the harbour, courts, airports and
# sports grounds read at GIF size; three near-identical dark tiles holding one
# baseball diamond each do not. The kept one is where the two models disagree.
OBB_TILES = [
    "P0861__1024__0___1648", "P1053__1024__0___90", "P1142__1024__0___824",
    "P1161__1024__3296___1648", "P1470__1024__3296___1648", "P1580__1024__824___824",
]  # fmt: skip


def font(size: int, bold: bool = False):
    from matplotlib import font_manager
    from PIL import ImageFont

    props = font_manager.FontProperties(
        family=["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        weight="semibold" if bold else "regular",
    )
    return ImageFont.truetype(font_manager.findfont(props), size)


def text_colour(rgb) -> tuple[int, int, int]:
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (20, 20, 20) if luminance > 150 else (255, 255, 255)


def save_gif(frames, path: Path, ms_per_frame: int) -> None:
    from PIL import Image

    palette = [
        f.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT) for f in frames
    ]
    palette[0].save(
        path, save_all=True, append_images=palette[1:], duration=ms_per_frame, loop=0, optimize=True
    )
    print(
        f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)"
    )


# ------------------------------------------------------------------ COCO


def draw_boxes(image_bgr, detections):
    """Axis-aligned boxes, the service's palette, an anti-aliased label pill."""
    from PIL import Image, ImageDraw

    from vision.imaging import colour_for

    canvas = Image.fromarray(image_bgr[..., ::-1].copy())
    draw = ImageDraw.Draw(canvas)
    label_font = font(13, bold=True)
    for d in sorted(detections, key=lambda d: d.confidence):
        b, g, r = colour_for(d.class_id)
        colour = (r, g, b)
        draw.rectangle([d.x1, d.y1, d.x2, d.y2], outline=colour, width=3)
        label = f"{d.class_name} {d.confidence:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=label_font)
        w, h = right - left + 10, bottom - top + 8
        y = d.y1 - h if d.y1 - h >= 0 else d.y1
        draw.rounded_rectangle([d.x1, y, d.x1 + w, y + h], radius=4, fill=colour)
        draw.text((d.x1 + 5, y + 4 - top), label, font=label_font, fill=text_colour(colour))
    return canvas


def caption(width: int, left: str, right: str, height: int = 40):
    from PIL import Image, ImageDraw

    bar = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(bar)
    draw.text((14, height // 2), left, font=font(15, bold=True), fill=INK, anchor="lm")
    draw.text((width - 14, height // 2), right, font=font(15), fill=INK_MUTED, anchor="rm")
    return bar


def make_coco() -> None:
    import cv2
    from PIL import Image

    from vision.detector import GpuDetector

    engine = MODELS / "yolo26n-fp16-dyn32.engine"
    detector = GpuDetector(model=str(engine), confidence=0.5)
    width, height = 640, 480
    frames, credits = [], []
    for image_id in COCO_IDS:
        name = f"{image_id:012d}.jpg"
        image = cv2.imread(str(COCO_IMAGES / name))
        for _ in range(5):
            detector.predict_batch([image])
        timings = []
        for _ in range(30):
            started = time.perf_counter()
            result = detector.predict_batch([image])[0]
            timings.append((time.perf_counter() - started) * 1000)
        drawn = draw_boxes(image, result.detections)
        scale = min(width / drawn.width, height / drawn.height)
        drawn = drawn.resize(
            (round(drawn.width * scale), round(drawn.height * scale)), Image.LANCZOS
        )
        frame = Image.new("RGB", (width, height + 40), SURFACE)
        frame.paste(drawn, ((width - drawn.width) // 2, (height - drawn.height) // 2))
        frame.paste(
            caption(
                width,
                "yolo26n · TensorRT FP16 · RTX 4060",
                f"{len(result.detections)} objects · {statistics.median(timings):.1f} ms",
            ),
            (0, height),
        )
        frames.append(frame)
        credits.append(name)
        print(f"  {name}: {len(result.detections)} objects, {statistics.median(timings):.2f} ms")
    MEDIA.mkdir(parents=True, exist_ok=True)
    save_gif(frames, MEDIA / "coco-detections.gif", 1800)


# ------------------------------------------------------------------ OBB


def obb_latency(family: str, size: str) -> float:
    speed = json.loads((RESULTS / "obb.json").read_text())["speed"]
    values = [
        e["latency_ms"]["p50"]
        for r in speed
        if r["family"] == family and r["size"] == size and not r["static"]
        for e in r["results"]
        if e["mode"] == "plain" and e["batch"] == 1
    ]
    return statistics.median(values)


def make_obb(size: str = "s", conf: float = 0.4) -> None:
    import cv2
    import numpy as np
    import torch
    from benchlib import TrtEngine
    from PIL import Image, ImageDraw
    from ultralytics.utils.ops import xywhr2xyxyxyxy

    found = {p.stem: p for p in (DATASETS / "dota8" / "images").rglob("*.jpg")}
    tiles = [found[name] for name in OBB_TILES]
    models = [
        ("YOLO26-OBB", "yolo26-obb", TrtEngine(MODELS / f"yolo26{size}-obb-fp16-dyn16.engine")),
        ("RiO-DETRv2", "riov2", TrtEngine(MODELS / f"riov2-{size}-obb-fp16-dyn16.engine")),
    ]
    panel, gap, header, footer = 480, 12, 40, 34
    width = 2 * panel + gap
    frames = []
    for tile in tiles:
        image = cv2.imread(str(tile))
        x = torch.from_numpy(
            np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255
        ).cuda()
        frame = Image.new("RGB", (width, header + panel + footer), SURFACE)
        draw_frame = ImageDraw.Draw(frame)
        seen = set()
        for column, (label, family, engine) in enumerate(models):
            out = engine(x)
            if family == "riov2":
                labels = torch.tensor(RIO_TO_DOTAV1)[out["labels"][0].long().cpu()]
                boxes, scores = out["boxes"][0].float().cpu(), out["scores"][0].float().cpu()
            else:
                raw = next(iter(out.values()))[0].float().cpu()  # x y w h score class angle
                labels, boxes, scores = raw[:, 5].long(), raw[:, [0, 1, 2, 3, 6]], raw[:, 4]
            keep = scores >= conf
            labels, boxes = labels[keep], boxes[keep]
            corners = xywhr2xyxyxyxy(boxes).numpy() * (panel / image.shape[1])
            view = Image.fromarray(
                cv2.resize(image, (panel, panel), interpolation=cv2.INTER_AREA)[..., ::-1]
            )
            draw = ImageDraw.Draw(view)
            for polygon, cls in zip(corners, labels.tolist(), strict=True):
                draw.polygon([tuple(p) for p in polygon], outline=DOTA_COLOURS[cls], width=2)
                seen.add(cls)
            left = column * (panel + gap)
            frame.paste(view, (left, header))
            draw_frame.text(
                (left + 4, header // 2),
                f"{label}-{size.upper()}",
                font=font(16, bold=True),
                fill=INK,
                anchor="lm",
            )
            draw_frame.text(
                (left + panel - 4, header // 2),
                f"{len(labels)} objects · {obb_latency(family, size):.2f} ms",
                font=font(15),
                fill=INK_MUTED,
                anchor="rm",
            )
        # legend: the classes either model found on this tile
        cursor, y = 6, header + panel + footer // 2
        legend_font = font(13)
        for cls in sorted(seen):
            draw_frame.rounded_rectangle(
                [cursor, y - 6, cursor + 12, y + 6], 2, fill=DOTA_COLOURS[cls]
            )
            draw_frame.text(
                (cursor + 17, y), DOTA_NAMES[cls], font=legend_font, fill=INK_MUTED, anchor="lm"
            )
            cursor += 17 + int(draw_frame.textlength(DOTA_NAMES[cls], font=legend_font)) + 16
        frames.append(frame)
        print(f"  {tile.name}: classes {sorted(DOTA_NAMES[c] for c in seen)}")
    MEDIA.mkdir(parents=True, exist_ok=True)
    save_gif(frames, MEDIA / "obb-detections.gif", 2200)


# ------------------------------------------------------------------ API


def make_api() -> None:
    os.environ["VISION_MODEL"] = str(MODELS / "yolo26n-fp16-dyn32.engine")
    os.environ["VISION_PIPELINE"] = "gpu"
    from starlette.testclient import TestClient

    from vision.service import app

    with TestClient(app) as client:
        response = client.post(
            "/detect/annotated",
            files={"file": ("bus.jpg", SAMPLE_IMAGE.read_bytes(), "image/jpeg")},
        )
        response.raise_for_status()
    out = ROOT / "docs" / "detection-example.jpg"
    out.write_bytes(response.content)
    print(
        f"wrote {out.relative_to(ROOT)} ({len(response.content) / 1e3:.0f} KB), "
        "from /detect/annotated"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["coco", "obb", "api"], action="append")
    args = parser.parse_args()
    jobs = {"coco": make_coco, "obb": make_obb, "api": make_api}
    for name in args.only or list(jobs):
        print(f"=== {name}")
        jobs[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
