"""Turn results/*.json into charts and docs/BENCHMARKS.md.

    python tools/report.py

Every chart is rendered twice - light and dark, each validated against its
own surface - and embedded with <picture> so GitHub serves the one matching
the reader's theme. Every chart has its data table beside it: the table is
the accessible view, and the place to copy numbers from.
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchlib import RESULTS, ROOT
from matplotlib.ticker import FuncFormatter, NullFormatter

DOCS = ROOT / "docs"
CHARTS = DOCS / "results"

# Validated categorical order (dataviz reference palette); fixed, never cycled.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": [
            "#2a78d6",
            "#eb6834",
            "#1baf7a",
            "#eda100",
            "#e87ba4",
            "#008300",
            "#6250d6",
            "#e34948",
        ],
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": [
            "#3987e5",
            "#d95926",
            "#199e70",
            "#c98500",
            "#d55181",
            "#0ca30c",
            "#9085e9",
            "#e66767",
        ],
    },
}
# Secondary encoding: series identity is never carried by colour alone.
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


# ------------------------------------------------------------------ chart scaffolding


def axes(theme: dict, title: str, subtitle: str, size=(9, 4.8)):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT})
    fig, ax = plt.subplots(figsize=size, dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0, pad=6)
    ax.grid(True, axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(
        0.012, 0.965, title, fontsize=13, fontweight="semibold", color=theme["primary"], va="top"
    )
    fig.text(0.012, 0.905, subtitle, fontsize=9.5, color=theme["secondary"], va="top")
    return fig, ax


def label_axis(ax, theme, xlabel=None, ylabel=None):
    if xlabel:
        ax.set_xlabel(xlabel, color=theme["secondary"], fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=theme["secondary"], fontsize=9.5)


def legend(ax, theme, **kwargs):
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=theme["secondary"], **kwargs)
    return leg


def legend_above(ax, theme, ncol=3):
    """A legend row between the subtitle and the plot, never over the data."""
    return ax.legend(
        frameon=False,
        fontsize=8.5,
        labelcolor=theme["secondary"],
        ncol=ncol,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
        handlelength=2.6,
        columnspacing=1.6,
    )


def save(fig, name: str, mode: str, top: float = 0.80) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=top, left=0.09, right=0.97, bottom=0.14)
    fig.savefig(CHARTS / f"{name}.{mode}.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def picture(name: str, alt: str) -> str:
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="results/{name}.dark.png">\n'
        f'  <img alt="{alt}" src="results/{name}.light.png">\n'
        "</picture>"
    )


def thousands(value, _pos=None):
    return f"{value / 1000:.0f}K" if value >= 10_000 else f"{value:,.0f}"


def line(ax, theme, slot, xs, ys, label, end_label=True):
    color = theme["series"][slot]
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=2,
        marker=MARKERS[slot],
        markersize=6.5,
        markeredgecolor=theme["surface"],
        markeredgewidth=1.6,
        label=label,
        zorder=3,
    )
    if end_label and xs:
        ax.annotate(
            label,
            (xs[-1], ys[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            fontsize=8,
            color=theme["secondary"],
            va="center",
        )


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def fmt(value, digits=1):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:,.{digits}f}"


# ------------------------------------------------------------------ sections


def by_label(records):
    return {r["label"]: r for r in records if "error" not in r}


def batch_entry(record, batch):
    return next((b for b in record.get("batches", []) if b["batch"] == batch), None)


RUNTIME_ORDER = [
    "PyTorch FP32",
    "PyTorch FP16",
    "ONNX Runtime CUDA FP32",
    "TensorRT FP32",
    "TensorRT FP16",
    "TensorRT INT8",
]


def section_runtime(pipeline: dict) -> str:
    records = by_label(pipeline["records"])
    runtime = [records[k] for k in RUNTIME_ORDER if k in records]
    if not runtime:
        return ""

    # Chart: throughput vs batch size, one line per runtime
    for mode, theme in THEMES.items():
        fig, ax = axes(
            theme,
            "Throughput as the batch grows",
            "yolo26n · 810×1080 photo · GPU pipeline · RTX 4060 · images per second",
        )
        for slot, record in enumerate(runtime):
            xs = [b["batch"] for b in record["batches"]]
            ys = [b["images_per_second"] for b in record["batches"]]
            line(ax, theme, slot, xs, ys, record["label"], end_label=False)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
        ax.set_ylim(bottom=0)
        label_axis(ax, theme, "batch size (images per forward pass)", "images / second")
        legend_above(ax, theme)
        save(fig, "batch-throughput", mode, top=0.72)

    static = records.get("TensorRT FP16 static b1 (CUDA graph)")
    rows = []
    for record in runtime + ([static] if static else []):
        b1, b8, b32 = (batch_entry(record, n) for n in (1, 8, 32))
        best = max(record["batches"], key=lambda b: b["images_per_second"])
        rows.append(
            [
                record["label"],
                fmt(b1["latency_ms"]["p50"], 2) if b1 else "–",
                fmt(b1["latency_ms"]["p99"], 2) if b1 else "–",
                fmt(b1["images_per_second"], 0) if b1 else "–",
                fmt(b8["images_per_second"], 0) if b8 else "–",
                fmt(b32["images_per_second"], 0) if b32 else "–",
                f"{fmt(best['images_per_second'], 0)} (b{best['batch']})",
                fmt(record.get("vram_added_mb"), 0),
                fmt(record.get("first_call_ms"), 0),
            ]
        )
    return "\n\n".join(
        [
            "## Runtime and precision",
            "Same model (yolo26n), same photograph, same GPU pipeline; only the runtime and numeric "
            "precision change. Latency is one `predict_batch` call, measured on the host clock, "
            "with the image already decoded.",
            picture(
                "batch-throughput",
                "Line chart of images per second against batch size for six "
                "runtime and precision combinations; TensorRT lines rise highest.",
            ),
            table(
                [
                    "Runtime",
                    "b1 p50 ms",
                    "b1 p99 ms",
                    "b1 img/s",
                    "b8 img/s",
                    "b32 img/s",
                    "Peak img/s",
                    "VRAM added MB",
                    "First call ms (no warmup)",
                ],
                rows,
            ),
        ]
    )


def section_stages(pipeline: dict) -> str:
    records = by_label(pipeline["records"])
    wanted = [
        ("Ultralytics predictor, PyTorch FP32", "Ultralytics · PyTorch FP32"),
        ("Ultralytics predictor, TensorRT FP16", "Ultralytics · TensorRT FP16"),
        ("GPU pipeline, TensorRT FP16", "GPU pipeline · TensorRT FP16"),
        ("GPU pipeline + nvJPEG decode, TensorRT FP16", "GPU pipeline + nvJPEG · TRT FP16"),
        (
            "GPU pipeline, TensorRT FP16, one-to-many head + NMS",
            "GPU pipeline · TRT FP16 · NMS head",
        ),
    ]
    present = [(records[k], short) for k, short in wanted if k in records]
    if not present:
        return ""
    batch = 8
    stage_map = {
        "decode / upload": ("upload_decode",),
        "preprocess": ("preprocess",),
        "inference": ("inference",),
        "postprocess": ("postprocess",),
    }
    bars = []
    for record, short in present:
        entry = batch_entry(record, batch)
        if not entry:
            continue
        stages = entry["stages_ms"]
        per_image = {
            name: sum(stages.get(k, 0.0) for k in keys) / batch for name, keys in stage_map.items()
        }
        wall = entry["latency_ms"]["p50"] / batch
        bars.append((short, per_image, wall, entry["images_per_second"]))

    for mode, theme in THEMES.items():
        fig, ax = axes(
            theme,
            "Where a batch's time goes",
            f"yolo26n · batch of {batch} · milliseconds per image, by pipeline stage",
            size=(9, 4.2),
        )
        ax.grid(True, axis="x", color=theme["grid"], linewidth=0.8)
        ax.grid(False, axis="y")
        names = list(stage_map)
        for row, (_short, per_image, _wall, _ips) in enumerate(reversed(bars)):
            left = 0.0
            for slot, name in enumerate(names):
                width = per_image[name]
                if width <= 0:
                    continue
                ax.barh(
                    row,
                    width,
                    left=left,
                    height=0.56,
                    color=theme["series"][slot],
                    edgecolor=theme["surface"],
                    linewidth=2,
                    label=name if row == 0 else None,
                )
                left += width
            ax.annotate(
                f"{left:.2f} ms",
                (left, row),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=theme["secondary"],
            )
        ax.set_yticks(range(len(bars)), [b[0] for b in reversed(bars)], color=theme["secondary"])
        ax.tick_params(axis="y", labelsize=8.5)
        ax.spines["bottom"].set_visible(False)
        label_axis(ax, theme, "milliseconds per image")
        legend(ax, theme, loc="lower right", ncol=4, bbox_to_anchor=(1.0, 1.0))
        fig.subplots_adjust(left=0.30)
        CHARTS.mkdir(parents=True, exist_ok=True)
        fig.subplots_adjust(top=0.78, right=0.95, bottom=0.14)
        fig.savefig(CHARTS / f"stages.{mode}.png", facecolor=fig.get_facecolor())
        plt.close(fig)

    rows = [
        [short, *(fmt(per_image[n], 3) for n in stage_map), fmt(wall, 3), fmt(ips, 0)]
        for short, per_image, wall, ips in bars
    ]
    return "\n\n".join(
        [
            "## Where the time goes",
            "Per-stage device time from CUDA events (GPU pipeline) or the predictor's own timers "
            "(Ultralytics), divided by the batch size. *Wall* is the host-measured p50 per image, "
            "which includes host work the stage timers do not see.",
            picture(
                "stages",
                "Stacked bars of milliseconds per image split into decode, "
                "preprocess, inference and postprocess for five pipeline variants.",
            ),
            table(["Pipeline", *[f"{n} ms" for n in stage_map], "wall ms/img", "img/s"], rows),
        ]
    )


def section_scaling(pipeline: dict, accuracy: dict | None) -> str:
    scaling = [r for r in pipeline["records"] if r.get("suite") == "scaling" and "error" not in r]
    if not scaling:
        return ""
    acc = {}
    if accuracy:
        for r in accuracy["records"]:
            if r.get("runtime") == "TensorRT":
                acc[(r["size"], r["precision"])] = r

    points = []
    for record in scaling:
        b1, b8 = batch_entry(record, 1), batch_entry(record, 8)
        a = acc.get((record["size"], record["precision"]))
        points.append(
            {
                "size": record["size"],
                "precision": record["precision"],
                "b1_p50": b1["latency_ms"]["p50"] if b1 else None,
                "b8_ms_per_image": 1000 / b8["images_per_second"] if b8 else None,
                "b8_ips": b8["images_per_second"] if b8 else None,
                "b32_ips": (batch_entry(record, 32) or {}).get("images_per_second"),
                "map": a["map50_95"] if a else None,
                "map_small": a["map_small"] if a else None,
                "vram": record.get("vram_added_mb"),
            }
        )

    if any(p["map"] is not None for p in points):
        for mode, theme in THEMES.items():
            fig, ax = axes(
                theme,
                "Accuracy against cost: the model-size frontier",
                "COCO val2017 mAP50-95 (all 5,000 images) vs GPU time per image at "
                "batch 8 · TensorRT · RTX 4060",
            )
            for slot, precision in enumerate(("fp16", "int8")):
                pts = sorted(
                    (
                        p
                        for p in points
                        if p["precision"] == precision
                        and p["map"] is not None
                        and p["b8_ms_per_image"]
                    ),
                    key=lambda p: p["b8_ms_per_image"],
                )
                xs = [p["b8_ms_per_image"] for p in pts]
                ys = [p["map"] for p in pts]
                line(ax, theme, slot, xs, ys, f"TensorRT {precision.upper()}", end_label=False)
                for p in pts:
                    ax.annotate(
                        f"yolo26{p['size']}",
                        (p["b8_ms_per_image"], p["map"]),
                        # FP16 labels above-left, INT8 below-right: never on a line
                        xytext=(-6, 8) if slot == 0 else (7, -13),
                        textcoords="offset points",
                        ha="right" if slot == 0 else "left",
                        fontsize=8,
                        color=theme["secondary"],
                    )
            ax.set_xscale("log")
            ticks = [1, 1.5, 2, 3, 4, 6, 8]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.xaxis.set_minor_formatter(NullFormatter())
            label_axis(
                ax, theme, "GPU milliseconds per image at batch 8 (log scale)", "mAP50-95 (%)"
            )
            legend(ax, theme, loc="lower right")
            save(fig, "pareto", mode)

    order = {s: i for i, s in enumerate("nsmlx")}
    points.sort(key=lambda p: (order[p["size"]], p["precision"]))
    rows = [
        [
            f"yolo26{p['size']}",
            p["precision"].upper(),
            fmt(p["map"], 1),
            fmt(p["map_small"], 1),
            fmt(p["b1_p50"], 2),
            fmt(p["b8_ips"], 0),
            fmt(p["b32_ips"], 0),
            fmt(p["vram"], 0),
        ]
        for p in points
    ]
    return "\n\n".join(
        [
            "## Model size, precision and accuracy",
            "Every YOLO26 size on TensorRT at FP16 and INT8. Accuracy is measured here, on all "
            "5,000 COCO val2017 images through the same GPU pipeline the service runs - not quoted "
            "from a model card. INT8 was calibrated on 128 train2017 images (coco128), none of "
            "which are in the evaluation set.",
            picture(
                "pareto",
                "Scatter of mAP against milliseconds per image for five model sizes "
                "at FP16 and INT8; larger models are more accurate and slower.",
            ),
            table(
                [
                    "Model",
                    "Precision",
                    "mAP50-95",
                    "mAP small",
                    "b1 p50 ms",
                    "b8 img/s",
                    "b32 img/s",
                    "VRAM added MB",
                ],
                rows,
            ),
        ]
    )


def section_accuracy(accuracy: dict | None) -> str:
    if not accuracy:
        return ""
    rows = [
        [
            r["label"],
            fmt(r["map50_95"], 2),
            fmt(r["map50"], 2),
            fmt(r["map_small"], 2),
            fmt(r["map_large"], 2),
            fmt(r["images_per_second_end_to_end"], 0),
        ]
        for r in accuracy["records"]
    ]
    return "\n\n".join(
        [
            "## Accuracy detail (COCO val2017, 5,000 images)",
            "Conventional evaluation settings: confidence ≥ 0.001, ≤ 300 detections, official "
            "pycocotools evaluator. *End-to-end img/s* is the whole offline job - JPEG decode on 8 "
            "CPU threads overlapped with the GPU, batch 32 - which is the figure that sizes a "
            "batch job over an image archive.",
            table(
                [
                    "Configuration",
                    "mAP50-95",
                    "mAP50",
                    "mAP small",
                    "mAP large",
                    "End-to-end img/s",
                ],
                rows,
            ),
        ]
    )


OBB_FAMILIES = [("yolo26-obb", "YOLO26-OBB"), ("riov2", "RiO-DETRv2")]


def obb_value(speed, family, size, static, mode, batch, field):
    """A result field, the median over the benchmark passes (robust to one
    disturbed pass on a desktop GPU); None if not measured."""
    import statistics

    values = [
        field(entry)
        for record in speed
        if (record["family"], record["size"], record["static"]) == (family, size, static)
        for entry in record["results"]
        if (entry["mode"], entry["batch"]) == (mode, batch)
    ]
    return statistics.median(values) if values else None


def obb_pass_spread(speed) -> float:
    """Largest relative throughput difference between passes of one measurement."""
    groups: dict[tuple, list[float]] = {}
    for record in speed:
        for entry in record["results"]:
            key = (record["engine"], entry["mode"], entry["batch"])
            groups.setdefault(key, []).append(entry["images_per_second"])
    spreads = [(max(v) - min(v)) / min(v) for v in groups.values() if len(v) > 1]
    return max(spreads) if spreads else 0.0


def obb_median_stability(speed) -> float:
    """Largest relative move of any median when one pass is dropped: how much
    a single disturbed pass could have changed a reported figure."""
    import statistics

    groups: dict[tuple, list[float]] = {}
    for record in speed:
        for entry in record["results"]:
            key = (record["engine"], entry["mode"], entry["batch"])
            groups.setdefault(key, []).append(entry["images_per_second"])
    worst = 0.0
    for values in groups.values():
        if len(values) < 3:
            continue
        median = statistics.median(values)
        for i in range(len(values)):
            rest = statistics.median(values[:i] + values[i + 1 :])
            worst = max(worst, abs(rest - median) / median)
    return worst


def section_obb(obb: dict | None) -> str:
    if not obb or not obb.get("speed"):
        return ""
    speed, published = obb["speed"], obb.get("published", {})
    sizes = [s for s in "nsmlx" if any(r["size"] == s for r in speed)]

    def ips(family, size, batch=8):
        return obb_value(
            speed, family, size, False, "plain", batch, lambda e: e["images_per_second"]
        )

    def p50(family, size, static=False, mode="plain"):
        return obb_value(speed, family, size, static, mode, 1, lambda e: e["latency_ms"]["p50"])

    def activation(family, size):
        return next(
            (
                r["activation_memory_mb"]
                for r in speed
                if r["family"] == family and r["size"] == size and not r["static"]
            ),
            None,
        )

    # RiO-DETRv2's speed as a share of YOLO26-OBB's at the same size: one
    # measure (relative images per second), so both batch sizes share one axis.
    def relative(size, batch):
        if batch == 8:
            y, r = ips("yolo26-obb", size), ips("riov2", size)
            return 100 * r / y if y and r else None
        y, r = p50("yolo26-obb", size), p50("riov2", size)
        return 100 * y / r if y and r else None  # latency, inverted to images per second

    for mode, theme in THEMES.items():
        fig, ax = axes(
            theme,
            "RiO-DETRv2 speed relative to YOLO26-OBB, size for size",
            "YOLO26-OBB = 100% · 1024×1024 DOTA tiles · TensorRT 11 FP16 · RTX 4060 · "
            "one harness for both",
        )
        ax.axhline(100, color=theme["muted"], linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(
            "YOLO26-OBB",
            (0, 100),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=8,
            color=theme["secondary"],
        )
        for slot, (batch, name) in enumerate(((8, "batch 8 throughput"), (1, "batch 1 latency"))):
            pts = [(i, relative(s, batch)) for i, s in enumerate(sizes)]
            pts = [(i, v) for i, v in pts if v]
            line(ax, theme, slot, [p[0] for p in pts], [p[1] for p in pts], name)
        ax.set_ylim(50, 115)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}%"))
        ax.set_xticks(range(len(sizes)), [s.upper() for s in sizes])
        ax.set_xlim(-0.3, len(sizes) + 0.2)
        label_axis(ax, theme, "model size", "RiO-DETRv2 images / second, % of YOLO26-OBB")
        legend(ax, theme, loc="lower right")
        save(fig, "obb-throughput", mode)

    speed_rows, ratios_b8, ratios_b1 = [], {}, {}
    for s in sizes:
        y8, r8 = ips("yolo26-obb", s), ips("riov2", s)
        y1, r1 = p50("yolo26-obb", s), p50("riov2", s)
        if y8 and r8:
            ratios_b8[s] = r8 / y8
        if y1 and r1:
            ratios_b1[s] = r1 / y1
        speed_rows.append(
            [
                s.upper(),
                fmt(y1, 2),
                fmt(r1, 2),
                fmt(y8, 0),
                fmt(r8, 0),
                f"{(r8 / y8 - 1) * 100:+.0f}%" if y8 and r8 else "–",
                f"{fmt(activation('yolo26-obb', s), 0)} / {fmt(activation('riov2', s), 0)}",
            ]
        )

    latency_rows, stream_penalty = [], {}
    for s in sizes:
        for family, name in OBB_FAMILIES:
            own, legacy = p50(family, s), p50(family, s, mode="default_stream")
            if own is None:
                continue
            if legacy:
                stream_penalty[(family, s)] = legacy / own - 1
            latency_rows.append(
                [
                    f"{name}-{s.upper()}",
                    fmt(own, 2),
                    f"{fmt(legacy, 2)} ({(legacy / own - 1) * 100:+.0f}%)" if legacy else "–",
                    fmt(p50(family, s, static=True), 2),
                    fmt(p50(family, s, static=True, mode="cuda_graph"), 2),
                ]
            )

    parity_rows = [
        [
            r["label"],
            f"{r['fp16_vs_fp32']['matched']}/{r['fp16_vs_fp32']['detections']}",
            fmt(r["fp16_vs_fp32"]["mean_abs_score_diff"], 4),
            f"{r['ground_truth_fp16']['recall'] * 100:.1f}%",
        ]
        for r in obb.get("parity", [])
    ]

    yolo_pub, rio_pub = published.get("yolo26-obb", {}), published.get("riov2", {})
    accuracy_rows = [
        [
            s.upper(),
            fmt(yolo_pub.get("params_m", {}).get(s), 1),
            fmt(rio_pub.get("params_m", {}).get(s), 1),
            f"{fmt(yolo_pub.get('map50_95', {}).get(s), 1)} / {fmt(yolo_pub.get('map50', {}).get(s), 1)}",
            f"{fmt(rio_pub.get('map50_95', {}).get(s), 1)} / {fmt(rio_pub.get('map50', {}).get(s), 1)}",
        ]
        for s in sizes
    ]

    spread, stability = obb_pass_spread(speed), obb_median_stability(speed)
    passes = len({r.get("pass") for r in speed})
    sessions = len({r.get("session", 0) for r in speed})
    when = f" over {sessions} sessions" if sessions > 1 else ""

    def span(ratios, group, pct):
        values = [ratios[s] for s in group if s in ratios]
        if not values:
            return "–"
        if pct:
            return f"{min(values) * 100:.0f}–{max(values) * 100:.0f}%"
        return f"{min(values):.2f}–{max(values):.2f}×"

    small, large = [s for s in "ns" if s in sizes], [s for s in "mlx" if s in sizes]
    headline = (
        f"At {' and '.join(s.upper() for s in small)}, RiO-DETRv2 delivers "
        f"**{span(ratios_b8, small, True)}** of YOLO26-OBB's throughput at batch 8 and takes "
        f"**{span(ratios_b1, small, False)}** its latency at batch 1. "
    )
    if large:
        extra_ms = {
            s: 1000 / ips("riov2", s) - 1000 / ips("yolo26-obb", s)
            for s in sizes
            if ips("riov2", s) and ips("yolo26-obb", s)
        }
        small_ms = [extra_ms[s] for s in small if s in extra_ms]
        large_ms = [extra_ms[s] for s in large if s in extra_ms]
        headline += (
            f"From {large[0].upper()} upward the gap closes: **{span(ratios_b8, large, True)}** of "
            f"the throughput, **{span(ratios_b1, large, False)}** the latency. Per image at batch "
            f"8, RiO-DETRv2 minus YOLO26-OBB is {min(small_ms):+.2f} to {max(small_ms):+.2f} ms at "
            f"the small sizes and {min(large_ms):+.2f} to {max(large_ms):+.2f} ms at the large ones."
        )
    parts = [
        "## Oriented boxes: YOLO26-OBB against RiO-DETRv2",
        "Two real-time oriented-box detectors trained on DOTA-v1.0, at every size, on one harness: "
        "YOLO26-OBB (a CNN with an NMS-free head) and RiO-DETRv2 (a detection transformer, the "
        "successor to ECCV 2026's RiO-DETR, https://github.com/RicePasteM/RiO-DETRv2). Both are "
        "exported to ONNX "
        "with their post-processing inside the graph, given FP16 by the same ModelOpt AutoCast "
        "step, and built by the same TensorRT 11 code with the same 1024×1024 profile tuned for "
        "batch 8 (`tools/export.py`, `tools/export_rio.py`). Timing is the engine alone, input "
        "already on the GPU, one synchronise per batch, each engine in its own process. The "
        f"matrix ran {passes} times{when}, interleaved, and each figure is the median of the "
        "passes. "
        "This GPU also drives the desktop, and a pass that shares it with another application "
        f"runs slow: the widest gap between two passes of one measurement is {spread * 100:.0f}%. "
        "The medians are robust to that - dropping any single pass moves none of them by more "
        f"than {stability * 100:.1f}%.",
        headline,
        picture(
            "obb-throughput",
            "Line chart of RiO-DETRv2 speed as a percentage of YOLO26-OBB at five model sizes; "
            "at batch 8 it rises from below 80% at N to about 100% from M upward, and at batch 1 "
            "it stays below 100% at every size.",
        ),
        table(
            [
                "Size",
                "YOLO26-OBB b1 p50 ms",
                "RiO-DETRv2 b1 p50 ms",
                "YOLO26-OBB b8 img/s",
                "RiO-DETRv2 b8 img/s",
                "RiO-DETRv2 vs YOLO26-OBB at b8",
                "Activation MB (YOLO / RiO)",
            ],
            speed_rows,
        ),
    ]
    if latency_rows:
        parts += [
            "Batch-1 latency four ways. The dynamic engine on its own CUDA stream is the figure "
            "above. The same engine on CUDA's legacy default stream - where TensorRT adds "
            "synchronisations of its own, and where a quick benchmark script usually runs it - "
            "shows how much the harness alone moves a result. Static batch-1 engines (N and S) "
            "check that neither family is judged on a profile tuned for batch 8, and CUDA-graph "
            "replay removes the per-launch CPU cost.",
            table(
                [
                    "Model",
                    "Dynamic b1 p50 ms",
                    "Same, default stream",
                    "Static b1 p50 ms",
                    "Static b1, CUDA graph",
                ],
                latency_rows,
            ),
        ]
    parts += [
        "### Accuracy, as published",
        "Not measured here. DOTA-v1.0 test labels are withheld - scores come from the official "
        "evaluation server - and RiO-DETRv2 trains on train+val, so no local split is fair to "
        "both. These are each author's test-server results, and the protocols differ: "
        f"YOLO26-OBB is {yolo_pub.get('protocol', '?').split(', ')[-1]} "
        f"({yolo_pub.get('source', '')}), RiO-DETRv2 is {rio_pub.get('protocol', '?').split(', ')[-1]} "
        f"({rio_pub.get('source', '')}). The RiO-DETRv2 README's own accuracy-latency chart "
        "re-scores YOLO26-OBB single-scale and puts the two within about 0.2 mAP50-95 of each "
        "other at every size, with RiO-DETRv2 0.3–1.4 points ahead on AP50; those values are "
        "read off a chart, so they are not tabulated.",
        table(
            [
                "Size",
                "YOLO26-OBB params M",
                "RiO-DETRv2 params M",
                "YOLO26-OBB mAP50-95 / AP50 (multi-scale)",
                "RiO-DETRv2 mAP50-95 / AP50 (single-scale)",
            ],
            accuracy_rows,
        ),
    ]
    if parity_rows:
        parts += [
            "FP16 parity, on the 8 DOTA tiles of Ultralytics' dota8: the share of FP32 detections "
            "(score ≥ 0.3) each FP16 engine reproduces - same class, rotated-box ProbIoU ≥ 0.9 - "
            "against the same graph in FP32 on ONNX Runtime, and recall of the labelled objects. "
            "The tiles are from the training data, so the recall column proves the boxes, angles "
            "and classes decode correctly, not accuracy.",
            table(
                [
                    "Engine",
                    "FP32 detections reproduced",
                    "Mean score difference",
                    "dota8 recall (FP16)",
                ],
                parity_rows,
            ),
        ]
    return "\n\n".join(parts)


SERVICE_ORDER = [
    "v1",
    "pt-nobatch",
    "trt16-nobatch",
    "trt16-batch",
    "trt16-batch-raw",
    "trt16-batch-raw-4w",
    "trt16-batch-nvjpeg",
]
SERVICE_SHORT = {
    "v1": "v1 (PyTorch, blocking)",
    "pt-nobatch": "v2 PyTorch",
    "trt16-nobatch": "v2 TRT FP16",
    "trt16-batch": "v2 TRT FP16 + batching",
    "trt16-batch-raw": "+ raw-body upload",
    "trt16-batch-raw-4w": "+ 4 workers per GPU",
    "trt16-batch-nvjpeg": "batching + nvJPEG decode",
}


def capacity_at(levels, slo_ms):
    """Highest throughput sustained with p99 under the SLO and nothing shed."""
    ok = [
        lv
        for lv in levels
        if lv.get("latency_ms")
        and lv["latency_ms"]["p99"] <= slo_ms
        and lv["shed_503"] == 0
        and lv["failed"] == 0
    ]
    return max(ok, key=lambda lv: lv["throughput_rps"]) if ok else None


def section_service(service: dict | None) -> tuple[str, dict]:
    if not service:
        return "", {}
    configs = {c["key"]: c for c in service["configs"]}
    present = [configs[k] for k in SERVICE_ORDER if k in configs]

    for metric, name, title, ylabel, log in (
        (
            "throughput_rps",
            "service-throughput",
            "HTTP throughput under concurrent load",
            "requests / second",
            False,
        ),
        ("p99", "service-p99", "Tail latency under concurrent load", "p99 latency, ms (log)", True),
    ):
        for mode, theme in THEMES.items():
            fig, ax = axes(
                theme,
                title,
                "yolo26n · 134 KB JPEG per request · closed loop · "
                "client and server on one machine",
            )
            for slot, config in enumerate(present):
                levels = [lv for lv in config["levels"] if lv.get("latency_ms")]
                xs = [lv["concurrency"] for lv in levels]
                ys = [
                    lv["throughput_rps"] if metric == "throughput_rps" else lv["latency_ms"]["p99"]
                    for lv in levels
                ]
                line(ax, theme, slot, xs, ys, SERVICE_SHORT[config["key"]], end_label=False)
            ax.set_xscale("log", base=2)
            ticks = sorted({lv["concurrency"] for c in present for lv in c["levels"]})
            ax.set_xticks(ticks, [str(t) for t in ticks])
            if log:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
            else:
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_formatter(FuncFormatter(thousands))
            label_axis(ax, theme, "requests in flight (concurrency)", ylabel)
            legend_above(ax, theme)
            save(fig, name, mode, top=0.72)

    rows, capacity = [], {}
    for config in present:
        levels = config["levels"]
        peak = max(levels, key=lambda lv: lv["throughput_rps"])
        c50, c100 = capacity_at(levels, 50), capacity_at(levels, 100)
        capacity[config["key"]] = {"slo50": c50, "slo100": c100, "peak": peak}
        rows.append(
            [
                SERVICE_SHORT[config["key"]],
                fmt(levels[0].get("latency_ms", {}).get("p50"), 1),
                f"{fmt(peak['throughput_rps'], 0)} @ c{peak['concurrency']}",
                fmt(c50["throughput_rps"], 0) if c50 else "–",
                fmt(c100["throughput_rps"], 0) if c100 else "–",
                fmt(peak.get("mean_batch_size"), 1),
                fmt(peak["gpu"].get("utilization_mean_pct"), 0),
                fmt(peak.get("server_cpu_pct"), 0),
                fmt(config.get("ready_seconds"), 1),
            ]
        )
    detail = []
    for config in present:
        detail.append(f"**{config['label']}**\n")
        detail.append(
            table(
                [
                    "Concurrency",
                    "req/s",
                    "p50 ms",
                    "p99 ms",
                    "mean batch",
                    "queue wait ms",
                    "GPU util %",
                    "server CPU %",
                    "503s",
                ],
                [
                    [
                        lv["concurrency"],
                        fmt(lv["throughput_rps"], 0),
                        fmt(lv.get("latency_ms", {}).get("p50"), 1),
                        fmt(lv.get("latency_ms", {}).get("p99"), 1),
                        fmt(lv.get("mean_batch_size"), 2),
                        fmt(lv.get("mean_queue_wait_ms"), 2),
                        fmt(lv["gpu"].get("utilization_mean_pct"), 0),
                        fmt(lv.get("server_cpu_pct"), 0),
                        lv["shed_503"],
                    ]
                    for lv in config["levels"]
                ],
            )
        )
    text = "\n\n".join(
        [
            "## The service under load",
            "Real HTTP: each virtual client uploads the 134 KB sample JPEG (as multipart form data, "
            "or as the raw request body for the raw-body configurations), waits for the JSON "
            "answer, and sends the next. The load generator runs in 4 processes "
            "on the same machine as the server, so both compete for the same CPU - a conservative "
            "setup for the server. *Capacity @ SLO* is the highest throughput reached with p99 under "
            "the target and no request shed.",
            picture(
                "service-throughput",
                "Line chart of requests per second against concurrency "
                "for seven service configurations; four TensorRT workers sharing the GPU "
                "scale highest.",
            ),
            picture("service-p99", "Line chart of p99 latency against concurrency on a log scale."),
            table(
                [
                    "Configuration",
                    "c1 p50 ms",
                    "Peak req/s",
                    "Capacity @ p99≤50 ms",
                    "Capacity @ p99≤100 ms",
                    "Batch @ peak",
                    "GPU util @ peak %",
                    "Server CPU @ peak %",
                    "Ready in s",
                ],
                rows,
            ),
            "<details><summary>Every concurrency level</summary>\n\n"
            + "\n\n".join(detail)
            + "\n\n</details>",
        ]
    )
    return text, capacity


def section_capacity(capacity: dict, pipeline: dict, service: dict | None) -> str:
    best_key = next(
        (k for k in ("trt16-batch-raw-4w", "trt16-batch") if capacity.get(k, {}).get("slo50")),
        None,
    )
    if not best_key or not capacity[best_key]["slo50"]:
        return ""
    rps = capacity[best_key]["slo50"]["throughput_rps"]
    limit_w = (service or {}).get("machine", {}).get("gpu_power_limit_w") or 130
    rows = []
    for fps in (5, 15, 30):
        rows.append([f"{fps} fps", f"{math.floor(rps / fps):,}"])
    cost_rows = []
    for price in (0.25, 0.75, 2.00):
        per_million = price / (rps * 3600) * 1e6
        cost_rows.append(
            [f"${price:.2f}", f"${per_million:.3f}", f"{rps * 3600 * 24 / 1e6:,.1f} M"]
        )
    return "\n\n".join(
        [
            "## Capacity planning",
            f"From the measured service capacity of **{rps:,.0f} requests/s per GPU at p99 ≤ 50 ms** "
            f"({SERVICE_SHORT[best_key]}: TensorRT FP16, dynamic batching, one RTX 4060).",
            "**Camera streams per GPU** (one detection per frame):",
            table(["Frame rate", "Streams per GPU"], rows),
            "**Cost per million images** at a hypothetical all-in GPU-hour price. Substitute your "
            "own; the arithmetic is `price / (throughput × 3600) × 10⁶`, at sustained full load:",
            table(["GPU-hour price", "Per million images", "Images per GPU-day"], cost_rows),
            f"**Energy ceiling.** This board does not report power draw through NVML, so energy is "
            f"bounded rather than measured: at its {limit_w:.0f} W power limit and {rps:,.0f} "
            f"images/s, at most **{limit_w / rps * 1000:.0f} mJ per image** "
            f"({limit_w / rps * 1e6 / 3600:.2f} Wh per thousand).",
            "**Autoscaling target.** `deploy/k8s/observability.yaml` runs four single-worker replicas "
            "per GPU and scales on request rate, with a per-replica target of 70% of a quarter of "
            f"this capacity: **{rps * 0.7 / 4:,.0f} req/s**.",
        ]
    )


def section_decode(pipeline: dict) -> str:
    decode = next((r for r in pipeline["records"] if r.get("suite") == "decode"), None)
    if not decode:
        return ""
    rows = [[r["label"], fmt(r["images_per_second"], 0)] for r in decode["results"]]
    return "\n\n".join(
        [
            "## JPEG decode: CPU threads or nvJPEG?",
            f"Decoding the {decode['image_bytes'] / 1024:.0f} KB, 810×1080 sample, in isolation.",
            table(["Decoder", "img/s"], rows),
        ]
    )


def section_cpu(pipeline: dict) -> str:
    records = [r for r in pipeline["records"] if r.get("suite") == "cpu" and "error" not in r]
    if not records:
        return ""
    rows = []
    for r in records:
        for b in r["batches"]:
            rows.append(
                [
                    r["label"],
                    b["batch"],
                    fmt(b["latency_ms"]["p50"], 1),
                    fmt(b["latency_ms"]["p99"], 1),
                    fmt(b["images_per_second"], 1),
                ]
            )
    return "\n\n".join(
        [
            "## Without a GPU",
            "The same yolo26n on the host CPU through the stock predictor - the fallback when no "
            "GPU is present, and the denominator of the GPU's speed-up.",
            table(["Configuration", "Batch", "p50 ms", "p99 ms", "img/s"], rows),
        ]
    )


def machine_block(*payloads) -> str:
    machine = next((p["machine"] for p in payloads if p and "machine" in p), {})
    keys = [
        "gpu",
        "gpu_memory_gb",
        "gpu_sm",
        "gpu_sms",
        "driver",
        "cpu",
        "cpu_threads",
        "os",
        "python",
        "torch",
        "cuda",
        "cudnn",
        "tensorrt",
        "onnxruntime",
        "ultralytics",
        "cv2",
        "fastapi",
        "timestamp",
    ]
    return "\n\n".join(
        [
            "## Test machine",
            table(["", ""], [[k, machine[k]] for k in keys if k in machine]),
        ]
    )


def main() -> int:
    pipeline = load("pipeline")
    accuracy = load("accuracy")
    service = load("service")
    obb = load("obb")
    if not pipeline:
        raise SystemExit("results/pipeline.json is missing; run tools/bench_pipeline.py first")

    service_text, capacity = section_service(service)
    parts = [
        "# Benchmark results",
        "Generated by `tools/report.py` from the JSON in `results/`. Every number below was "
        "measured on the machine described at the end. The one exception is labelled where it "
        "appears: DOTA accuracy for the oriented-box models, which only the official evaluation "
        "server can score, is each author's published result. Reproduce with the commands in "
        "the README.",
        section_runtime(pipeline),
        section_stages(pipeline),
        section_scaling(pipeline, accuracy),
        service_text,
        section_capacity(capacity, pipeline, service),
        section_accuracy(accuracy),
        section_obb(obb),
        section_decode(pipeline),
        section_cpu(pipeline),
        machine_block(pipeline, accuracy, service),
    ]
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "BENCHMARKS.md"
    out.write_text("\n\n".join(p for p in parts if p) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} and {len(list(CHARTS.glob('*.png')))} charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
