"""Image decoding and annotation, using OpenCV.

Kept separate from the detector so that drawing has no opinion about where
boxes came from, and decoding can be unit-tested without a model.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

MAX_DECODE_BYTES = 20 * 1024 * 1024  # refuse anything larger than 20 MB

# Distinct, colour-blind-safe box colours, cycled by class id.
_PALETTE = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (204, 121, 167), (150, 150, 150),
]


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes are not a decodable image."""


def decode_image(payload: bytes) -> np.ndarray:
    """Decode image bytes into a BGR array.

    Decoding is done from memory rather than a temp file so the service never
    writes user-supplied data to disk.
    """
    if not payload:
        raise ImageDecodeError("The uploaded file was empty.")
    if len(payload) > MAX_DECODE_BYTES:
        raise ImageDecodeError(
            f"Image is larger than the {MAX_DECODE_BYTES // (1024 * 1024)} MB limit."
        )

    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageDecodeError(
            "Could not decode the upload as an image. Supported: JPEG, PNG, BMP, WebP."
        )
    return image


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode a BGR array as JPEG bytes."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode the image as JPEG.")
    return buffer.tobytes()


def colour_for(class_id: int) -> tuple[int, int, int]:
    """Stable colour for a class id."""
    return _PALETTE[class_id % len(_PALETTE)]


def annotate(image: np.ndarray, detections: list[Detection], show_confidence: bool = True):
    """Draw boxes and labels onto a copy of the image.

    The label is drawn inside the box when it would otherwise run off the top
    of the frame, which happens for any object detected against the top edge.
    """
    canvas = image.copy()
    height = canvas.shape[0]
    thickness = max(1, round(min(canvas.shape[:2]) / 400))

    for detection in detections:
        colour = colour_for(detection.class_id)
        x1, y1 = int(detection.x1), int(detection.y1)
        x2, y2 = int(detection.x2), int(detection.y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

        label = detection.class_name
        if show_confidence:
            label = f"{label} {detection.confidence:.2f}"

        scale = max(0.4, thickness * 0.35)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )

        top = y1 - text_h - baseline
        if top < 0:  # box hugs the top edge; put the label inside it instead
            top = y1
        cv2.rectangle(
            canvas, (x1, top), (x1 + text_w, top + text_h + baseline), colour, -1
        )
        cv2.putText(
            canvas,
            label,
            (x1, min(top + text_h, height - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return canvas
