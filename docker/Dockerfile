# syntax=docker/dockerfile:1.10
#
# One Dockerfile, two targets, chosen by build argument:
#
#   docker build -f docker/Dockerfile -t yolo26:cpu .
#   docker build -f docker/Dockerfile --build-arg EXTRA=gpu -t yolo26:gpu .
#
# The GPU image needs no CUDA base image. PyTorch, TensorRT and ONNX Runtime
# ship their CUDA 13 user-space libraries as wheels, and the NVIDIA Container
# Toolkit injects the driver at run time. A CUDA base would put a second,
# unused copy of the same libraries in the image.
#
# Dependencies come from uv.lock and nothing else (`--frozen`): the image is
# the lockfile, byte for byte, and Ultralytics' runtime pip-install is off.

ARG PYTHON=3.13

# ---------------------------------------------------------------- build
FROM python:${PYTHON}-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

ARG EXTRA=cpu
WORKDIR /src

# Dependencies first, in their own layer: source edits do not reinstall torch.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-default-groups --extra ${EXTRA}

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --extra ${EXTRA} --no-editable

# Weights, pinned by digest. A cold start must not depend on reaching GitHub,
# and an upstream re-upload must not change what runs.
ADD --checksum=sha256:9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt /models/yolo26n.pt

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON}-slim-trixie AS runtime

# OpenCV's headless build still links GLib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 vision

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=vision:vision /models /models
# Engine builder, for the Kubernetes init container (deploy/k8s/deployment.yaml).
COPY tools/export.py /opt/tools/export.py

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_AUTOINSTALL=false \
    YOLO_OFFLINE=true \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    VISION_MODEL=/models/yolo26n.pt

USER vision
WORKDIR /home/vision
EXPOSE 8000

# Readiness, not liveness: the container is up long before the model is warm.
# Python rather than curl, so the image carries no HTTP client it doesn't need.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4).status != 200)"]

# One worker per GPU: a second worker would load a second copy of the model
# onto the same device and the two would contend for it. Scale out with
# replicas (one per GPU), not with workers.
CMD ["uvicorn", "vision.service:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
