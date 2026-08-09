# CPU image. See the GPU section in README.md for the CUDA variant.
#
# Two stages: dependencies are installed into a virtualenv in the builder and
# copied across, so the build toolchain never reaches the final image.

FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch. Installed first and from its own index so the resolver does
# not pull the much larger CUDA build as a transitive dependency.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt .
# The service never opens a window, so the headless OpenCV build is correct
# here: it drops the GUI stack and the libGL system libraries it would need.
RUN sed 's/^opencv-python>=/opencv-python-headless>=/' requirements.txt > requirements.docker.txt \
    && pip install -r requirements.docker.txt


FROM python:3.13-slim AS runtime

# libglib2 is still required by OpenCV even in its headless build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Run unprivileged.
RUN useradd --create-home --uid 10001 vision
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=vision:vision src/ ./src/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VISION_MODEL=yolo26n.pt \
    ULTRALYTICS_DIR=/home/vision/.ultralytics

USER vision

# Bake the weights into the image. Without this the first container start
# reaches out to the internet, which fails in an air-gapped cluster and makes
# startup time depend on a third party.
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

EXPOSE 8000

# Readiness, not liveness: the container is up long before the model is warm.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/readyz || exit 1

CMD ["uvicorn", "vision.service:app", "--host", "0.0.0.0", "--port", "8000"]
