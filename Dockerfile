# Farm2Fork AI inference service (FastAPI + PyTorch, CPU).
FROM python:3.11-slim

# CPU-only PyTorch wheels keep the image small. Override the index to build
# from plain PyPI when download.pytorch.org is unreachable.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV APP_ENV=production \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-runtime.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=false pip install --extra-index-url "${TORCH_INDEX_URL}" -r requirements-runtime.lock

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps ".[api]"

COPY configs ./configs
# Photo-similarity reference for the served weights (checked by fingerprint).
COPY reference ./reference
COPY scripts/validate_release.py ./scripts/validate_release.py

# Mount an independently verified trained artifact. Missing weights disable grading.
RUN mkdir -p models
ENV MODEL_CHECKPOINT=/app/models/best_model.pth \
    MODEL_CONFIG=/app/configs/grade_only_four_crops_dedup_15ep.yaml \
    MODEL_PHOTO_REFERENCE=/app/reference/photo_reference.npz \
    ALLOW_UNTRAINED_MODEL=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OFFICIAL_PRICES=/app/configs/official_prices.json

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "crop_grading.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
