# Farm2Fork AI inference service (FastAPI + PyTorch, CPU).
FROM python:3.11-slim

# CPU-only PyTorch wheels keep the image small. Override the index to build
# from plain PyPI when download.pytorch.org is unreachable.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.2.0" "torchvision>=0.17.0"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]" "timm>=0.9.12" "numpy>=1.26.0"

COPY configs ./configs
# Photo-similarity reference for the served weights (checked by fingerprint).
COPY reference ./reference

# Mount an independently verified trained artifact. Missing weights disable grading.
RUN mkdir -p models
ENV MODEL_CHECKPOINT=/app/models/best_model.pth \
    MODEL_CONFIG=/app/configs/grade_only_four_crops_dedup_15ep.yaml \
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
