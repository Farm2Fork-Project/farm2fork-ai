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

# Trained weights are not in git. Mount or copy them to this path; without
# them the service runs in its labelled "untrained" preview mode.
RUN mkdir -p models/checkpoints/grade_cond_dedup
ENV MODEL_CHECKPOINT=/app/models/checkpoints/grade_cond_dedup/best_model.pth \
    MODEL_CONFIG=/app/configs/four_crops_dedup_15ep.yaml \
    PRICE_RULES=/app/configs/price_rules.yaml

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "crop_grading.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
