# farm2fork-ai

AI-powered crop quality grading pipeline for Farm2Fork.

The first MVP targets six Pakistani crops:

- wheat
- rice
- mango
- maize
- cotton
- sugarcane

Each crop is graded as `A`, `B`, `C`, or `D`.

## Current Status

This repository is scaffolded for model training, inference, and API deployment. The active implementation plan lives in [plan/crop_grading_implementation_plan.md](plan/crop_grading_implementation_plan.md).

## Setup

Create and activate a Python 3.10+ environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

Validate the scaffold and config:

```powershell
python scripts/validate_setup.py
```

Validate dataset manifests:

```powershell
python scripts/validate_manifests.py --no-file-check --allow-missing
```

Build bootstrap quality manifests from confirmed rice and mango mappings:

```powershell
python scripts/build_bootstrap_quality_manifests.py
```

Smoke test dataset loading:

```powershell
python scripts/smoke_test_dataset.py
```

Smoke test model forward pass:

```powershell
python scripts/smoke_test_model.py
```

## Project Layout

```text
api/                 FastAPI service code
configs/             Hyperparameters and paths
data/raw/            Original datasets, never modified
data/processed/      Cleaned/split training images
data/metadata/       CSV manifests for train/val/test
models/checkpoints/  Training checkpoints
models/exports/      ONNX/TFLite exports
notebooks/           Exploration and error analysis
scripts/             CLI entry points
src/crop_grading/    Python package
tests/               Automated tests
```
