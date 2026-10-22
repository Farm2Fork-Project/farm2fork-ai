# farm2fork-ai

AI-powered crop quality grading pipeline for Farm2Fork.

The grading MVP supports four crops selected explicitly by the farmer:

- wheat
- rice
- mango
- maize
Supported photos receive an optional `A`, `B`, `C`, or `D` model suggestion, confirmed by
the farmer. **Other crop** is a client-side selection that skips quality inference; unsupported
direct requests are rejected. Cotton and sugarcane have no grading training data.

## Current Status

The MVP serving flow and release artifact are documented in [docs/mvp-grading.md](docs/mvp-grading.md).
The research scripts and historical implementation plan remain available.

## Inference service (FastAPI)

The service in `src/crop_grading/service/` is what the Farm2Fork backend calls
(master context 10.2). Only the Nest backend should reach it.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness, and whether a **trained** checkpoint is loaded |
| `POST /predict/quality` | Multipart `image` + `crop` → grade A–D, probabilities, confidence |
| `POST /predict/price` | Product facts → rule-based PKR range per unit (`method: rule_based`) |

**Quality model.** The default is the frozen plain grade-only EfficientNet-B0 seed-42
checkpoint. The farmer supplies the crop; crop recognition is not required. The service
also loads legacy crop-conditioned checkpoints to permit gradual rollout/rollback.
Missing or mismatched weights make grading unavailable (503), while prices still work.
Explicit `ALLOW_UNTRAINED_MODEL=true` enables labeled development previews only.
The supplied-photo diagnostic showed poor transfer; suggestions require farmer confirmation
and must not be described as certified assessments or 92.8% field accuracy.

**Photo checks (referral).** Softmax confidence did not separate right from wrong on the
supplied phone photos: the four most confident predictions were all wrong. A grade is
therefore shown only when the photo resembles the training photos *and* confidence is at
least 0.7; otherwise the response sets `needsReview` with a `reviewReason`
(`unfamiliar_photo`, `low_confidence`, `photo_check_unavailable`) and clients ask for a
retake or a manual grade. Similarity is the mean cosine similarity to the 5 nearest training
images in the backbone embedding; the threshold is the validation 5th percentile (0.6575).
`reference/photo_reference.npz` is built by `scripts/build_photo_reference.py` and is tied
to the exact weights by a state-dict fingerprint; any other checkpoint refers every photo.
Thresholds were fixed on validation data only. Held-out test: 88.8% of images graded at
96.9% accuracy (94.2% ungated). Supplied phone photos: 24 of 25 referred, 1 shown (wrong).

**Price reference.** `/predict/price` returns observed, dated AMIS Punjab wholesale mandi
prices, not a model: the farmer's own market when it reported (backend sends the farm
city), otherwise the middle half of fair-quality prices across Punjab markets. Every
response carries the commodity, trading date, market scope and source. Products without an
official commodity, or units that cannot be converted, get 422 instead of a guess.
Refresh the snapshot with `scripts/update_official_prices.py` (writes
`configs/official_prices.json`; core crops walk back to their last day with at least three
reporting markets) and release a new image. Prices are wholesale, not farm-gate.

```bash
pip install -e ".[ml,api,dev]"
uvicorn crop_grading.service.app:app --port 8000
# or
docker build -t farm2fork-ai . && docker run -p 8000:8000 farm2fork-ai
```

| Variable | Default |
| --- | --- |
| `MODEL_CHECKPOINT` | `models/checkpoints/grade_only_four_crops_dedup_15ep/best_model.pth` (Docker: `/app/models/best_model.pth`) |
| `MODEL_CONFIG` | `configs/grade_only_four_crops_dedup_15ep.yaml` |
| `MODEL_TRAINED_CROPS` | `wheat,rice,mango,maize` (drives `cropSupported`) |
| `ALLOW_UNTRAINED_MODEL` | `false` |
| `AI_SERVICE_TOKEN` | unset; when set, `/predict/*` requires header `X-Internal-Token` |
| `MAX_IMAGE_BYTES` | `8388608` |

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

Run a tiny training sanity check:

```powershell
python scripts/sanity_train_subset.py
```

Run a small checkpointed training experiment:

```powershell
python scripts/train.py
```

Add `--no-progress` to training or evaluation commands if you want plain logs only.
Training uses grade class weights by default. Use `--class-weights none` to disable them, or `--class-weights both` to weight crop and grade losses.
`best_model.pth` is selected by a combined score by default: crop accuracy, adjacent grade accuracy, and exact grade accuracy.
Use `--sampler balanced` to oversample rare crop-grade groups during training.
Training runs are appended to `outputs/experiments/training_runs.csv`.
Training supports early stopping with `--patience` and `--min-delta`; default patience is 8 epochs.

Evaluate a checkpoint:

```powershell
python scripts/evaluate.py --backbone efficientnet_b0 --checkpoint models/checkpoints/best_model.pth
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
