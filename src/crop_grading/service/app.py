"""Farm2Fork AI service (master context 10.2).

Endpoints
- GET  /health            liveness + which model is loaded and whether it is trained
- POST /predict/quality   multipart: image + crop  -> grade A-D with probabilities
- POST /predict/price     JSON product facts       -> rule-based PKR range per unit

Run:  uvicorn crop_grading.service.app:app --host 0.0.0.0 --port 8000

Only the Nest backend should call this service (it runs on the internal Docker
network). When AI_SERVICE_TOKEN is set every /predict call must carry it in
the X-Internal-Token header.
"""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from crop_grading.constants import CROP_CLASSES
from crop_grading.service.grading import (
    InvalidImageError,
    ModelUnavailableError,
    QualityGrader,
)
from crop_grading.service.pricing import PriceNotEstimableError, PriceRules
from crop_grading.service.settings import ServiceSettings

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = ServiceSettings.from_env()
    app.state.settings = settings
    app.state.prices = PriceRules(settings.price_rules_path)
    try:
        app.state.grader = QualityGrader(settings)
        app.state.grader_error = None
    except ModelUnavailableError as exc:
        # Price estimates still work; /predict/quality answers 503.
        app.state.grader = None
        app.state.grader_error = str(exc)
    yield


app = FastAPI(
    title="Farm2Fork AI service",
    version="1.0.0",
    description="Crop quality grading and rule-based price estimates.",
    lifespan=lifespan,
)


def require_internal_token(
    request: Request,
    x_internal_token: Annotated[str | None, Header()] = None,
) -> None:
    expected = request.app.state.settings.service_token
    if expected is None:
        return
    if x_internal_token is None or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid internal service token.")


# --- health ------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"]
    qualityModel: Literal["trained", "untrained", "unavailable"]
    qualityModelVersion: str | None
    trainedCrops: list[str]
    priceModelVersion: str
    priceMethod: Literal["rule_based"]


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    grader: QualityGrader | None = request.app.state.grader
    return HealthResponse(
        status="ok",
        qualityModel=grader.status if grader else "unavailable",
        qualityModelVersion=grader.version if grader else None,
        trainedCrops=list(request.app.state.settings.trained_crops),
        priceModelVersion=request.app.state.prices.version,
        priceMethod="rule_based",
    )


# --- quality -----------------------------------------------------------------


class QualityResponse(BaseModel):
    qualityGrade: Literal["A", "B", "C", "D"]
    confidenceScore: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    crop: str
    cropSupported: bool = Field(
        description="False when the model has not been trained on this crop yet."
    )
    lowConfidence: bool
    modelStatus: Literal["trained", "untrained"]
    modelVersion: str


@app.post(
    "/predict/quality",
    response_model=QualityResponse,
    dependencies=[Depends(require_internal_token)],
)
async def predict_quality(
    request: Request,
    image: Annotated[UploadFile, File(description="Photo of the produce (JPEG/PNG/WebP)")],
    crop: Annotated[str, Form(description=f"One of: {', '.join(CROP_CLASSES)}")],
) -> QualityResponse:
    grader: QualityGrader | None = request.app.state.grader
    if grader is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, request.app.state.grader_error)

    crop = crop.strip().lower()
    if crop not in CROP_CLASSES:
        raise HTTPException(
            422,
            f"Unknown crop '{crop}'. Expected one of: {', '.join(CROP_CLASSES)}.",
        )
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a JPEG, PNG or WebP photo."
        )

    limit = request.app.state.settings.max_image_bytes
    data = await image.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            413,
            f"Photo is larger than {limit // (1024 * 1024)} MB.",
        )

    try:
        prediction = grader.grade(data, crop)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    return QualityResponse(
        qualityGrade=prediction.grade,
        confidenceScore=prediction.confidence,
        probabilities=prediction.probabilities,
        crop=crop,
        cropSupported=crop in grader.trained_crops,
        lowConfidence=prediction.low_confidence,
        modelStatus=grader.status,
        modelVersion=grader.version,
    )


# --- price -------------------------------------------------------------------


class PriceRequest(BaseModel):
    productName: str = Field(min_length=1, max_length=120)
    category: str = Field(min_length=1, max_length=40)
    unit: str = Field(min_length=1, max_length=16)
    quantity: float | None = Field(default=None, gt=0)
    qualityGrade: Literal["A", "B", "C", "D"] | None = None
    location: str | None = Field(default=None, max_length=120)
    asOf: date | None = Field(
        default=None, description="Day to price for (defaults to today); drives seasonality."
    )


class PriceResponse(BaseModel):
    predictedMinPrice: float
    predictedMaxPrice: float
    unit: str
    currency: Literal["PKR"] = "PKR"
    confidenceScore: float
    modelVersion: str
    method: Literal["rule_based"] = "rule_based"
    basis: Literal["crop", "category"]
    matchedCrop: str | None
    seasonalFactor: float
    gradeFactor: float


@app.post(
    "/predict/price",
    response_model=PriceResponse,
    dependencies=[Depends(require_internal_token)],
)
def predict_price(request: Request, body: PriceRequest) -> PriceResponse:
    try:
        estimate = request.app.state.prices.estimate(
            product_name=body.productName,
            category=body.category,
            unit=body.unit,
            quality_grade=body.qualityGrade,
            on=body.asOf,
        )
    except PriceNotEstimableError as exc:
        raise HTTPException(422, str(exc)) from exc

    return PriceResponse(
        predictedMinPrice=estimate.min_price,
        predictedMaxPrice=estimate.max_price,
        unit=estimate.unit,
        confidenceScore=estimate.confidence,
        modelVersion=estimate.model_version,
        basis=estimate.basis,
        matchedCrop=estimate.matched_crop,
        seasonalFactor=estimate.seasonal_factor,
        gradeFactor=estimate.grade_factor,
    )
