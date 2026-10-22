"""Farm2Fork AI service (master context 10.2).

Endpoints
- GET  /health            liveness + which model is loaded and whether it is trained
- POST /predict/quality   multipart: image + crop  -> grade A-D with probabilities
- POST /predict/price     JSON product facts       -> dated official wholesale PKR range per unit

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

from crop_grading.constants import SUPPORTED_GRADE_CROPS
from crop_grading.service.grading import (
    InvalidImageError,
    ModelUnavailableError,
    QualityGrader,
)
from crop_grading.service.pricing import OfficialPrices, PriceNotEstimableError
from crop_grading.service.settings import ServiceSettings

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = ServiceSettings.from_env()
    app.state.settings = settings
    app.state.prices = OfficialPrices(settings.official_prices_path)
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
    description="Crop quality grading and official wholesale price references.",
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
    photoCheck: Literal["enabled", "unavailable"] = Field(
        description="Whether unfamiliar photos can be detected and referred for review."
    )
    trainedCrops: list[str]
    priceModelVersion: str
    priceMethod: Literal["official_wholesale"]


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    grader: QualityGrader | None = request.app.state.grader
    return HealthResponse(
        status="ok",
        qualityModel=grader.status if grader else "unavailable",
        qualityModelVersion=grader.version if grader else None,
        photoCheck="enabled" if grader and grader.photo_reference else "unavailable",
        trainedCrops=list(request.app.state.settings.trained_crops),
        priceModelVersion=request.app.state.prices.version,
        priceMethod="official_wholesale",
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
    needsReview: bool = Field(
        description="True when the grade must not be shown: ask for a retake or a manual grade."
    )
    reviewReason: Literal["low_confidence", "unfamiliar_photo", "photo_check_unavailable"] | None
    photoSimilarity: float | None = Field(
        description="Similarity to the training photos (higher is more familiar)."
    )
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
    crop: Annotated[
        str, Form(description=f"Farmer-selected crop: {', '.join(SUPPORTED_GRADE_CROPS)}")
    ],
) -> QualityResponse:
    crop = crop.strip().lower()
    if crop not in request.app.state.settings.trained_crops:
        raise HTTPException(
            422,
            "Grading is available only for selected supported crops: "
            f"{', '.join(request.app.state.settings.trained_crops)}. Other crops are not graded.",
        )
    grader: QualityGrader | None = request.app.state.grader
    if grader is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, request.app.state.grader_error)
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
        needsReview=prediction.needs_review,
        reviewReason=prediction.review_reason,
        photoSimilarity=prediction.photo_similarity,
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
    location: str | None = Field(
        default=None, max_length=120, description="Farmer's city; its own mandi is used if it reported."
    )


class PriceResponse(BaseModel):
    predictedMinPrice: float
    predictedMaxPrice: float
    unit: str
    currency: Literal["PKR"] = "PKR"
    modelVersion: str
    method: Literal["official_wholesale"] = "official_wholesale"
    commodity: str = Field(description="Official commodity name(s) the range comes from.")
    observedOn: date = Field(description="Mandi trading day of the prices.")
    marketScope: str = Field(description="The farmer's market, or Punjab for the middle half of markets.")
    marketCount: int
    source: str
    sourceUrl: str


@app.post(
    "/predict/price",
    response_model=PriceResponse,
    dependencies=[Depends(require_internal_token)],
)
def predict_price(request: Request, body: PriceRequest) -> PriceResponse:
    prices: OfficialPrices = request.app.state.prices
    try:
        estimate = prices.estimate(
            product_name=body.productName, unit=body.unit, location=body.location
        )
    except PriceNotEstimableError as exc:
        raise HTTPException(422, str(exc)) from exc

    return PriceResponse(
        predictedMinPrice=estimate.min_price,
        predictedMaxPrice=estimate.max_price,
        unit=estimate.unit,
        modelVersion=estimate.version,
        commodity=estimate.commodity,
        observedOn=estimate.observed_on,
        marketScope=estimate.market_scope,
        marketCount=estimate.market_count,
        source=prices.source,
        sourceUrl=prices.source_url,
    )
