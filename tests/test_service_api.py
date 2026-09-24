"""HTTP-level tests for the inference service (no trained weights required)."""

from __future__ import annotations

import io
from datetime import date

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from crop_grading.models.conditioned_model import CropConditionedGradeModel


def _jpeg(color=(200, 150, 40), size=(320, 240)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture()
def make_client(monkeypatch, tmp_path):
    def build(**env: str) -> TestClient:
        monkeypatch.setenv("MODEL_CHECKPOINT", str(tmp_path / "missing.pth"))
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        from crop_grading.service.app import app

        return TestClient(app)

    return build


def test_health_reports_an_untrained_model_honestly(make_client):
    with make_client() as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["qualityModel"] == "untrained"
    assert body["qualityModelVersion"].endswith("-untrained")
    assert body["priceMethod"] == "rule_based"


def test_quality_prediction_is_labelled_untrained_without_weights(make_client):
    with make_client() as client:
        response = client.post(
            "/predict/quality",
            files={"image": ("mango.jpg", _jpeg(), "image/jpeg")},
            data={"crop": "Mango"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["modelStatus"] == "untrained"
    assert body["qualityGrade"] in {"A", "B", "C", "D"}
    assert set(body["probabilities"]) == {"A", "B", "C", "D"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["crop"] == "mango"
    assert body["cropSupported"] is True


def test_loads_a_training_checkpoint_and_reports_trained(make_client, tmp_path):
    model = CropConditionedGradeModel(backbone_name="efficientnet_b0", pretrained=False)
    checkpoint = tmp_path / "best_model.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)

    with make_client(MODEL_CHECKPOINT=str(checkpoint)) as client:
        health = client.get("/health").json()
        graded = client.post(
            "/predict/quality",
            files={"image": ("wheat.png", _jpeg(), "image/jpeg")},
            data={"crop": "cotton"},
        ).json()
    assert health["qualityModel"] == "trained"
    assert not health["qualityModelVersion"].endswith("untrained")
    assert graded["modelStatus"] == "trained"
    # Cotton has no training data yet: the service says so.
    assert graded["cropSupported"] is False


def test_refuses_to_serve_untrained_when_disabled(make_client):
    with make_client(ALLOW_UNTRAINED_MODEL="false") as client:
        assert client.get("/health").json()["qualityModel"] == "unavailable"
        response = client.post(
            "/predict/quality",
            files={"image": ("x.jpg", _jpeg(), "image/jpeg")},
            data={"crop": "rice"},
        )
        assert response.status_code == 503
        # Pricing does not depend on the vision model.
        assert (
            client.post(
                "/predict/price",
                json={"productName": "Basmati", "category": "grains", "unit": "kg"},
            ).status_code
            == 200
        )


@pytest.mark.parametrize(
    ("files", "data", "expected"),
    [
        ({"image": ("x.jpg", _jpeg(), "image/jpeg")}, {"crop": "banana"}, 422),
        ({"image": ("x.txt", b"hello", "text/plain")}, {"crop": "rice"}, 415),
        ({"image": ("x.jpg", b"not really a jpeg", "image/jpeg")}, {"crop": "rice"}, 400),
    ],
)
def test_rejects_bad_quality_requests(make_client, files, data, expected):
    with make_client() as client:
        assert client.post("/predict/quality", files=files, data=data).status_code == expected


def test_rejects_oversized_photos(make_client):
    with make_client(MAX_IMAGE_BYTES="1000") as client:
        response = client.post(
            "/predict/quality",
            files={"image": ("big.jpg", _jpeg(size=(800, 800)), "image/jpeg")},
            data={"crop": "rice"},
        )
    assert response.status_code == 413


def test_internal_token_is_enforced_when_configured(make_client):
    with make_client(AI_SERVICE_TOKEN="s3cret") as client:
        body = {"productName": "Wheat", "category": "grains", "unit": "kg"}
        assert client.post("/predict/price", json=body).status_code == 401
        assert (
            client.post(
                "/predict/price", json=body, headers={"X-Internal-Token": "wrong"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/predict/price", json=body, headers={"X-Internal-Token": "s3cret"}
            ).status_code
            == 200
        )
        # Health stays open for container probes.
        assert client.get("/health").status_code == 200


def test_price_estimate_uses_crop_season_grade_and_unit(make_client):
    with make_client() as client:
        per_kg = client.post(
            "/predict/price",
            json={
                "productName": "Chaunsa Mangoes",
                "category": "fruits",
                "unit": "kg",
                "qualityGrade": "A",
                "asOf": date(2026, 6, 15).isoformat(),
            },
        ).json()
        per_ton = client.post(
            "/predict/price",
            json={
                "productName": "Chaunsa Mangoes",
                "category": "fruits",
                "unit": "ton",
                "qualityGrade": "A",
                "asOf": date(2026, 6, 15).isoformat(),
            },
        ).json()
    assert per_kg["method"] == "rule_based"
    assert per_kg["basis"] == "crop"
    assert per_kg["matchedCrop"] == "mango"
    assert per_kg["seasonalFactor"] == 0.85  # June: mango harvest
    assert per_kg["gradeFactor"] == 1.1
    assert per_kg["confidenceScore"] <= 0.5
    assert per_kg["predictedMinPrice"] < per_kg["predictedMaxPrice"]
    assert per_ton["predictedMinPrice"] == pytest.approx(per_kg["predictedMinPrice"] * 1000)


def test_price_falls_back_to_category_and_refuses_unconvertible_units(make_client):
    with make_client() as client:
        tomatoes = client.post(
            "/predict/price",
            json={"productName": "Desi Tomatoes", "category": "vegetables", "unit": "kg"},
        ).json()
        dozen = client.post(
            "/predict/price",
            json={"productName": "Eggs", "category": "dairy", "unit": "dozen"},
        )
    assert tomatoes["basis"] == "category"
    assert tomatoes["confidenceScore"] < 0.45
    assert dozen.status_code == 422
