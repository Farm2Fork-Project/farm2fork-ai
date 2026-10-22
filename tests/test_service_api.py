"""HTTP-level tests for the inference service (no trained weights required)."""

from __future__ import annotations

import io
import json

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from crop_grading.models.conditioned_model import CropConditionedGradeModel
from crop_grading.models.multitask_model import CropGradingModel


def _jpeg(color=(200, 150, 40), size=(320, 240)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture()
def make_client(monkeypatch, tmp_path):
    def build(**env: str) -> TestClient:
        monkeypatch.setenv("MODEL_CHECKPOINT", str(tmp_path / "missing.pth"))
        monkeypatch.setenv("ALLOW_UNTRAINED_MODEL", "true")
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
    assert body["priceMethod"] == "official_wholesale"


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
            data={"crop": "wheat"},
        ).json()
    assert health["qualityModel"] == "trained"
    assert not health["qualityModelVersion"].endswith("untrained")
    assert graded["modelStatus"] == "trained"
    assert graded["cropSupported"] is True


@pytest.mark.parametrize("crop", ["other", "cotton", "sugarcane", "banana"])
def test_unsupported_crop_never_calls_grader(make_client, monkeypatch, crop):
    from crop_grading.service.grading import QualityGrader

    calls = []
    monkeypatch.setattr(QualityGrader, "grade", lambda *args: calls.append(args))
    with make_client() as client:
        response = client.post(
            "/predict/quality",
            files={"image": ("x.jpg", _jpeg(), "image/jpeg")},
            data={"crop": crop},
        )
    assert response.status_code == 422
    assert calls == []
    assert "qualityGrade" not in response.json()


def test_grade_only_checkpoint_serves_selected_crop_without_crop_recognition(make_client, tmp_path):
    model = CropGradingModel(backbone_name="efficientnet_b0", pretrained=False)
    checkpoint = tmp_path / "grade_only.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    with make_client(MODEL_CHECKPOINT=str(checkpoint)) as client:
        health = client.get("/health").json()
        response = client.post(
            "/predict/quality",
            files={"image": ("rice.jpg", _jpeg(), "image/jpeg")},
            data={"crop": "rice"},
        )
    assert health["qualityModel"] == "trained"
    assert health["qualityModelVersion"].startswith("grade-only-")
    assert response.status_code == 200
    assert response.json()["crop"] == "rice"
    assert response.json()["cropSupported"] is True


def test_direct_grader_rejects_known_but_untrained_crop(make_client):
    from crop_grading.service.grading import QualityGrader
    from crop_grading.service.settings import ServiceSettings

    with make_client():
        grader = QualityGrader(ServiceSettings.from_env())
        with pytest.raises(ValueError, match="not supported"):
            grader.grade(_jpeg(), "cotton")


def test_default_settings_never_enable_random_grades(monkeypatch, tmp_path):
    from crop_grading.service.grading import ModelUnavailableError, QualityGrader
    from crop_grading.service.settings import ServiceSettings

    monkeypatch.delenv("ALLOW_UNTRAINED_MODEL", raising=False)
    monkeypatch.setenv("MODEL_CHECKPOINT", str(tmp_path / "absent.pth"))
    settings = ServiceSettings.from_env()
    assert settings.allow_untrained is False
    with pytest.raises(ModelUnavailableError):
        QualityGrader(settings)


def test_configuration_cannot_claim_training_on_cotton(monkeypatch):
    from crop_grading.service.settings import ServiceSettings

    monkeypatch.setenv("MODEL_TRAINED_CROPS", "wheat,cotton")
    with pytest.raises(ValueError, match="MODEL_TRAINED_CROPS"):
        ServiceSettings.from_env()


def test_mismatched_checkpoint_disables_grading_without_breaking_pricing(make_client, tmp_path):
    checkpoint = tmp_path / "invalid.pth"
    torch.save({"model_state_dict": {"wrong.weight": torch.zeros(1)}}, checkpoint)
    with make_client(MODEL_CHECKPOINT=str(checkpoint), ALLOW_UNTRAINED_MODEL="false") as client:
        assert client.get("/health").json()["qualityModel"] == "unavailable"
        assert (
            client.post(
                "/predict/quality",
                files={"image": ("x.jpg", _jpeg(), "image/jpeg")},
                data={"crop": "wheat"},
            ).status_code
            == 503
        )
        assert (
            client.post(
                "/predict/price",
                json={"productName": "Wheat", "category": "grains", "unit": "kg"},
            ).status_code
            == 200
        )


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


def _official_prices(tmp_path):
    path = tmp_path / "official_prices.json"
    path.write_text(
        json.dumps(
            {
                "source": "AMIS Punjab",
                "source_url": "http://www.amis.pk",
                "price_basis": "test",
                "fetched_at_pkt": "2026-10-22T18:00:00+05:00",
                "commodities": {
                    "Wheat": {
                        "amis_id": 1,
                        "unit": "100kg",
                        "observed_on": "2026-10-22",
                        "markets": {
                            "Faisalabad": {"min": 12750, "max": 13000, "fqp": 12875},
                            "Rawalpindi": {"min": 13200, "max": 15000, "fqp": 14100},
                            "Sahiwal": {"min": 11000, "max": 11300, "fqp": 11150},
                            "Layyah": {"min": 11700, "max": 12500, "fqp": 12100},
                            "BahawalPur": {"min": 9500, "max": 9600, "fqp": 9550},
                        },
                    },
                    "Mango(Chounsa)": {
                        "amis_id": 48,
                        "unit": "100kg",
                        "observed_on": "2026-09-10",
                        "markets": {"Multan": {"min": 22000, "max": 23000, "fqp": 22500}},
                    },
                    "Mango(Sindhri)": {
                        "amis_id": 56,
                        "unit": "100kg",
                        "observed_on": "2026-08-01",
                        "markets": {"Lahore": {"min": 15000, "max": 16000, "fqp": 15500}},
                    },
                    "Banana(DOZEN)": {
                        "amis_id": 32,
                        "unit": "dozen",
                        "observed_on": "2026-10-22",
                        "markets": {"Lahore": {"min": 125, "max": 142, "fqp": 134}},
                    },
                    "Green Chilli": {
                        "amis_id": 77,
                        "unit": "100kg",
                        "observed_on": "2026-10-22",
                        "markets": {"Lahore": {"min": 10900, "max": 11400, "fqp": 11150}},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _price(client, **body):
    return client.post("/predict/price", json={"category": "grains", "unit": "kg", **body})


def test_price_is_the_middle_of_dated_official_market_prices(make_client, tmp_path):
    with make_client(OFFICIAL_PRICES=str(_official_prices(tmp_path))) as client:
        per_kg = _price(client, productName="Desi Gandum").json()
        per_ton = _price(client, productName="Wheat", unit="ton").json()
        health = client.get("/health").json()
    assert per_kg["method"] == "official_wholesale"
    assert per_kg["commodity"] == "Wheat"
    assert per_kg["observedOn"] == "2026-10-22"
    assert per_kg["marketScope"] == "Punjab"
    assert per_kg["marketCount"] == 5
    assert per_kg["source"] == "AMIS Punjab"
    # Quartiles of fair-quality prices (Rs/100kg) converted to Rs/kg.
    assert per_kg["predictedMinPrice"] == pytest.approx(111.5)
    assert per_kg["predictedMaxPrice"] == pytest.approx(128.75)
    assert per_ton["predictedMinPrice"] == pytest.approx(per_kg["predictedMinPrice"] * 1000)
    assert health["priceMethod"] == "official_wholesale"
    assert health["priceModelVersion"] == "amis-punjab-2026-10-22"


def test_price_uses_the_farmers_own_market_when_it_reported(make_client, tmp_path):
    with make_client(OFFICIAL_PRICES=str(_official_prices(tmp_path))) as client:
        local = _price(client, productName="Wheat", location="Faisalabad").json()
        elsewhere = _price(client, productName="Wheat", location="Okara").json()
    assert local["marketScope"] == "Faisalabad"
    assert (local["predictedMinPrice"], local["predictedMaxPrice"]) == (127.5, 130.0)
    assert elsewhere["marketScope"] == "Punjab"


def test_out_of_season_crop_reports_its_last_trading_day(make_client, tmp_path):
    with make_client(OFFICIAL_PRICES=str(_official_prices(tmp_path))) as client:
        mango = _price(client, productName="Mangoes", category="fruits").json()
        sindhri = _price(client, productName="Sindhri aam", category="fruits").json()
    assert mango["commodity"] == "Mango(Chounsa)"
    assert mango["observedOn"] == "2026-09-10"
    assert (mango["predictedMinPrice"], mango["predictedMaxPrice"]) == (225.0, 225.0)
    assert sindhri["commodity"] == "Mango(Sindhri)"
    assert sindhri["observedOn"] == "2026-08-01"


def test_price_refuses_products_and_units_without_official_prices(make_client, tmp_path):
    with make_client(OFFICIAL_PRICES=str(_official_prices(tmp_path))) as client:
        eggs = _price(client, productName="Desi eggs", category="dairy", unit="dozen")
        wheat_piece = _price(client, productName="Wheat", unit="piece")
        bananas = _price(client, productName="Bananas", category="fruits", unit="dozen")
        bananas_kg = _price(client, productName="Bananas", category="fruits")
        chilli = _price(client, productName="Green chillies", category="vegetables")
        greens = _price(client, productName="Green peas", category="vegetables")
    assert eggs.status_code == 422
    assert wheat_piece.status_code == 422
    assert bananas.json()["predictedMinPrice"] == 134.0
    assert bananas_kg.status_code == 422
    assert chilli.json()["commodity"] == "Green Chilli"
    assert greens.status_code == 422


def test_shipped_snapshot_prices_the_health_probe_product(make_client):
    with make_client() as client:
        mango = _price(client, productName="mango", category="fruits")
    assert mango.status_code == 200
    assert mango.json()["predictedMaxPrice"] >= mango.json()["predictedMinPrice"] > 0


def _grade_only_checkpoint(tmp_path):
    model = CropGradingModel(backbone_name="efficientnet_b0", pretrained=False)
    checkpoint = tmp_path / "grade_only.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    return model, checkpoint


def _reference_for(model, tmp_path, *, image: bytes, min_similarity: float, fingerprint=None):
    from crop_grading.data.transforms import build_eval_transforms
    from crop_grading.service.photo_reference import save_reference, weights_fingerprint

    model.eval()
    tensor = build_eval_transforms(224)(Image.open(io.BytesIO(image)).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        embedding = torch.nn.functional.normalize(model.backbone(tensor), dim=1).numpy()
    path = tmp_path / "photo_reference.npz"
    save_reference(
        path,
        embeddings=embedding,
        k=5,
        min_similarity=min_similarity,
        fingerprint=fingerprint or weights_fingerprint(model.state_dict()),
        provenance={"test": True},
    )
    return path


def _grade(client, image: bytes, crop="wheat"):
    return client.post(
        "/predict/quality", files={"image": ("p.jpg", image, "image/jpeg")}, data={"crop": crop}
    ).json()


def test_trained_model_without_photo_reference_refers_every_photo(make_client, tmp_path):
    _, checkpoint = _grade_only_checkpoint(tmp_path)
    with make_client(
        MODEL_CHECKPOINT=str(checkpoint), MODEL_PHOTO_REFERENCE=str(tmp_path / "absent.npz")
    ) as client:
        health = client.get("/health").json()
        body = _grade(client, _jpeg())
    assert health["photoCheck"] == "unavailable"
    assert body["needsReview"] is True
    assert body["reviewReason"] == "photo_check_unavailable"


def test_reference_for_other_weights_is_not_used(make_client, tmp_path):
    model, checkpoint = _grade_only_checkpoint(tmp_path)
    reference = _reference_for(model, tmp_path, image=_jpeg(), min_similarity=0.0, fingerprint="0" * 64)
    with make_client(MODEL_CHECKPOINT=str(checkpoint), MODEL_PHOTO_REFERENCE=str(reference)) as client:
        assert client.get("/health").json()["photoCheck"] == "unavailable"
        assert _grade(client, _jpeg())["reviewReason"] == "photo_check_unavailable"


def test_unfamiliar_photo_is_referred_even_when_confident(make_client, tmp_path):
    model, checkpoint = _grade_only_checkpoint(tmp_path)
    reference = _reference_for(model, tmp_path, image=_jpeg(), min_similarity=1.01)
    with make_client(MODEL_CHECKPOINT=str(checkpoint), MODEL_PHOTO_REFERENCE=str(reference)) as client:
        client.app.state.grader.confidence_threshold = 0.0
        body = _grade(client, _jpeg())
    assert body["needsReview"] is True
    assert body["reviewReason"] == "unfamiliar_photo"
    assert body["photoSimilarity"] < 1.01


def test_familiar_photo_is_graded_only_when_confident(make_client, tmp_path):
    model, checkpoint = _grade_only_checkpoint(tmp_path)
    reference = _reference_for(model, tmp_path, image=_jpeg(), min_similarity=0.9)
    with make_client(MODEL_CHECKPOINT=str(checkpoint), MODEL_PHOTO_REFERENCE=str(reference)) as client:
        assert client.get("/health").json()["photoCheck"] == "enabled"
        client.app.state.grader.confidence_threshold = 1.01
        unsure = _grade(client, _jpeg())
        client.app.state.grader.confidence_threshold = 0.0
        sure = _grade(client, _jpeg())
    assert unsure["needsReview"] is True and unsure["reviewReason"] == "low_confidence"
    assert sure["needsReview"] is False and sure["reviewReason"] is None
    assert sure["photoSimilarity"] > 0.99


def test_untrained_preview_is_not_gated(make_client):
    with make_client() as client:
        body = _grade(client, _jpeg())
    assert body["modelStatus"] == "untrained"
    assert body["needsReview"] is False
