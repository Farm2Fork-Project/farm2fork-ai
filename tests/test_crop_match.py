"""Crop-match check: selected-crop probability and the refer-only review reason."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from crop_grading.service.crop_match import (
    CropMatchChecker,
    CropMatchError,
)
from crop_grading.service.grading import REVIEW_CROP_MISMATCH, REVIEW_LOW_CONFIDENCE

NAMES = ("wheat", "rice", "non_crop")


def _checker(direction: tuple[float, float, float], min_probability: float = 0.5):
    embeddings = torch.eye(3)
    vector = torch.tensor(direction)
    return CropMatchChecker(
        encode_image=lambda _image: vector,
        class_names=NAMES,
        class_embeddings=embeddings,
        min_probability=min_probability,
    )


def _image() -> Image.Image:
    return Image.new("RGB", (32, 32), (200, 150, 40))


def test_selected_crop_probability_follows_the_closest_class():
    checker = _checker((1.0, 0.1, 0.0))
    assert checker.probability(_image(), "wheat") > 0.99
    assert checker.probability(_image(), "rice") < 0.01


def test_matches_applies_the_threshold():
    checker = _checker((1.0, 0.0, 0.0), min_probability=0.9)
    assert checker.matches(0.95)
    assert not checker.matches(0.5)


def test_unknown_crop_is_rejected():
    with pytest.raises(ValueError):
        _checker((1.0, 0.0, 0.0)).probability(_image(), "cotton")


def test_missing_weights_fail_loudly(tmp_path):
    with pytest.raises(CropMatchError):
        CropMatchChecker.load(tmp_path / "nope.safetensors", 0.5)


def test_invalid_threshold_fails_loudly(tmp_path):
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"x")
    with pytest.raises(CropMatchError):
        CropMatchChecker.load(weights, 1.5)


@pytest.fixture()
def grader(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_CHECKPOINT", str(tmp_path / "missing.pth"))
    monkeypatch.setenv("ALLOW_UNTRAINED_MODEL", "true")
    from crop_grading.service.app import app

    with TestClient(app) as client:
        client.get("/health")
        yield app.state.grader


def _trained(grader, matcher):
    grader.status = "trained"
    grader.photo_reference = SimpleNamespace(min_similarity=0.5)
    grader.crop_matcher = matcher
    return grader


def test_mismatch_refers_before_other_checks(grader):
    _trained(grader, _checker((0.0, 1.0, 0.0)))
    assert grader._review_reason(0.9, False, 0.01) == REVIEW_CROP_MISMATCH


def test_match_falls_through_to_the_remaining_checks(grader):
    _trained(grader, _checker((1.0, 0.0, 0.0)))
    assert grader._review_reason(0.9, False, 0.99) is None
    assert grader._review_reason(0.9, True, 0.99) == REVIEW_LOW_CONFIDENCE


def test_check_off_changes_nothing(grader):
    _trained(grader, None)
    assert grader._review_reason(0.9, False, None) is None


def test_health_and_response_report_the_check_state(make_client_off):
    with make_client_off() as client:
        assert client.get("/health").json()["cropCheck"] == "disabled"
        buffer = io.BytesIO()
        _image().save(buffer, format="JPEG")
        body = client.post(
            "/predict/quality",
            files={"image": ("a.jpg", buffer.getvalue(), "image/jpeg")},
            data={"crop": "wheat"},
        ).json()
    assert body["cropMatch"] is None


@pytest.fixture()
def make_client_off(monkeypatch, tmp_path):
    def build() -> TestClient:
        monkeypatch.setenv("MODEL_CHECKPOINT", str(tmp_path / "missing.pth"))
        monkeypatch.setenv("ALLOW_UNTRAINED_MODEL", "true")
        monkeypatch.delenv("CROP_MATCH_WEIGHTS", raising=False)
        from crop_grading.service.app import app

        return TestClient(app)

    return build
