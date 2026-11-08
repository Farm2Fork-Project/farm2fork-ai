"""Fail-closed release configuration and official-data validation."""

import json
from pathlib import Path

import pytest

from crop_grading.service.pricing import OfficialPrices
from crop_grading.service.settings import ServiceSettings


def snapshot(tmp_path: Path, **changes) -> Path:
    document = {
        "source": "AMIS Punjab",
        "source_url": "http://www.amis.pk",
        "fetched_at_pkt": "2026-10-22T18:00:00+05:00",
        "commodities": {
            "Wheat": {
                "unit": "100kg",
                "observed_on": "2026-10-22",
                "markets": {"Lahore": {"min": 100, "max": 120, "fqp": 110}},
            }
        },
    }
    document.update(changes)
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize(
    "change",
    [
        {"commodities": {}},
        {"source_url": "javascript:alert(1)"},
        {"fetched_at_pkt": "invalid"},
        {"commodities": {"Wheat": {"unit": "100kg", "observed_on": "2026-10-22", "markets": {}}}},
        {
            "commodities": {
                "Wheat": {
                    "unit": "100kg",
                    "observed_on": "2026-10-23",
                    "markets": {"Lahore": {"min": 100, "max": 120, "fqp": 110}},
                }
            }
        },
        {
            "commodities": {
                "Wheat": {
                    "unit": "100kg",
                    "observed_on": "2026-10-22",
                    "markets": {"Lahore": {"min": 200, "max": 120, "fqp": 110}},
                }
            }
        },
    ],
)
def test_invalid_official_snapshot_fails_before_serving(tmp_path, change):
    with pytest.raises(ValueError):
        OfficialPrices(snapshot(tmp_path, **change))


def test_production_requires_internal_auth_and_forbids_random_model(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("AI_SERVICE_TOKEN", raising=False)
    with pytest.raises(ValueError, match="AI_SERVICE_TOKEN"):
        ServiceSettings.from_env()
    monkeypatch.setenv("AI_SERVICE_TOKEN", "release-internal-token-example")
    monkeypatch.setenv("ALLOW_UNTRAINED_MODEL", "true")
    with pytest.raises(ValueError, match="ALLOW_UNTRAINED_MODEL"):
        ServiceSettings.from_env()


@pytest.mark.parametrize(
    "embeddings,k,threshold",
    [
        ([], 1, 0.7),
        ([[float("nan"), 1.0]], 1, 0.7),
        ([[1.0, 0.0]], 0, 0.7),
        ([[10.0, 0.0]], 1, 0.7),
        ([[1.0, 0.0]], 1, float("nan")),
    ],
)
def test_malformed_photo_reference_never_enables_grading(tmp_path, embeddings, k, threshold):
    import numpy as np

    from crop_grading.service.photo_reference import (
        PhotoReference,
        PhotoReferenceError,
        save_reference,
    )

    path = tmp_path / "reference.npz"
    save_reference(
        path,
        embeddings=np.array(embeddings),
        k=k,
        min_similarity=threshold,
        fingerprint="a" * 64,
        provenance={"test": True},
    )
    with pytest.raises(PhotoReferenceError):
        PhotoReference.load(path, "a" * 64)


def test_photo_reference_embedding_dimension_matches_served_backbone(tmp_path):
    import numpy as np

    from crop_grading.service.photo_reference import (
        PhotoReference,
        PhotoReferenceError,
        save_reference,
    )

    path = tmp_path / "reference.npz"
    save_reference(
        path,
        embeddings=np.array([[1.0, 0.0]]),
        k=1,
        min_similarity=0.7,
        fingerprint="b" * 64,
        provenance={"test": True},
    )
    with pytest.raises(PhotoReferenceError, match="dimension"):
        PhotoReference.load(path, "b" * 64, expected_dimension=1280)


def test_official_refresh_never_replaces_last_good_snapshot_with_empty_data(tmp_path):
    from scripts.update_official_prices import write_snapshot

    destination = snapshot(tmp_path)
    previous = destination.read_bytes()
    with pytest.raises(ValueError):
        write_snapshot({"commodities": {}}, destination)
    assert destination.read_bytes() == previous


def test_official_refresh_is_atomic_on_write_failure(tmp_path, monkeypatch):
    import os

    from scripts.update_official_prices import write_snapshot

    destination = snapshot(tmp_path)
    previous = destination.read_bytes()

    def fail_replace(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        write_snapshot(json.loads(previous), destination)
    assert destination.read_bytes() == previous
    assert list(tmp_path.glob(".prices-*.tmp")) == []
