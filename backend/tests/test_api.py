import os

os.environ["APP_ENV"] = "test"

import pytest
from fastapi.testclient import TestClient

import app.main as api_main
from app.main import app
from app.settings import Settings


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def prediction_payload(**overrides) -> dict:
    payload = {
        "antenna_family_id": "antennes-filaires",
        "antenna_variant": "Dipôle demi-onde",
        "gap": 39.47,
        "surface_width": 75,
        "surface_length": 75,
        "epsilon_r": 5,
        "frequency": 2.4,
    }
    payload.update(overrides)
    return payload


def test_health_and_readiness(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = client.get("/api/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["scope_verified"] is False


def test_catalog_is_loaded_from_excel(client: TestClient) -> None:
    response = client.get("/api/antennas")
    assert response.status_code == 200
    payload = response.json()
    assert payload["family_count"] == 16
    assert payload["antenna_count"] == 39
    assert any(family["id"] == "patch-micro-ruban" for family in payload["families"])


def test_point_prediction_has_traceability(client: TestClient) -> None:
    response = client.post("/api/predict", json=prediction_payload())
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["s11_db"], float)
    assert payload["model_id"].startswith("s11-xgb-")
    assert payload["scientific_status"] == "internal_grouped_validation_only"
    assert payload["model_scope_verified"] is False
    assert payload["estimated_absolute_error_p95_db"] > 0
    assert payload["known_training_geometry"] is True
    assert any("CST" in warning for warning in payload["warnings"])


def test_extrapolation_is_reported(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json=prediction_payload(gap=60),
    )
    assert response.status_code == 200
    assert response.json()["is_extrapolation"] is True
    assert any("gap" in warning for warning in response.json()["warnings"])


def test_unseen_width_length_relation_is_reported(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json=prediction_payload(surface_width=70, surface_length=80),
    )
    assert response.status_code == 200
    assert response.json()["is_extrapolation"] is True
    assert any(
        "largeur et la longueur" in warning
        for warning in response.json()["warnings"]
    )


def test_sweep_returns_bands_and_minimum(client: TestClient) -> None:
    payload = prediction_payload()
    payload.pop("frequency")
    payload.update(
        {
            "start_frequency": 0.5,
            "end_frequency": 4.0,
            "points": 301,
            "threshold_db": -10,
        }
    )
    response = client.post("/api/predict-sweep", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["curve"]) == 301
    assert body["minimum"] == min(body["curve"], key=lambda point: point["s11"])
    assert body["threshold_db"] == -10
    assert body["total_bandwidth"] == pytest.approx(
        sum(band["bandwidth"] for band in body["bands"])
    )


def test_invalid_antenna_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json=prediction_payload(antenna_variant="Inconnue"),
    )
    assert response.status_code == 422


def test_invalid_dimensions_and_extra_fields_are_rejected(client: TestClient) -> None:
    invalid = client.post(
        "/api/predict",
        json=prediction_payload(gap=-1),
    )
    assert invalid.status_code == 422

    extra = client.post(
        "/api/predict",
        json=prediction_payload(unexpected="value"),
    )
    assert extra.status_code == 422


def test_production_guard_blocks_unverified_model(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main,
        "SETTINGS",
        Settings(
            environment="production",
            cors_origins=(),
            allowed_hosts=("testserver",),
            enable_docs=False,
            require_verified_model_scope=True,
        ),
    )
    with TestClient(app) as guarded_client:
        response = guarded_client.post(
            "/api/predict",
            json=prediction_payload(),
        )
    assert response.status_code == 503
    assert "aucune famille" in response.json()["detail"]
