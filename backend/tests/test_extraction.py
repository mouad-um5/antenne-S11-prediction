import json
from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.extraction as extraction


FAMILIES = [{
    "id": "antennes-filaires",
    "name": "Antennes filaires",
    "antennas": ["Dipôle demi-onde"],
}]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setattr(extraction, "get_antenna_catalog", lambda: {"families": FAMILIES})
    app = FastAPI()
    app.include_router(extraction.router)
    with TestClient(app) as test_client:
        yield test_client


def gemini_reply(monkeypatch, fields=None, *, text=None, finish="STOP"):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps({
        "candidates": [{
            "finishReason": finish,
            "content": {"parts": [{"text": text if text is not None else json.dumps(fields)}]},
        }],
    }).encode()
    transport = MagicMock(return_value=response)
    monkeypatch.setattr(extraction, "urlopen", transport)
    return transport


def test_partial_extraction_keeps_missing_fields_empty(client, monkeypatch):
    transport = gemini_reply(monkeypatch, {"gap": 39.47, "frequency": 2.4})
    response = client.post("/api/extract-inputs", json={"paragraph": "Gap de 39,47 à 2400 MHz."})
    assert response.status_code == 200
    body = response.json()
    assert body["fields"]["gap"] == 39.47
    assert body["fields"]["frequency"] == 2.4
    assert body["fields"]["surface_width"] is None
    assert body["fields"]["points"] is None
    assert "antenna_family_id" in body["missing_fields"]
    request = transport.call_args.args[0]
    assert request.full_url.endswith("/gemini-3.6-flash:generateContent")
    assert request.get_header("X-goog-api-key") == "test-secret-key"
    sent = json.loads(request.data)
    assert sent["contents"][0]["parts"][0]["text"] == "Gap de 39,47 à 2400 MHz."
    assert sent["generationConfig"]["responseFormat"]["text"]["mimeType"] == "APPLICATION_JSON"
    assert "test-secret-key" not in response.text


def test_catalog_and_invalid_values_are_checked(client, monkeypatch):
    gemini_reply(monkeypatch, {
        "antenna_family_id": "antennes-filaires",
        "antenna_variant": "Variante inventée",
        "gap": -2, "surface_width": True, "epsilon_r": 5,
        "points": 300.5, "threshold_db": 10,
        "start_frequency": 4, "end_frequency": 0.5,
    })
    body = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne."}).json()
    assert body["fields"]["antenna_variant"] is None
    assert body["fields"]["antenna_family_id"] == "antennes-filaires"
    assert body["fields"]["epsilon_r"] == 5
    for name in ("gap", "surface_width", "points", "threshold_db", "end_frequency"):
        assert body["fields"][name] is None
    assert body["warnings"]


def test_unique_variant_resolves_family():
    result = extraction.validate_extraction({"antenna_variant": "Dipôle demi-onde"}, FAMILIES)
    assert result.fields.antenna_family_id == "antennes-filaires"


def test_unknown_family_is_not_silently_replaced():
    result = extraction.validate_extraction({
        "antenna_family_id": "inconnue", "antenna_variant": "Dipôle demi-onde",
    }, FAMILIES)
    assert result.fields.antenna_family_id is None
    assert result.fields.antenna_variant is None


def test_missing_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne."})
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


@pytest.mark.parametrize("paragraph", ["", "          ", "court", "x" * 6001])
def test_input_limits_prevent_upstream_call(client, monkeypatch, paragraph):
    transport = gemini_reply(monkeypatch, {})
    response = client.post("/api/extract-inputs", json={"paragraph": paragraph})
    assert response.status_code == 422
    transport.assert_not_called()


@pytest.mark.parametrize("code,expected", [(429, 429), (403, 503), (404, 503), (500, 502)])
def test_upstream_errors_are_sanitized(client, monkeypatch, code, expected):
    transport = MagicMock(side_effect=HTTPError(
        "https://example.invalid", code, "test-secret-key", {}, None,
    ))
    monkeypatch.setattr(extraction, "urlopen", transport)
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne."})
    assert response.status_code == expected
    assert "test-secret-key" not in response.text
    assert transport.call_count == 1


@pytest.mark.parametrize("failure,expected", [(TimeoutError(), 504), (URLError("secret"), 502)])
def test_connection_errors(client, monkeypatch, failure, expected):
    monkeypatch.setattr(extraction, "urlopen", MagicMock(side_effect=failure))
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne."})
    assert response.status_code == expected


@pytest.mark.parametrize("text,finish", [
    ("not json", "STOP"), ('{"gap": 3}', "MAX_TOKENS"),
    ('{"unexpected": 1}', "STOP"), ("[]", "STOP"),
])
def test_invalid_model_response(client, monkeypatch, text, finish):
    gemini_reply(monkeypatch, text=text, finish=finish)
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne."})
    assert response.status_code == 502


def test_complete_extraction(client, monkeypatch):
    fields = {
        "antenna_family_id": "antennes-filaires", "antenna_variant": "Dipôle demi-onde",
        "gap": 39.47, "surface_width": 75, "surface_length": 75, "epsilon_r": 5,
        "frequency": 2.4, "start_frequency": 0.5, "end_frequency": 4,
        "points": 301, "threshold_db": -10,
    }
    gemini_reply(monkeypatch, fields)
    body = client.post("/api/extract-inputs", json={"paragraph": "Description complète de mon antenne."}).json()
    assert body["fields"] == fields
    assert body["missing_fields"] == []
