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


def test_model_catalog_does_not_need_api_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.get("/api/extraction-models")
    assert response.status_code == 200
    body = response.json()
    assert {model["id"] for model in body["models"]} == {
        "gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite",
        "openai/gpt-oss-20b", "@cf/meta/llama-3.1-8b-instruct-fast", "mistral-small-latest",
    }
    assert body["default_model"] == "gemini-3.6-flash"
    assert all(model["label"] and model["description"] and model["provider_label"] for model in body["models"])
    assert next(model for model in body["models"] if model["id"] == "openai/gpt-oss-20b")["provider"] == "groq"


@pytest.mark.parametrize("model", [
    "gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite",
])
def test_selected_model_is_used_and_reported(client, monkeypatch, model):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    transport = gemini_reply(monkeypatch, {"gap": 39.47})
    response = client.post("/api/extract-inputs", json={
        "paragraph": "Mon antenne a un gap de 39,47.", "model": model,
    })
    assert response.status_code == 200
    assert response.json()["model"] == model
    assert response.json()["fields"]["gap"] == 39.47
    assert transport.call_args.args[0].full_url.endswith(f"/{model}:generateContent")
    assert transport.call_count == 1


def test_default_model_remains_compatible_with_old_clients(client, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    transport = gemini_reply(monkeypatch, {"frequency": 2.4})
    assert client.get("/api/extraction-models").json()["default_model"] == "gemini-3.8-flash"
    response = client.post("/api/extract-inputs", json={"paragraph": "Frequence de 2,4 GHz."})
    assert response.status_code == 200
    assert response.json()["model"] == "gemini-3.8-flash"
    assert transport.call_args.args[0].full_url.endswith("/gemini-3.8-flash:generateContent")


@pytest.mark.parametrize("model", ["gemini-unknown", "", "../gemini-3.6-flash", "https://example.org", 42])
def test_unknown_models_are_rejected_before_request(client, monkeypatch, model):
    transport = gemini_reply(monkeypatch, {})
    response = client.post("/api/extract-inputs", json={
        "paragraph": "Mon antenne a un gap de 39,47.", "model": model,
    })
    assert response.status_code == 422
    transport.assert_not_called()


def test_invalid_default_is_rejected_but_explicit_selection_works(client, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-unknown")
    assert client.get("/api/extraction-models").status_code == 503
    transport = gemini_reply(monkeypatch, {})
    response = client.post("/api/extract-inputs", json={"paragraph": "Mon antenne a un gap de 39,47."})
    assert response.status_code == 503
    transport.assert_not_called()
    response = client.post("/api/extract-inputs", json={
        "paragraph": "Mon antenne a un gap de 39,47.", "model": "gemini-3.6-flash",
    })
    assert response.status_code == 200


def groq_reply(monkeypatch, fields=None, *, text=None, finish="stop", refusal=None):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-secret")
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps({
        "choices": [{
            "finish_reason": finish,
            "message": {
                "content": text if text is not None else json.dumps(fields),
                "refusal": refusal,
            },
        }],
    }).encode()
    transport = MagicMock(return_value=response)
    monkeypatch.setattr(extraction, "urlopen", transport)
    return transport


def test_groq_routes_to_own_endpoint_and_key_without_gemini_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    transport = groq_reply(monkeypatch, {
        "antenna_variant": "Dipôle demi-onde", "gap": 39.47, "frequency": 2.4,
    })
    paragraph = "Dipôle demi-onde, gap de 39,47 et fréquence de 2400 MHz."
    response = client.post("/api/extract-inputs", json={
        "paragraph": paragraph, "model": "openai/gpt-oss-20b",
    })
    assert response.status_code == 200
    result = response.json()
    assert result["model"] == "openai/gpt-oss-20b"
    assert result["fields"]["antenna_family_id"] == "antennes-filaires"
    assert result["fields"]["gap"] == 39.47
    assert result["fields"]["frequency"] == 2.4
    assert result["fields"]["surface_width"] is None
    request = transport.call_args.args[0]
    assert request.full_url == "https://api.groq.com/openai/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-groq-secret"
    assert request.get_header("User-agent") == "AntennaPrediction/1.0"
    assert request.get_header("X-goog-api-key") is None
    body = json.loads(request.data)
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["messages"][1] == {"role": "user", "content": paragraph}
    output = body["response_format"]
    assert output["type"] == "json_schema"
    assert output["json_schema"]["strict"] is True
    schema = output["json_schema"]["schema"]
    assert set(schema["required"]) == set(extraction.LABELS)
    assert schema["additionalProperties"] is False
    assert "maxLength" not in json.dumps(schema)
    assert "exclusiveMinimum" not in json.dumps(schema)
    assert "test-groq-secret" not in response.text
    assert transport.call_count == 1


def test_groq_missing_key_does_not_fall_back_to_gemini(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    transport = gemini_reply(monkeypatch, {})
    response = client.post("/api/extract-inputs", json={
        "paragraph": "Description de mon antenne.", "model": "openai/gpt-oss-20b",
    })
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]
    transport.assert_not_called()


def test_groq_uses_same_field_and_catalog_validation(client, monkeypatch):
    groq_reply(monkeypatch, {
        "antenna_family_id": "inconnue", "gap": -2, "points": 3,
        "surface_width": 75, "start_frequency": 4, "end_frequency": 0.5,
    })
    response = client.post("/api/extract-inputs", json={
        "paragraph": "Description de mon antenne.", "model": "openai/gpt-oss-20b",
    })
    assert response.status_code == 200
    result = response.json()
    assert result["fields"]["surface_width"] == 75
    for name in ("antenna_family_id", "gap", "points", "end_frequency"):
        assert result["fields"][name] is None
        assert name in result["missing_fields"]
    assert result["warnings"]


@pytest.mark.parametrize("text,finish,refusal", [
    ("not json", "stop", None),
    ('{"gap": 3}', "length", None),
    ('{"gap": 3}', "stop", "Cannot comply"),
    ('{"unexpected": 1}', "stop", None),
    ("[]", "stop", None),
    ("null", "stop", None),
])
def test_groq_rejects_incomplete_or_invalid_response(client, monkeypatch, text, finish, refusal):
    groq_reply(monkeypatch, text=text, finish=finish, refusal=refusal)
    response = client.post("/api/extract-inputs", json={
        "paragraph": "Description de mon antenne.", "model": "openai/gpt-oss-20b",
    })
    assert response.status_code == 502
    assert "Groq" in response.json()["detail"]


@pytest.mark.parametrize("code,expected", [(400, 502), (401, 503), (403, 503), (404, 503), (429, 429), (500, 502)])
def test_groq_errors_do_not_expose_keys_or_retry(client, monkeypatch, code, expected):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-secret")
    transport = MagicMock(side_effect=HTTPError(
        "https://api.groq.com", code, "test-groq-secret", {}, None,
    ))
    monkeypatch.setattr(extraction, "urlopen", transport)
    response = client.post("/api/extract-inputs", json={
        "paragraph": "Description de mon antenne.", "model": "openai/gpt-oss-20b",
    })
    assert response.status_code == expected
    assert "Groq" in response.json()["detail"]
    assert "test-groq-secret" not in response.text
    assert transport.call_count == 1


@pytest.mark.parametrize("failure,expected", [(TimeoutError(), 504), (URLError("secret"), 502)])
def test_groq_connection_errors(client, monkeypatch, failure, expected):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-secret")
    monkeypatch.setattr(extraction, "urlopen", MagicMock(side_effect=failure))
    response = client.post("/api/extract-inputs", json={
        "paragraph": "Description de mon antenne.", "model": "openai/gpt-oss-20b",
    })
    assert response.status_code == expected


def test_groq_can_be_default_model(client, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "openai/gpt-oss-20b")
    groq_reply(monkeypatch, {"frequency": 2.4})
    assert client.get("/api/extraction-models").json()["default_model"] == "openai/gpt-oss-20b"
    response = client.post("/api/extract-inputs", json={"paragraph": "Fréquence de 2,4 GHz."})
    assert response.status_code == 200
    assert response.json()["model"] == "openai/gpt-oss-20b"


NEW_MODELS = ["@cf/meta/llama-3.1-8b-instruct-fast", "mistral-small-latest"]


def provider_reply(monkeypatch, model, fields=None, *, payload=None, raw=None):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-cloudflare-secret")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-secret")
    if payload is None:
        if model == NEW_MODELS[0]:
            payload = {"success": True, "errors": [], "result": {"response": json.dumps(fields)}}
        else:
            payload = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(fields)}}]}
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = raw if raw is not None else json.dumps(payload).encode()
    transport = MagicMock(return_value=response)
    monkeypatch.setattr(extraction, "urlopen", transport)
    return transport


@pytest.mark.parametrize("model", NEW_MODELS)
def test_new_provider_routing_and_validation(client, monkeypatch, model):
    transport = provider_reply(monkeypatch, model, {
        "antenna_variant": "Dipôle demi-onde", "frequency": 2.4, "gap": -1,
        "points": True, "surface_width": 75, "start_frequency": 4, "end_frequency": 1,
    })
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    paragraph = "Dipôle demi-onde à 2400 MHz et largeur 75."
    response = client.post("/api/extract-inputs", json={"paragraph": paragraph, "model": model})
    assert response.status_code == 200
    result = response.json()
    assert result["model"] == model
    assert result["fields"]["antenna_family_id"] == "antennes-filaires"
    assert result["fields"]["frequency"] == 2.4
    assert result["fields"]["surface_width"] == 75
    for name in ("gap", "points", "end_frequency", "surface_length"):
        assert result["fields"][name] is None
        assert name in result["missing_fields"]
    request = transport.call_args.args[0]
    body = json.loads(request.data)
    assert body["messages"][1] == {"role": "user", "content": paragraph}
    assert "Dipôle demi-onde" in body["messages"][0]["content"]
    assert body["stream"] is False
    assert request.get_header("X-goog-api-key") is None
    if model == NEW_MODELS[0]:
        assert request.full_url == "https://api.cloudflare.com/client/v4/accounts/" + "a" * 32 + "/ai/run/" + model
        assert request.get_header("Authorization") == "Bearer test-cloudflare-secret"
        assert "response_format" not in body
        assert all(name in body["messages"][0]["content"] for name in extraction.LABELS)
    else:
        assert request.full_url == "https://api.mistral.ai/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer test-mistral-secret"
        assert body["model"] == model
        output = body["response_format"]
        assert output["type"] == "json_schema"
        assert output["json_schema"]["strict"] is True
        schema = output["json_schema"]["schema"]
        assert set(schema["required"]) == set(extraction.LABELS)
        assert schema["additionalProperties"] is False
    assert transport.call_count == 1
    assert "secret" not in response.text


@pytest.mark.parametrize("model,missing", [
    (NEW_MODELS[0], "CLOUDFLARE_ACCOUNT_ID"),
    (NEW_MODELS[0], "CLOUDFLARE_API_TOKEN"),
    (NEW_MODELS[1], "MISTRAL_API_KEY"),
])
def test_new_provider_missing_credentials_no_fallback(client, monkeypatch, model, missing):
    transport = provider_reply(monkeypatch, model, {})
    monkeypatch.delenv(missing)
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne.", "model": model})
    assert response.status_code == 503
    assert missing in response.json()["detail"]
    transport.assert_not_called()


@pytest.mark.parametrize("account", ["../../other", "https://example.org", "a" * 31, "g" * 32])
def test_cloudflare_account_is_validated_before_request(client, monkeypatch, account):
    transport = provider_reply(monkeypatch, NEW_MODELS[0], {})
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", account)
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne.", "model": NEW_MODELS[0]})
    assert response.status_code == 503
    transport.assert_not_called()


@pytest.mark.parametrize("model", NEW_MODELS)
@pytest.mark.parametrize("code,expected", [(400, 502), (401, 503), (403, 503), (404, 503), (429, 429), (500, 502)])
def test_new_provider_errors_are_redacted_without_retry(client, monkeypatch, model, code, expected):
    transport = provider_reply(monkeypatch, model, {})
    transport.side_effect = HTTPError("https://example.invalid", code, "private-provider-secret", {}, None)
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne.", "model": model})
    assert response.status_code == expected
    assert "secret" not in response.text
    assert transport.call_count == 1


@pytest.mark.parametrize("model", NEW_MODELS)
@pytest.mark.parametrize("failure,expected", [(TimeoutError(), 504), (URLError("secret"), 502)])
def test_new_provider_connection_errors(client, monkeypatch, model, failure, expected):
    transport = provider_reply(monkeypatch, model, {})
    transport.side_effect = failure
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne.", "model": model})
    assert response.status_code == expected
    assert "secret" not in response.text


@pytest.mark.parametrize("model", NEW_MODELS)
@pytest.mark.parametrize("raw", [b"not json", b"[]", b"null", b"{}", b"\xff"])
def test_new_provider_malformed_envelopes(client, monkeypatch, model, raw):
    provider_reply(monkeypatch, model, raw=raw)
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne.", "model": model})
    assert response.status_code == 502


@pytest.mark.parametrize("model", NEW_MODELS)
@pytest.mark.parametrize("content", ['{"gap":', '{"unknown": 3}', "[]", "null", "Cannot comply"])
def test_new_provider_invalid_content(client, monkeypatch, model, content):
    if model == NEW_MODELS[0]:
        payload = {"success": True, "result": {"response": content}}
    else:
        payload = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
    provider_reply(monkeypatch, model, payload=payload)
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne.", "model": model})
    assert response.status_code == 502


@pytest.mark.parametrize("payload", [
    {"success": False, "errors": [{"message": "private-provider-secret"}]},
    {"success": True, "errors": [{"message": "private-provider-secret"}], "result": {"response": "{}"}},
    {"success": True, "result": {"response": '{"gap": 2}', "finish_reason": "length"}},
    {"success": True, "result": {"response": "{}", "tool_calls": [{"name": "unknown"}]}},
])
def test_cloudflare_error_envelopes_and_truncation(client, monkeypatch, payload):
    provider_reply(monkeypatch, NEW_MODELS[0], payload=payload)
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne.", "model": NEW_MODELS[0]})
    assert response.status_code == 502
    assert "private-provider-secret" not in response.text


@pytest.mark.parametrize("content", ['\x60\x60\x60json\n{"frequency": 2.4}\n\x60\x60\x60', {"frequency": 2.4}])
def test_cloudflare_json_fences_and_object(client, monkeypatch, content):
    provider_reply(monkeypatch, NEW_MODELS[0], payload={"success": True, "result": {"response": content}})
    response = client.post("/api/extract-inputs", json={"paragraph": "Fréquence de 2,4 GHz.", "model": NEW_MODELS[0]})
    assert response.status_code == 200
    assert response.json()["fields"]["frequency"] == 2.4


@pytest.mark.parametrize("finish,message", [
    ("length", {"content": '{"gap": 2}'}),
    ("stop", {"content": '{"gap": 2}', "refusal": "Cannot comply"}),
    ("stop", {"content": "{}", "tool_calls": [{"name": "unknown"}]}),
])
def test_mistral_rejects_truncation_and_refusal(client, monkeypatch, finish, message):
    provider_reply(monkeypatch, NEW_MODELS[1], payload={"choices": [{"finish_reason": finish, "message": message}]})
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne.", "model": NEW_MODELS[1]})
    assert response.status_code == 502


def test_mistral_text_chunks(client, monkeypatch):
    provider_reply(monkeypatch, NEW_MODELS[1], payload={"choices": [{
        "finish_reason": "stop",
        "message": {"content": [
            {"type": "thinking", "thinking": "ignored"},
            {"type": "text", "text": '{"frequency": 2.4}'},
        ]},
    }]})
    response = client.post("/api/extract-inputs", json={"paragraph": "Fréquence de 2,4 GHz.", "model": NEW_MODELS[1]})
    assert response.status_code == 200
    assert response.json()["fields"]["frequency"] == 2.4


@pytest.mark.parametrize("model", NEW_MODELS)
def test_new_provider_can_be_default(client, monkeypatch, model):
    monkeypatch.setenv("GEMINI_MODEL", model)
    provider_reply(monkeypatch, model, {"frequency": 2.4})
    assert client.get("/api/extraction-models").json()["default_model"] == model
    response = client.post("/api/extract-inputs", json={"paragraph": "Fréquence de 2,4 GHz."})
    assert response.status_code == 200
    assert response.json()["model"] == model


@pytest.mark.parametrize("model,credential", [
    (NEW_MODELS[0], "CLOUDFLARE_API_TOKEN"),
    (NEW_MODELS[1], "MISTRAL_API_KEY"),
])
def test_authentication_errors_identify_configuration(client, monkeypatch, model, credential, caplog):
    transport = provider_reply(monkeypatch, model, {})
    transport.side_effect = HTTPError("https://example.invalid", 401, "private-secret", {}, None)
    response = client.post("/api/extract-inputs", json={"paragraph": "private-user-paragraph", "model": model})
    assert response.status_code == 503
    assert "401" in response.json()["detail"]
    assert credential in response.json()["detail"]
    assert model in caplog.text
    assert "upstream_status=401" in caplog.text
    assert "private-secret" not in caplog.text + response.text
    assert "private-user-paragraph" not in caplog.text


@pytest.mark.parametrize("retry_after,expected", [("30", "30"), ("", None), ("private-secret", None), ("999999999999", None)])
def test_rate_limit_delay_is_preserved_only_when_valid(client, monkeypatch, retry_after, expected):
    transport = provider_reply(monkeypatch, NEW_MODELS[1], {})
    transport.side_effect = HTTPError("https://example.invalid", 429, "private-secret", {"Retry-After": retry_after}, None)
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne.", "model": NEW_MODELS[1]})
    assert response.status_code == 429
    assert response.headers.get("Retry-After") == expected
    detail = response.json()["detail"]
    assert "Mistral" in detail
    assert "30 secondes" in detail if expected else "ne précise pas de délai" in detail
    assert "private-secret" not in response.text
    assert transport.call_count == 1


def test_cloudflare_permission_error_is_actionable(client, monkeypatch):
    transport = provider_reply(monkeypatch, NEW_MODELS[0], {})
    transport.side_effect = HTTPError("https://example.invalid", 403, "private-secret", {}, None)
    response = client.post("/api/extract-inputs", json={"paragraph": "Description de mon antenne.", "model": NEW_MODELS[0]})
    assert response.status_code == 503
    assert "Workers AI Read et Edit" in response.json()["detail"]
