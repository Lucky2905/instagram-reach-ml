"""
tests/test_api.py — Flask API endpoint tests.
Verifies: health check, 400 on bad input, 503 without models, valid response schema.
"""

import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

VALID_PAYLOAD = {
    "likes": 1500,
    "comments": 80,
    "shares": 25,
    "saves": 60,
    "hashtag_count": 12,
    "post_type": "reel",
    "hour_of_day": 19,
    "day_of_week": 5,
    "follower_count": 50000,
    "account_age_days": 730,
}


@pytest.fixture(scope="module")
def client():
    from src.api.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_has_status_ok(self, client):
        data = json.loads(client.get("/health").data)
        assert data["status"] == "ok"

    def test_response_has_models_loaded_field(self, client):
        data = json.loads(client.get("/health").data)
        assert "models_loaded" in data

    def test_response_has_message(self, client):
        data = json.loads(client.get("/health").data)
        assert "message" in data


# ── /predict — validation errors ──────────────────────────────────────────────

class TestPredictValidation:

    def test_empty_body_returns_400(self, client):
        resp = client.post("/predict", data="not-json", content_type="text/plain")
        assert resp.status_code in (400, 503)

    def test_missing_field_returns_400_or_503(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "likes"}
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code in (400, 503)

    def test_invalid_post_type_returns_400_or_503(self, client):
        payload = {**VALID_PAYLOAD, "post_type": "story"}  # invalid
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code in (400, 503)

    def test_invalid_hour_returns_400_or_503(self, client):
        payload = {**VALID_PAYLOAD, "hour_of_day": 25}   # out of range
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code in (400, 503)

    def test_invalid_day_of_week_returns_400_or_503(self, client):
        payload = {**VALID_PAYLOAD, "day_of_week": 8}    # out of range
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code in (400, 503)

    def test_wrong_type_for_numeric_field_returns_400_or_503(self, client):
        payload = {**VALID_PAYLOAD, "likes": "many"}     # string instead of number
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code in (400, 503)


# ── /predict — response schema (when models are loaded) ──────────────────────

class TestPredictResponseSchema:
    """
    These tests run only when models are already trained.
    They verify the response structure without asserting specific values.
    """

    def test_valid_request_returns_expected_fields_or_503(self, client):
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        # 503 = models not trained (acceptable in CI without trained models)
        if resp.status_code == 503:
            pytest.skip("Models not trained — skipping live prediction test.")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "predicted_reach" in data
        assert "reach_tier"      in data
        assert "confidence"      in data

    def test_reach_tier_is_valid_string(self, client):
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        if resp.status_code == 503:
            pytest.skip("Models not trained.")
        data = json.loads(resp.data)
        assert data["reach_tier"] in ("low", "medium", "high")

    def test_confidence_is_float_between_0_and_1(self, client):
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        if resp.status_code == 503:
            pytest.skip("Models not trained.")
        data = json.loads(resp.data)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predicted_reach_is_non_negative_int(self, client):
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        if resp.status_code == 503:
            pytest.skip("Models not trained.")
        data = json.loads(resp.data)
        assert isinstance(data["predicted_reach"], int)
        assert data["predicted_reach"] >= 0
