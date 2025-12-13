from fastapi.testclient import TestClient

from src.api.main import app, evaluate_content_quality


class _MockModel:
    def __init__(self, response):
        self._response = response
        self.received = None

    def predict(self, payload):
        self.received = payload
        return self._response


def test_healthcheck_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "environment" in body


def test_generate_ad_with_mock_model():
    mock_model = _MockModel(
        [
            {
                "creative_text": "Brighten your day with the perfect mug!",
                "layout_hint": "Hero image with CTA button",
            }
        ]
    )
    app.state.model = mock_model
    client = TestClient(app)

    response = client.post(
        "/generate_ad",
        json={"title": "  Cozy Mug  ", "description": "Keep drinks warm with style"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["creative_text"] == "Brighten your day with the perfect mug!"
    assert payload["layout_hint"] == "Hero image with CTA button"
    assert payload["quality"]["word_count"] > 0
    assert mock_model.received[0]["title"] == "Cozy Mug"
    assert mock_model.received[0]["description"] == "Keep drinks warm with style"


def test_generate_ad_handles_string_prediction():
    mock_model = _MockModel(["Simple creative output"])
    app.state.model = mock_model
    client = TestClient(app)

    response = client.post(
        "/generate_ad",
        json={"title": "Lamp", "description": "LED desk lamp"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["creative_text"] == "Simple creative output"
    assert payload["layout_hint"] is None


def test_generate_ad_returns_503_when_model_missing(monkeypatch):
    def _raise():
        raise RuntimeError("no model")

    monkeypatch.setattr("src.api.main.load_generation_model", _raise)
    app.state.model = None
    client = TestClient(app)

    response = client.post(
        "/generate_ad",
        json={"title": "Item", "description": "Item description"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Generation model is unavailable"


def test_evaluate_content_quality_metrics():
    quality = evaluate_content_quality("Short sentence here. Another short sentence.")

    assert quality.word_count == 6
    assert quality.average_sentence_length == 3.0
    assert 0 <= quality.readability_score <= 100
