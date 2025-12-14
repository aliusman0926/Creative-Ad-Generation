"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from time import perf_counter
from pathlib import Path
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from ..core import configure_logging, settings

configure_logging()
logger = logging.getLogger(__name__)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests",
    labelnames=["endpoint", "method", "status_code"],
)
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Throughput of API endpoints",
    labelnames=["endpoint", "method", "status_code"],
)
REQUEST_ERRORS = Counter(
    "api_request_errors_total",
    "Count of API responses considered errors",
    labelnames=["endpoint", "method", "status_code"],
)
QUALITY_SCORE = Histogram(
    "api_content_quality_score",
    "Distribution of generated content quality scores",
    labelnames=["endpoint"],
    buckets=(0, 25, 50, 60, 70, 80, 90, 100),
)
INFLIGHT_REQUESTS = Gauge(
    "api_requests_in_progress",
    "In-flight API requests",
    labelnames=["endpoint", "method"],
)


class GenerateAdRequest(BaseModel):
    """Request body for generating creative ad content."""

    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)

    @validator("title", "description")
    def _sanitize_text(cls, value: str) -> str:  # noqa: D417 - FastAPI validator
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("must not be empty or whitespace")
        return cleaned


class ContentQuality(BaseModel):
    """Simple heuristics for monitoring generated content quality."""

    word_count: int
    average_sentence_length: float
    average_word_length: float
    readability_score: float


class GenerateAdResponse(BaseModel):
    """Response body for generated creative ad content."""

    creative_text: str
    layout_hint: str | None = None
    quality: ContentQuality


def evaluate_content_quality(text: str) -> ContentQuality:
    """Compute lightweight quality metrics for a generated text snippet."""

    normalized = text.replace("\n", " ")
    words = [token for token in normalized.split(" ") if token]
    sentences = [segment.strip() for segment in normalized.replace("!", ".").replace("?", ".").split(".") if segment.strip()]

    word_count = len(words)
    average_sentence_length = word_count / max(1, len(sentences))
    average_word_length = sum(len(word) for word in words) / max(1, word_count)

    # A bounded readability proxy: shorter sentences and words keep the score higher.
    readability_score = 100 - (average_sentence_length - 20) * 2 - (average_word_length - 5) * 10
    readability_score = max(0.0, min(100.0, readability_score))

    return ContentQuality(
        word_count=word_count,
        average_sentence_length=round(average_sentence_length, 2),
        average_word_length=round(average_word_length, 2),
        readability_score=round(readability_score, 2),
    )


def _extract_prediction_text(prediction: Any) -> tuple[str, str | None]:
    """Normalize the model prediction to text plus an optional layout hint."""

    if isinstance(prediction, str):
        return prediction.strip(), None

    if isinstance(prediction, dict):
        text = prediction.get("creative_text") or prediction.get("text") or prediction.get("prediction", "")
        hint = prediction.get("layout_hint") or prediction.get("layout")
        return text.strip(), hint.strip() if isinstance(hint, str) else None

    if isinstance(prediction, (list, tuple)) and prediction:
        return _extract_prediction_text(prediction[0])

    raise ValueError("Model prediction is empty or unsupported")


def load_generation_model() -> Any:
    """Load the latest MLflow-registered model or a packaged artifact."""

    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    if settings.mlflow_registered_model_name:
        try:
            logger.info("Loading registered model: %s", settings.mlflow_registered_model_name)
            return mlflow.pyfunc.load_model(
                model_uri=f"models:/{settings.mlflow_registered_model_name}/latest"
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Falling back from registered model: %s", exc)

    artifact_path = Path(settings.model_artifact_path)
    if artifact_path.exists():
        logger.info("Loading packaged model artifact from %s", artifact_path)
        return mlflow.pyfunc.load_model(model_uri=str(artifact_path.resolve()))

    raise RuntimeError("No available generation model")


app = FastAPI(title=settings.app_name)
app.state.model = None


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    """Simple healthcheck endpoint to verify the API is running."""

    return {
        "status": "ok",
        "environment": settings.environment,
        "debug": str(settings.debug),
    }


def _get_model() -> Any:
    model = getattr(app.state, "model", None)
    if model is not None:
        return model

    try:
        model = load_generation_model()
        app.state.model = model
        return model
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unable to load generation model: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation model is unavailable",
        ) from exc


@app.middleware("http")
async def metrics_middleware(request, call_next):  # type: ignore[override]
    """Collect latency, throughput, and error metrics for incoming requests."""

    start_time = perf_counter()
    endpoint = request.url.path
    method = request.method
    INFLIGHT_REQUESTS.labels(endpoint=endpoint, method=method).inc()
    try:
        response = await call_next(request)
    except Exception:
        duration = perf_counter() - start_time
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method, status_code="500").observe(duration)
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status_code="500").inc()
        REQUEST_ERRORS.labels(endpoint=endpoint, method=method, status_code="500").inc()
        INFLIGHT_REQUESTS.labels(endpoint=endpoint, method=method).dec()
        raise

    duration = perf_counter() - start_time
    status_code = str(response.status_code)
    REQUEST_LATENCY.labels(endpoint=endpoint, method=method, status_code=status_code).observe(duration)
    REQUEST_COUNT.labels(endpoint=endpoint, method=method, status_code=status_code).inc()
    if response.status_code >= 500:
        REQUEST_ERRORS.labels(endpoint=endpoint, method=method, status_code=status_code).inc()

    INFLIGHT_REQUESTS.labels(endpoint=endpoint, method=method).dec()
    return response


@app.post("/generate_ad", response_model=GenerateAdResponse, tags=["generation"])
def generate_ad(request: GenerateAdRequest) -> GenerateAdResponse:
    """Generate creative ad text with optional layout hints."""

    model = _get_model()
    prediction = model.predict([request.dict()])
    try:
        creative_text, layout_hint = _extract_prediction_text(prediction)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    quality = evaluate_content_quality(creative_text)
    QUALITY_SCORE.labels(endpoint="/generate_ad").observe(quality.readability_score)
    return GenerateAdResponse(creative_text=creative_text, layout_hint=layout_hint, quality=quality)


@app.get("/metrics", tags=["monitoring"])
def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""

    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
