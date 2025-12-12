"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from ..core import configure_logging, settings

configure_logging()
app = FastAPI(title=settings.app_name)


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    """Simple healthcheck endpoint to verify the API is running."""

    return {
        "status": "ok",
        "environment": settings.environment,
        "debug": str(settings.debug),
    }
