"""Application settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - best-effort fallback for tests
    def load_dotenv(*_: Any, **__: Any) -> bool:  # type: ignore
        """No-op fallback when python-dotenv is unavailable."""

        return False

from pydantic import BaseSettings, Field

# Load environment variables from a .env file located at the project root.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


class Settings(BaseSettings):
    """Base settings for the project, leveraging Pydantic for validation."""

    app_name: str = Field("Creative Ad Generation", description="Application name")
    environment: str = Field("development", description="Runtime environment name")
    debug: bool = Field(False, description="Enable debug mode")
    api_host: str = Field("0.0.0.0", description="Host interface for local API server")
    api_port: int = Field(8000, description="Port for local API server")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return a dict representation with aliases applied by default."""

        kwargs.setdefault("by_alias", True)
        return super().dict(*args, **kwargs)


settings = Settings()
