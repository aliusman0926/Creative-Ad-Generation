"""Logging configuration utilities for the project."""

from __future__ import annotations

import logging
import logging.config
from typing import Any, Dict

LOG_LEVEL = "INFO"


def default_logging_dict() -> Dict[str, Any]:
    """Return a default logging configuration dictionary.

    The configuration uses a basic console handler with a concise format that includes
    timestamps, logger names, log levels, and messages.
    """

    formatter = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": formatter,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": LOG_LEVEL,
            }
        },
        "root": {
            "handlers": ["console"],
            "level": LOG_LEVEL,
        },
    }


def configure_logging(logging_config: Dict[str, Any] | None = None) -> None:
    """Configure application logging with the provided configuration.

    Args:
        logging_config: Optional logging configuration dictionary. If omitted, a
            sensible default configuration is applied.
    """

    config = logging_config or default_logging_dict()
    logging.config.dictConfig(config)


configure_logging()
