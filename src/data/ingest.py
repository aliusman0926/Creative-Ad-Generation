"""Utilities for ingesting product metadata from CSV sources.

This module provides small helpers to fetch CSV payloads from an HTTP endpoint,
normalize/validate rows with Pydantic, and persist the raw payload into object
storage (locally stubbed for development).
"""
from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Iterable, List, Sequence
from urllib.request import urlopen

from pydantic import BaseModel, ValidationError, validator


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "sources"
# Default catalog is a curated e-commerce feed (titles + short descriptions)
# suitable for ad-creative generation experiments.
DEFAULT_SOURCE_PATH = DATA_ROOT / "ecommerce_product_catalog.csv"
DEFAULT_SOURCE_URL = DEFAULT_SOURCE_PATH.as_uri()


class ProductRecord(BaseModel):
    """Typed representation of a product row."""

    product_id: str
    name: str
    price: float
    currency: str = "USD"
    category: str | None = None
    description: str | None = None

    @validator("price")
    def price_must_be_positive(cls, value: float) -> float:
        if value < 0:
            raise ValueError("price must be non-negative")
        return round(value, 2)

    @validator("currency")
    def currency_uppercase(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("currency cannot be empty")
        return value.upper()


class LocalObjectStore:
    """Minimal stub to mimic writing to object storage.

    In production this could be backed by S3/MinIO; for local development we
    simply materialize the object to the filesystem.
    """

    def __init__(self, base_path: str | Path = "data/raw") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write_text(self, content: str, key: str) -> Path:
        destination = self.base_path / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return destination


def load_csv_from_url(url: str, timeout: int = 10) -> str:
    """Fetch CSV payload from a URL.

    The function stays dependency-light to ease execution inside Airflow workers.
    """

    with urlopen(url, timeout=timeout) as response:  # nosec B310 - controlled input
        return response.read().decode("utf-8")


def parse_csv(csv_payload: str) -> List[dict]:
    """Parse CSV content into a list of dictionaries."""

    buffer = StringIO(csv_payload)
    reader = csv.DictReader(buffer)
    return [row for row in reader]


def normalize_record(raw: dict) -> dict:
    """Normalize raw CSV rows into the schema expected by :class:`ProductRecord`."""

    def clean(value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value

    price_raw = raw.get("price") or raw.get("unit_price") or 0
    if isinstance(price_raw, str):
        cleaned_price = price_raw.replace("$", "").replace(",", "").strip()
    else:
        cleaned_price = price_raw

    normalized = {
        "product_id": clean(raw.get("product_id") or raw.get("id") or raw.get("sku") or ""),
        "name": clean(raw.get("name") or raw.get("title") or ""),
        "price": float(cleaned_price),
        "currency": (clean(raw.get("currency")) or "USD").upper(),
        "category": clean(raw.get("category") or raw.get("segment")),
        "description": clean(raw.get("description") or raw.get("short_description")),
    }
    return normalized


def validate_records(records: Iterable[dict]) -> List[ProductRecord]:
    """Validate and coerce a sequence of dictionaries into ``ProductRecord`` objects."""

    validated: List[ProductRecord] = []
    errors: list[ValidationError] = []

    for record in records:
        try:
            validated.append(ProductRecord(**record))
        except ValidationError as exc:  # pragma: no cover - exercised in tests
            errors.append(exc)

    if errors:
        # Combine errors for better visibility to task logs
        raise ValidationError.from_exception_data(
            title="ProductRecord",
            line_errors=[error for exc in errors for error in exc.raw_errors],
        )

    return validated


def serialize_records_to_csv(records: Sequence[ProductRecord]) -> str:
    """Serialize validated records back to CSV for storage."""

    if not records:
        return ""

    output = StringIO()
    fieldnames = list(records[0].dict().keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow(record.dict())
    return output.getvalue()


def persist_raw_payload(csv_payload: str, object_store: LocalObjectStore, prefix: str = "products") -> Path:
    """Persist the raw CSV payload to object storage.

    The key includes a timestamp to keep ingestion outputs immutable.
    """

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    key = f"{prefix}/{timestamp}.csv"
    return object_store.write_text(csv_payload, key)


def ingest_from_url(url: str, object_store: LocalObjectStore) -> Path:
    """Fetch, validate, and persist product metadata from a CSV URL."""

    raw_payload = load_csv_from_url(url)
    parsed_rows = parse_csv(raw_payload)
    normalized_rows = [normalize_record(row) for row in parsed_rows]
    validated_rows = validate_records(normalized_rows)
    serialized = serialize_records_to_csv(validated_rows)
    return persist_raw_payload(serialized, object_store)
