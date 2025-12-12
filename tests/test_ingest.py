import pytest

from src.data.ingest import (
    DEFAULT_SOURCE_URL,
    LocalObjectStore,
    ProductRecord,
    ingest_from_url,
    normalize_record,
    parse_csv,
    serialize_records_to_csv,
    validate_records,
)


def test_parse_csv_and_normalize():
    csv_payload = "product_id,name,price,currency\n001, Sample ,12.34,usd\n002,Another,5.50,EUR"
    rows = parse_csv(csv_payload)
    normalized = [normalize_record(row) for row in rows]

    assert normalized[0]["product_id"] == "001"
    assert normalized[0]["name"] == "Sample"
    assert normalized[0]["price"] == 12.34
    assert normalized[0]["currency"] == "USD"


def test_validate_records_enforces_schema():
    normalized = [
        {"product_id": "001", "name": "Widget", "price": 10.0, "currency": "usd"},
        {"product_id": "002", "name": "Thing", "price": 5.0, "currency": "EUR"},
    ]

    validated = validate_records(normalized)
    assert all(isinstance(record, ProductRecord) for record in validated)
    assert validated[0].currency == "USD"
    assert validated[1].currency == "EUR"

    with pytest.raises(Exception):
        validate_records([{"product_id": "003", "name": "Bad", "price": -1, "currency": "usd"}])


def test_serialize_and_store(tmp_path):
    records = [
        ProductRecord(product_id="a", name="Alpha", price=1.23, currency="USD"),
        ProductRecord(product_id="b", name="Beta", price=4.56, currency="EUR"),
    ]

    csv_output = serialize_records_to_csv(records)
    object_store = LocalObjectStore(tmp_path / "bucket")
    destination = object_store.write_text(csv_output, "products/test.csv")

    assert destination.exists()
    content = destination.read_text(encoding="utf-8")
    assert "product_id" in content
    assert "Alpha" in content


def test_ingest_from_real_source(tmp_path):
    object_store = LocalObjectStore(tmp_path / "bucket")
    destination = ingest_from_url(DEFAULT_SOURCE_URL, object_store)

    assert destination.exists()
    csv_text = destination.read_text(encoding="utf-8")
    assert "AirPods Pro" in csv_text
    assert "description" in csv_text.splitlines()[0]
    assert "product_id" in csv_text.splitlines()[0]