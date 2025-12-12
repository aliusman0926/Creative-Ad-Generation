"""Airflow DAG to ingest product metadata from a CSV source."""
from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data.ingest import (
    LocalObjectStore,
    normalize_record,
    parse_csv,
    persist_raw_payload,
    serialize_records_to_csv,
    validate_records,
    load_csv_from_url,
)


def _fetch_csv(**context) -> str:
    source_url = context["params"].get("source_url") or os.getenv(
        "PRODUCT_SOURCE_URL", "https://example.com/products.csv"
    )
    return load_csv_from_url(source_url)


def _validate_and_normalize(**context):
    raw_payload = context["ti"].xcom_pull(task_ids="fetch_csv")
    parsed_rows = parse_csv(raw_payload)
    normalized_rows = [normalize_record(row) for row in parsed_rows]
    validated_rows = validate_records(normalized_rows)
    return serialize_records_to_csv(validated_rows)


def _persist(**context):
    serialized = context["ti"].xcom_pull(task_ids="validate_and_normalize")
    object_store = LocalObjectStore(os.getenv("OBJECT_STORE_PATH", "data/raw"))
    return str(persist_raw_payload(serialized, object_store))


def _dag() -> DAG:
    with DAG(
        dag_id="ingest_products",
        start_date=datetime(2024, 1, 1),
        schedule_interval=None,
        catchup=False,
        tags=["products", "ingestion"],
        params={"source_url": None},
    ) as dag:
        fetch_csv = PythonOperator(task_id="fetch_csv", python_callable=_fetch_csv)

        validate_and_normalize = PythonOperator(
            task_id="validate_and_normalize",
            python_callable=_validate_and_normalize,
        )

        persist = PythonOperator(task_id="persist_raw", python_callable=_persist)

        fetch_csv >> validate_and_normalize >> persist

    return dag


dag = _dag()
