"""Airflow DAG to run scheduled batch inference jobs for creative ad generation."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
import sys
from pathlib import Path

# --- ensure project root (the folder that contains "src") is on sys.path ---
DAG_FILE = Path(__file__).resolve()

project_root = None
for p in [DAG_FILE.parent, *DAG_FILE.parents]:
    if (p / "src").is_dir():
        project_root = p
        break

if project_root is not None:
    sys.path.insert(0, str(project_root))
else:
    # Fallback for common Airflow docker layout if you mount src to /opt/airflow/src
    sys.path.insert(0, "/opt/airflow")


from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data.ingest import LocalObjectStore

logger = logging.getLogger(__name__)


def task_failure_alert(context):
    """Log details about task failures for observability and alerting hooks."""

    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown_dag"
    task_id = context.get("task_instance").task_id if context.get("task_instance") else "unknown_task"
    exception = context.get("exception")
    logger.error("Task %s in DAG %s failed: %s", task_id, dag_id, exception)


def _load_requests(**context):
    """Load batch inference requests from a JSON file or provide a fallback sample."""

    request_path = Path(os.getenv("BATCH_REQUESTS_PATH", "data/batch/requests.json"))
    if request_path.exists():
        payload = json.loads(request_path.read_text())
        logger.info("Loaded %s requests from %s", len(payload), request_path)
        return payload

    fallback = [
        {
            "title": "Stainless Steel Water Bottle",
            "description": "Keeps drinks cold for 24 hours and hot for 12, with a leak-proof lid.",
        }
    ]
    logger.warning("Request file not found at %s; using fallback sample", request_path)
    return fallback


def _generate_creatives(**context):
    """Generate creative ad copy for each request."""

    requests: list[dict] = context["ti"].xcom_pull(task_ids="load_requests") or []
    generated: list[dict[str, str]] = []
    for request in requests:
        title = request.get("title", "Untitled Product")
        description = request.get("description", "")
        creative_text = (
            f"Discover {title}! {description} Get yours today and enjoy the difference."
        )
        generated.append({
            "title": title,
            "description": description,
            "creative_text": creative_text,
        })

    logger.info("Generated creatives for %s requests", len(generated))
    return generated


def _persist_outputs(**context):
    """Persist generated creatives to object storage and a lightweight SQLite database."""

    outputs: list[dict[str, str]] = context["ti"].xcom_pull(task_ids="generate_creatives") or []
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    object_store = LocalObjectStore(os.getenv("INFERENCE_STORE_PATH", "data/batch/inference"))
    object_key = f"runs/{timestamp}.json"
    object_path = object_store.write_text(json.dumps(outputs, indent=2), object_key)
    logger.info("Persisted %s generated creatives to %s", len(outputs), object_path)

    db_path = Path(os.getenv("INFERENCE_DB_PATH", "data/batch/inference/results.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                creative_text TEXT NOT NULL
            )
            """
        )
        cursor.executemany(
            """
            INSERT INTO batch_results (run_timestamp, title, description, creative_text)
            VALUES (?, ?, ?, ?)
            """,
            [
                (timestamp, item.get("title", ""), item.get("description", ""), item.get("creative_text", ""))
                for item in outputs
            ],
        )
        connection.commit()
    finally:
        connection.close()

    logger.info("Persisted outputs to SQLite database at %s", db_path)
    return {"object_store_path": str(object_path), "db_path": str(db_path), "records": len(outputs)}


def _dag() -> DAG:
    default_args = {
        "owner": "ad-generation",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "on_failure_callback": task_failure_alert,
    }

    with DAG(
        dag_id="batch_inference",
        start_date=datetime(2024, 1, 1),
        schedule_interval="@hourly",
        catchup=False,
        tags=["inference", "batch"],
        default_args=default_args,
    ) as dag:
        load_requests = PythonOperator(
            task_id="load_requests",
            python_callable=_load_requests,
            provide_context=True,
        )

        generate_creatives = PythonOperator(
            task_id="generate_creatives",
            python_callable=_generate_creatives,
            provide_context=True,
        )

        persist_outputs = PythonOperator(
            task_id="persist_outputs",
            python_callable=_persist_outputs,
            provide_context=True,
        )

        load_requests >> generate_creatives >> persist_outputs

    return dag


dag = _dag()
