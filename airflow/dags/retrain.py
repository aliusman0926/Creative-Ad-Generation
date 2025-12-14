"""Airflow DAG to retrain the creative ad model when new data arrives."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import mlflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from mlflow.tracking import MlflowClient

from pathlib import Path
import sys

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

from src.models.train import TrainingConfig, train_model

logger = logging.getLogger(__name__)


def task_failure_alert(context):
    """Log failures for alerting hooks on retraining DAG tasks."""

    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown_dag"
    task_id = context.get("task_instance").task_id if context.get("task_instance") else "unknown_task"
    exception = context.get("exception")
    logger.error("Task %s in DAG %s failed: %s", task_id, dag_id, exception)


def _prepare_training_config(**context):
    """Load training configuration and attach the new data path."""

    data_path = context["params"].get("data_path") or os.getenv(
        "RETRAIN_DATA_PATH", "data/training/new_data.csv"
    )
    config_path = os.getenv("RETRAIN_CONFIG_PATH")
    config = TrainingConfig.from_json(config_path) if config_path else TrainingConfig()
    config.train_file = data_path
    logger.info("Prepared training config with data file: %s", data_path)
    return config


def _train_model(**context):
    """Trigger model training using the prepared configuration."""

    config: TrainingConfig = context["ti"].xcom_pull(task_ids="prepare_training_config")
    result = train_model(config)
    logger.info("Training completed with run_id=%s", result.get("run_id"))
    return result


def _promote_best_model(**context):
    """Promote the latest trained model version to production in the registry."""

    training_result = context["ti"].xcom_pull(task_ids="train_model") or {}
    registry_name = os.getenv(
        "MODEL_REGISTRY_NAME", training_result.get("registered_model", "creative-ad-generator")
    )
    if training_result.get("mlflow_tracking_uri"):
        mlflow.set_tracking_uri(training_result["mlflow_tracking_uri"])
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{registry_name}'")
    if not versions:
        raise ValueError(f"No model versions found in registry '{registry_name}' to promote")

    best_version = sorted(
        versions, key=lambda version: (version.creation_timestamp or 0, int(version.version)), reverse=True
    )[0]
    client.transition_model_version_stage(
        name=registry_name,
        version=best_version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.set_registered_model_alias(registry_name, "champion", best_version.version)
    logger.info(
        "Promoted model %s version %s to Production", registry_name, best_version.version
    )
    return {"registry_name": registry_name, "production_version": best_version.version}


def _dag() -> DAG:
    default_args = {
        "owner": "ad-generation",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "on_failure_callback": task_failure_alert,
    }

    data_path = os.getenv("RETRAIN_DATA_PATH", "data/training/new_data.csv")

    with DAG(
        dag_id="retrain_model",
        start_date=datetime(2024, 1, 1),
        schedule_interval=None,
        catchup=False,
        tags=["training", "retrain"],
        default_args=default_args,
        params={"data_path": data_path},
    ) as dag:
        wait_for_new_data = FileSensor(
            task_id="wait_for_new_data",
            filepath=data_path,
            poke_interval=60,
            timeout=60 * 60,
            mode="poke",
        )

        prepare_training_config = PythonOperator(
            task_id="prepare_training_config",
            python_callable=_prepare_training_config,
            provide_context=True,
        )

        train_model_task = PythonOperator(
            task_id="train_model",
            python_callable=_train_model,
            provide_context=True,
        )

        promote_best_model = PythonOperator(
            task_id="promote_best_model",
            python_callable=_promote_best_model,
            provide_context=True,
        )

        wait_for_new_data >> prepare_training_config >> train_model_task >> promote_best_model

    return dag


dag = _dag()
