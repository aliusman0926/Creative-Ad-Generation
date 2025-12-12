from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from src.models.train import TrainingConfig, train_model


def test_train_smoke(tmp_path: Path):
    dataset_path = tmp_path / "pairs.csv"
    dataset_path.write_text(
        """title,description,creative_text
Reusable Straw,Steel straw with case,Skip the plastic and sip sustainably.
Desk Lamp,LED lamp with dimmer,Brighten your workspace with ease.
Yoga Mat,Non-slip with cushion,Stretch into comfort on every pose.
"""
    )

    tracking_dir = tmp_path / "mlruns"
    config = TrainingConfig(
        model_name="hf-internal-testing/tiny-random-T5",
        model_type="seq2seq",
        train_file=str(dataset_path),
        output_dir=str(tmp_path / "artifacts"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_steps=1,
        eval_steps=1,
        val_size=0.2,
        test_size=0.0,
        mlflow_tracking_uri=f"file://{tracking_dir}",
        mlflow_registered_model_name=None,
        max_train_steps=1,
    )

    result = train_model(config)

    assert "metrics" in result
    assert Path(result["model_path"]).exists()
