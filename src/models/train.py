"""Training entrypoint for creative ad text generation models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import mlflow
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.data.dataset import CreativeAdDataset, CreativeSample, load_creative_samples, split_creative_samples


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning a text generation model."""

    model_name: str = "google/flan-t5-small"
    output_dir: str = "artifacts/model"
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    weight_decay: float = 0.01
    warmup_steps: int = 0
    seed: int = 42
    max_input_length: int = 256
    max_target_length: int = 64
    val_size: float = 0.1
    test_size: float = 0.1
    logging_steps: int = 10
    eval_steps: int = 50
    save_strategy: str = "epoch"
    model_type: str = "seq2seq"  # "seq2seq" or "causal"
    train_file: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "creative-ad-generation"
    mlflow_registered_model_name: Optional[str] = "creative-ad-generator"
    run_name: str = "creative-ad-training"
    max_train_steps: Optional[int] = None

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)


def _load_model_and_tokenizer(config: TrainingConfig):
    if config.model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _load_samples(config: TrainingConfig) -> list[CreativeSample]:
    if config.train_file:
        return load_creative_samples(config.train_file)

    # Fallback minimal dataset for exploratory runs.
    return [
        CreativeSample(
            title="Stainless Steel Water Bottle",
            description="Keeps drinks cold for 24 hours and hot for 12, with a leak-proof lid.",
            creative_text="Stay hydrated in style with our stainless steel bottle—your drink's best friend!",
        ),
        CreativeSample(
            title="Wireless Noise-Canceling Earbuds",
            description="Immersive sound with adaptive noise cancellation and 24-hour battery life.",
            creative_text="Unplug, unwind, and let the soundtrack of your day shine with our earbuds.",
        ),
        CreativeSample(
            title="Ergonomic Office Chair",
            description="Adjustable lumbar support, breathable mesh, and smooth-rolling wheels.",
            creative_text="Give your back a break—comfort meets productivity in our ergonomic chair.",
        ),
    ]


def train_model(config: TrainingConfig) -> dict[str, Any]:
    """Train the model and log results to MLflow."""

    if config.mlflow_tracking_uri:
        tracking_uri = config.mlflow_tracking_uri
        if tracking_uri.startswith("file://") or "://" not in tracking_uri:
            tracking_uri = Path(tracking_uri.replace("file://", "")).resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    samples = _load_samples(config)
    split = split_creative_samples(samples, val_size=config.val_size, test_size=config.test_size, seed=config.seed)

    model, tokenizer = _load_model_and_tokenizer(config)

    train_dataset = CreativeAdDataset(split.train, tokenizer, max_input_length=config.max_input_length, max_target_length=config.max_target_length)
    eval_dataset = CreativeAdDataset(split.val, tokenizer, max_input_length=config.max_input_length, max_target_length=config.max_target_length)

    data_collator = (
        DataCollatorForSeq2Seq(tokenizer, model=model)
        if config.model_type == "seq2seq"
        else DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        report_to=["none"],
        seed=config.seed,
        max_steps=config.max_train_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    with mlflow.start_run(run_name=config.run_name) as run:
        mlflow.log_params(asdict(config))
        train_result = trainer.train()
        metrics = train_result.metrics
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

        mlflow.log_metrics(metrics)

        model_dir = Path(config.output_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
        mlflow.log_artifacts(model_dir)

        if config.mlflow_registered_model_name:
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=config.mlflow_registered_model_name,
            )

    return {
        "metrics": metrics,
        "run_id": run.info.run_id,
        "model_path": str(Path(config.output_dir).resolve()),
    }


def main(config_path: Optional[str] = None) -> None:
    """Run training using either defaults or a JSON configuration file."""

    config = TrainingConfig.from_json(config_path) if config_path else TrainingConfig()
    train_model(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a creative text generator.")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file", default=None)
    args = parser.parse_args()
    main(args.config)