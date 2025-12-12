"""Utilities for loading and preparing creative ad generation datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
import csv
import random

try:  # pragma: no cover - import guard for optional dependency
    from transformers import PreTrainedTokenizerBase
except ModuleNotFoundError:  # pragma: no cover - handled in dataset initializer
    PreTrainedTokenizerBase = object  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class CreativeSample:
    """Single training example containing product details and creative text."""

    title: str
    description: str
    creative_text: str

    def prompt(self) -> str:
        """Format the sample into a prompt the model can consume."""

        return (
            f"Product Title: {self.title}\n"
            f"Description: {self.description}\n"
            "Creative Ad:"
        )


@dataclass
class DatasetSplit:
    """Structured split of the dataset for training, validation, and testing."""

    train: Sequence[CreativeSample]
    val: Sequence[CreativeSample]
    test: Sequence[CreativeSample]


def load_creative_samples(path: str | Path) -> list[CreativeSample]:
    """Load creative samples from a CSV file.

    The CSV must contain ``title``, ``description``, and ``creative_text`` columns.
    """

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved_path}")

    samples: list[CreativeSample] = []
    with resolved_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"title", "description", "creative_text"}
        if not required.issubset(reader.fieldnames or set()):  # type: ignore[arg-type]
            raise ValueError(
                "Dataset requires columns: title, description, creative_text"
            )

        for row in reader:
            samples.append(
                CreativeSample(
                    title=row["title"].strip(),
                    description=row["description"].strip(),
                    creative_text=row["creative_text"].strip(),
                )
            )

    return samples


def split_creative_samples(
    samples: Sequence[CreativeSample],
    *,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> DatasetSplit:
    """Split samples into train, validation, and test sets."""

    if not 0 <= val_size < 1 or not 0 <= test_size < 1:
        raise ValueError("val_size and test_size must be between 0 and 1")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1")

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    total = len(shuffled)
    test_count = int(total * test_size)
    val_count = int(total * val_size)

    test_split = shuffled[:test_count]
    val_split = shuffled[test_count : test_count + val_count]
    train_split = shuffled[test_count + val_count :]

    return DatasetSplit(train=train_split, val=val_split, test=test_split)


class CreativeAdDataset:
    """PyTorch dataset for creative ad generation."""

    def __init__(
        self,
        samples: Sequence[CreativeSample],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_input_length: int = 256,
        max_target_length: int = 64,
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
            raise ImportError("PyTorch must be installed to use CreativeAdDataset") from exc

        prompts = [sample.prompt() for sample in samples]
        targets = [sample.creative_text for sample in samples]

        model_inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                padding=True,
                truncation=True,
                max_length=max_target_length,
                return_tensors="pt",
            )

        self.input_ids = model_inputs["input_ids"]
        self.attention_mask = model_inputs["attention_mask"]
        self.labels = labels["input_ids"]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def samples_to_dicts(samples: Iterable[CreativeSample]) -> list[dict[str, str]]:
    """Convert samples to dictionaries, useful for logging and serialization."""

    return [asdict(sample) for sample in samples]
