from pathlib import Path

import pytest

pytest.importorskip("transformers")

from src.data.dataset import CreativeAdDataset, CreativeSample, load_creative_samples, split_creative_samples


def test_split_creative_samples(tmp_path: Path):
    samples = [
        CreativeSample(title=f"Title {i}", description="Desc", creative_text="Creative")
        for i in range(20)
    ]

    split = split_creative_samples(samples, val_size=0.2, test_size=0.1, seed=123)

    assert len(split.test) == 2
    assert len(split.val) == 4
    assert len(split.train) == 14
    # Ensure deterministic ordering with seed
    assert split.test[0].title != split.test[1].title


def test_load_creative_samples(tmp_path: Path):
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(
        """title,description,creative_text
Travel Backpack,Durable and water-resistant,Ready for every adventure.
Phone Case,Impact-resistant with grip,Protect your phone in style.
"""
    )

    samples = load_creative_samples(csv_path)
    assert len(samples) == 2
    assert samples[0].prompt().startswith("Product Title: Travel Backpack")


@pytest.mark.parametrize("model_name", ["hf-internal-testing/tiny-random-T5"])
def test_creative_ad_dataset_tokenization(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    samples = [
        CreativeSample(
            title="Sample", description="Desc", creative_text="Creative"
        )
    ]

    dataset = CreativeAdDataset(samples, tokenizer, max_input_length=8, max_target_length=4)
    item = dataset[0]

    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert len(dataset) == 1
